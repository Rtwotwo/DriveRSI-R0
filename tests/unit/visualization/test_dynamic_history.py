import json
from pathlib import Path

from driversi.visualization.dynamic_history import project


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def plan():
    return {'model': 'chainflow_stage1', 'sources': [{'id': 'train-a', 'branches': [{'id': str(i)} for i in range(6)]},
                                                  {'id': 'train-b', 'branches': [{'id': str(i)} for i in range(6, 12)]}],
            'limits': {'minimum_samples': 4, 'minimum_sources': 2},
            'training': {'target_token': 'MUST_NOT_EXPORT_TEST_ID'}}


def test_plan_is_not_reported_as_generated_or_trained(tmp_path):
    write(tmp_path / 'plan.json', plan())
    view = project(tmp_path)
    assert view['planned_reused_rollouts'] == 12
    assert view['physical_rollouts_reused'] == view['new_teacher_rollouts'] == 0
    assert view['admitted_samples'] == 0 and view['pending_samples'] == 12
    assert view['training']['native_training_started'] is False
    assert view['training']['status'] == 'not_started'
    assert 'MUST_NOT_EXPORT_TEST_ID' not in json.dumps(view)


def test_asset_phase_visible_before_source_settles(tmp_path):
    write(tmp_path / 'plan.json', plan())
    root = tmp_path / 'sources/train-a/asset/execution/reconstruction'
    write(root / 'status.json', {'status': 'joint_native_training', 'eligible_objects': 3})
    write(root / 'point-cloud-progress.json', {'completed_frames': 23})
    view = project(tmp_path)
    assert view['sources'][0]['phase'] == 'joint_native_training'
    assert view['sources'][0]['prepared_frames'] == 23
    assert view['training']['native_training_started'] is False


def test_reuse_rejections_and_training_preparation_distinguished(tmp_path):
    write(tmp_path / 'plan.json', plan())
    write(tmp_path / 'progress.json', {'status': 'running', 'admitted_samples': 1,
          'physical_rollouts_reused': 3, 'new_teacher_rollouts': 0, 'training_started': True,
          'sources': [{'id': 'train-a', 'asset_status': 'completed', 'branches': [
              {'id': '0', 'status': 'training_sample_ready'},
              {'id': '1', 'status': 'rejected', 'reason': 'visible_object_unsupported'},
              {'id': '2', 'status': 'unknown'}]}]})
    view = project(tmp_path)
    assert view['admitted_samples'] == view['rejected_samples'] == view['unknown_samples'] == 1
    assert view['physical_rollouts_reused'] == 3 and view['new_teacher_rollouts'] == 0
    assert view['training']['status'] == 'preparing'
    assert not view['training']['native_training_started']
    assert view['history_images_per_sample'] == 16 and view['future_rgb_per_sample'] == 0
    assert view['training']['treatment'] == 'synthetic_1_real_7'


def test_actual_optimizer_progress_controls_training_claim(tmp_path):
    write(tmp_path / 'plan.json', plan())
    write(tmp_path / 'progress.json', {'status': 'running', 'training_started': True})
    write(tmp_path / 'training/execution/tasks/pair/control_train/native/training-progress.json',
          {'total_updates': 3})
    view = project(tmp_path)
    assert view['training']['native_training_started']
    assert view['training']['observed_updates_per_arm'] == {'control': 3, 'treatment': 0}


def test_anchor_counts_and_prior_asset_adoption(tmp_path):
    value = plan()
    value['gates'] = {'image_contract': 'anchor_rgb_history_future_states_v1'}
    old = tmp_path / 'old_asset'
    write(old / 'job.json', {'output': str(old)})
    write(old / 'status.json', {'status': 'joint_native_training', 'eligible_objects': 4})
    write(old / 'point-cloud-progress.json', {'completed_frames': 23})
    value['sources'][0]['prebuilt_asset_job'] = str(old / 'job.json')
    write(tmp_path / 'plan.json', value)
    view = project(tmp_path)
    assert view['input_rgb_per_sample'] == 4 and view['future_rgb_per_sample'] == 0
    assert view['history_state_frames'] == 4 and view['future_numeric_frames'] == 10
    assert view['rgb_scope'] == 'anchor_only'
    assert view['sources'][0]['phase'] == 'joint_native_training'
    assert view['sources'][0]['prepared_frames'] == 23


def test_multiple_rejection_reasons_remain_separate(tmp_path):
    write(tmp_path / 'plan.json', plan())
    write(tmp_path / 'progress.json', {'sources': [{'id': 'train-a', 'branches': [
        {'id': '0', 'status': 'rejected', 'reason': ['missing_object', 'binding_mismatch']},
        {'id': '1', 'status': 'rejected', 'reason': ['missing_object']}]}]})
    view = project(tmp_path)
    assert view['rejected_samples'] == 2
    assert view['sources'][0]['rejections'] == {'missing_object': 2, 'binding_mismatch': 1}
