import copy
from pathlib import Path

import pytest

from driversi.rendering.gaussian import dynamic_reconstruction as dr


def metadata():
    return {'log-0': {'frame_infos': [{'track_tokens': ['car', 'far', 'human', 'bike'],
                                      'gt_names': ['vehicle', 'vehicle', 'pedestrian', 'bicycle']}]}}


def points():
    return {'objects': [{'video_token': 'log-0', 'token': token, 'measured_colored_points': count}
                        for token, count in [('car', 100), ('far', 99), ('human', 200), ('bike', 300)]]}


def test_native_support_is_measured_and_class_specific():
    rows = {r['token']: r for r in dr.classify_objects(metadata(), points())}
    assert rows['car']['supported']
    assert rows['far']['reason'] == 'insufficient_measured_points'
    assert not rows['human']['supported'] and not rows['bike']['supported']
    assert rows['human']['reason'] == 'unsupported_actor_class'


def test_missing_object_cloud_is_explicit():
    rows = dr.classify_objects(metadata(), {'objects': []})
    assert len(rows) == 4 and not any(r['supported'] for r in rows)


def test_identity_class_conflict_rejected():
    value = metadata()
    value['log-0']['frame_infos'].append({'track_tokens': ['car'], 'gt_names': ['pedestrian']})
    with pytest.raises(ValueError, match='changes class'):
        dr.classify_objects(value, points())


def test_output_cannot_escape_experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, 'project_root', lambda: tmp_path)
    assert dr._owned(tmp_path / 'exp/run/asset') == tmp_path / 'exp/run/asset'
    with pytest.raises(ValueError):
        dr._owned(tmp_path / 'dataset')
    with pytest.raises(ValueError):
        dr._owned(tmp_path / 'exp')


def asset():
    torch = pytest.importorskip('torch')
    return {'background': {'config': {}, 'state_dict': {}}, 'skybox': {'config': {}, 'state_dict': {}},
            'rigid_object_car': {'config': {'type': 'MirroredRigidSubModel'},
                                 'state_dict': {'gauss_params.means': torch.ones((3, 3))}}}


def save(tmp_path, value):
    import torch
    path = tmp_path / 'portable.ckpt'
    torch.save(value, path)
    return path


def test_portable_has_real_named_nonempty_nodes(tmp_path):
    path = save(tmp_path, asset())
    rows = dr.validate_portable(path, [{'token': 'car', 'supported': True}])
    assert rows == [{'token': 'car', 'node': 'rigid_object_car', 'gaussians': 3,
                     'native_type': 'MirroredRigidSubModel'}]


@pytest.mark.parametrize('kind', ['missing', 'extra', 'no_background', 'nonfinite', 'empty', 'wrong_type'])
def test_invalid_binding_fails_closed(tmp_path, kind):
    torch = pytest.importorskip('torch')
    value = asset()
    if kind == 'missing':
        value.pop('rigid_object_car')
    elif kind == 'extra':
        value['rigid_object_unmeasured'] = copy.deepcopy(value['rigid_object_car'])
    elif kind == 'no_background':
        value.pop('background')
    elif kind == 'nonfinite':
        value['rigid_object_car']['state_dict']['gauss_params.means'][0, 0] = float('nan')
    elif kind == 'empty':
        value['rigid_object_car']['state_dict']['gauss_params.means'] = torch.zeros((0, 3))
    else:
        value['rigid_object_car']['config']['type'] = 'PaintedBox'
    with pytest.raises(ValueError):
        dr.validate_portable(save(tmp_path, value), [{'token': 'car', 'supported': True}])


def test_no_eligible_objects_cannot_claim_completion(tmp_path):
    with pytest.raises(ValueError):
        dr.validate_portable(save(tmp_path, asset()), [])


def test_native_completion_reconciles_without_gpu_training(tmp_path):
    torch = pytest.importorskip('torch')
    path = tmp_path / 'step-000030000.ckpt'
    assert not dr.native_training_complete(tmp_path, 30001)
    torch.save({'step': 30000, 'pipeline': {'weight': torch.ones(1)}}, path)
    assert dr.native_training_complete(tmp_path, 30001)
    torch.save({'step': 30000, 'pipeline': {'weight': torch.ones(1)}, 'optimizers': {}}, path)
    assert not dr.native_training_complete(tmp_path, 30001)


def test_bundle_requires_bound_measured_terrain(tmp_path):
    np = pytest.importorskip('numpy')
    import json
    with pytest.raises(ValueError, match='terrain outputs'):
        dr.terrain_outputs(tmp_path)
    folder = tmp_path / 'road_height_map'
    folder.mkdir()
    source = tmp_path / 'measured.pcd'
    source.write_bytes(b'measured returns')
    np.save(folder / 'road_height_map.npy', np.array([[1., float('nan')]]))
    (folder / 'sim2.json').write_text(json.dumps({'s': 4., 'R': [[1, 0], [0, 1]], 't': [0, 0]}))
    (folder / 'provenance.json').write_text(json.dumps({'method': 'measured_road_cells_v1',
        'interpolation': False, 'source': {str(source): dr.digest(source)}}))
    assert len(dr.terrain_outputs(tmp_path)) == 3
    source.write_bytes(b'changed')
    with pytest.raises(ValueError, match='source changed'):
        dr.terrain_outputs(tmp_path)
