"""Ref2VA contract tests: references, camera appearance scope and frozen delivery."""
import copy
import json
from pathlib import Path
import zipfile

import pytest

from driversi.rendering.whitebox.conditioning import PALETTE
from driversi.rendering.whitebox.delivery import MEMBERS, package_delivery
from driversi.rendering.whitebox.minimax_h3 import (
    CAMERAS, FIELDS, build_document, export_document, validate_request,
)

TOKEN = '0123456789abcdef'


@pytest.fixture
def source(tmp_path):
    bundle = tmp_path / 'bundle'; video = bundle / 'source-renders'; video.mkdir(parents=True)
    for camera in CAMERAS:
        (video / (camera + '.mp4')).write_bytes(('fixture-video-' + camera).encode())
    (bundle / 'source.json').write_text(json.dumps(dict(token=TOKEN, source_split='navtrain',
        role='source_log', palette=dict(colors=PALETTE))))
    review = tmp_path / 'review.json'
    review.write_text(json.dumps(dict(token=TOKEN, status='human_reviewed_for_generation_trial', claims=[
        dict(text='Pale building facade.', evidence_ids=['CAM_F0_f0']),
        dict(text='Trees and dark fencing.', evidence_ids=['CAM_L0_f0'])])))
    return bundle, review


def test_four_individual_requests_native_format_and_camera_specific_prose(source):
    bundle, review = source
    document = build_document(bundle, review, TOKEN)
    assert document['source']['fps'] == 10 and document['target']['native_fps'] == 24
    assert not document['multi_view']['joint_consistency_verified']
    assert not document['use']['generation_executed'] and not document['use']['training_admitted']
    jobs = {j['camera_id']: j for j in document['jobs']}
    assert set(jobs) == set(CAMERAS)
    for camera, job in jobs.items():
        assert tuple(job['prompt_fields']) == FIELDS
        assert len(job['request']['conditions']) == 1
        assert job['request']['conditions'][0]['uri'].endswith('/' + camera + '.mp4')
        assert job['request']['target']['duration_seconds'] == 7
        assert 'fps' not in job['request']['target']
        validate_request(job['request'])
    assert 'Pale building facade.' in jobs['CAM_F0']['request']['prompt']
    assert 'Pale building facade.' not in jobs['CAM_L0']['request']['prompt']
    assert 'No camera-specific appearance claim was verified' in jobs['CAM_R0']['request']['prompt']


@pytest.mark.parametrize('case', ['extra_video', 'undefined_reference', 'wrong_order', 'ten_fps_target'])
def test_unbound_references_or_wrong_native_contract_fail(source, case):
    bundle, review = source
    request = copy.deepcopy(build_document(bundle, review, TOKEN)['jobs'][0]['request'])
    if case == 'extra_video': request['conditions'] *= 4
    elif case == 'undefined_reference': request['prompt'] += '\nUse <Picture 1>.'
    elif case == 'wrong_order': request['prompt'] = request['prompt'].replace('subject_definitions:', 'definitions:')
    elif case == 'ten_fps_target': request['target']['fps'] = 10
    with pytest.raises(ValueError): validate_request(request)


def test_semantic_palette_change_requires_new_profile(source):
    bundle, review = source
    path = bundle / 'source.json'; data = json.loads(path.read_text()); data['palette']['colors']['vehicle'] = '#000000'; path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='palette'): build_document(bundle, review, TOKEN)


def test_h3_internal_json_stays_outside_five_file_archive_and_reuses(source, tmp_path):
    bundle, review = source
    document, prompt = export_document(bundle, tmp_path / 'h3', TOKEN, review)
    receipt = package_delivery(bundle, tmp_path / 'download', TOKEN, review, prompt_text=prompt)
    with zipfile.ZipFile(receipt['archive']) as z:
        assert sorted(z.namelist()) == sorted(MEMBERS)
        assert all(j['camera_id'] in z.read(MEMBERS[-1]).decode() for j in document['jobs'])
        assert len(z.namelist()) == 5
    before = (tmp_path / 'h3/condition.json').stat().st_mtime_ns
    assert export_document(bundle, tmp_path / 'h3', TOKEN, review) == (document, prompt)
    assert (tmp_path / 'h3/condition.json').stat().st_mtime_ns == before
    assert package_delivery(bundle, tmp_path / 'download', TOKEN, review, prompt_text=prompt)['status'] == 'reused'


def test_changed_seed_does_not_overwrite_frozen_conditions(source, tmp_path):
    bundle, review = source
    export_document(bundle, tmp_path / 'h3', TOKEN, review, seed=0)
    with pytest.raises(ValueError, match='new output version'):
        export_document(bundle, tmp_path / 'h3', TOKEN, review, seed=1)


def test_compound_multicamera_landmarks_are_withheld_until_split(source):
    bundle, review = source
    data = json.loads(review.read_text())
    data['claims'] += [dict(text='Curved glass facade and dark grid facade.',
        evidence_ids=['CAM_F0_f-3', 'CAM_B0_f-3'])]
    review.write_text(json.dumps(data))
    jobs = {j['camera_id']: j for j in build_document(bundle, review, TOKEN)['jobs']}
    for camera in ('CAM_F0', 'CAM_B0'):
        assert 'Curved glass facade and dark grid facade.' not in jobs[camera]['request']['prompt']
        assert len(jobs[camera]['withheld_appearance_claims']) == 1
    data['claims'] = [dict(text='Curved glass facade.', evidence_ids=['CAM_F0_f-3', 'CAM_F0_f+0']),
                      dict(text='Dark grid facade.', evidence_ids=['CAM_B0_f-3'])]
    review.write_text(json.dumps(data))
    jobs = {j['camera_id']: j for j in build_document(bundle, review, TOKEN)['jobs']}
    assert 'Curved glass facade.' in jobs['CAM_F0']['request']['prompt']
    assert 'Dark grid facade.' not in jobs['CAM_F0']['request']['prompt']
    assert 'Dark grid facade.' in jobs['CAM_B0']['request']['prompt']
    assert 'Curved glass facade.' not in jobs['CAM_B0']['request']['prompt']


def test_malformed_camera_evidence_fails(source):
    bundle, review = source
    data = json.loads(review.read_text()); data['claims'][0]['evidence_ids'] = ['unknown_camera']
    review.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='evidence ID'): build_document(bundle, review, TOKEN)


def test_legacy_h3_document_shape_remains_unchanged(source):
    bundle,review=source
    old=build_document(bundle,review,TOKEN)
    assert 'display_profile' not in old['source']
    assert all('explicitly planar condition' not in job['request']['prompt'] for job in old['jobs'])
    raw=json.loads((bundle/'source.json').read_text())
    raw['display_profile']='planar_consistent_v1'
    (bundle/'source.json').write_text(json.dumps(raw))
    new=build_document(bundle,review,TOKEN)
    assert new['source']['display_profile']=='planar_consistent_v1'
    assert all('explicitly planar condition' in job['request']['prompt'] for job in new['jobs'])
