import copy
import json
from pathlib import Path

import pytest

from driversi.data.catalog import geometry_caption as g
from driversi.data.catalog import bailian_caption as client
from driversi.rendering.whitebox.exporter import sha256
from tests.unit.rendering.test_scene_description_v2 import raw_and_bundle


def appearance():
    return {**{k: {'text': 'unknown', 'evidence_ids': []} for k in g.FIELDS},
            'distinctive_features': []}


def test_history_no_future_tracks_or_motion(tmp_path):
    _, bundle = raw_and_bundle(tmp_path)
    before = g.compile_geometry(bundle, 'retrieval_history')
    changed = copy.deepcopy(bundle)
    for f in changed['frames'][21:]:
        f['objects'] = []
        f['ego_rear_axle_to_local'][0][3] += 900
    assert g.compile_geometry(changed, 'retrieval_history') == before
    assert before['frame_count'] == 21
    assert max(s['time_relative_s'] for t in before['actor_tracks'].values() for s in t) == 0
    assert 'b' not in before['actor_tracks']
    assert g.compile_geometry(bundle, 'video_clip')['frame_count'] == 71


def test_fixed_anchor_coordinates_heading_not_vlm(tmp_path):
    _, bundle = raw_and_bundle(tmp_path)
    geometry = g.compile_geometry(bundle, 'video_clip')
    sample = next(x for x in geometry['actor_tracks']['a'] if x['time_relative_s'] == 0)
    assert sample['position_in_anchor_ego_m'][0] == 16
    assert sample['heading_in_anchor_ego_degrees'] == pytest.approx(179.8, abs=.01)
    assert all(x['physical_appearance']=='unknown_unbound' for x in geometry['actor_summaries'])


@pytest.mark.parametrize('extra', ['actor_positions', 'motion', 'trajectories', 'ego_action'])
def test_vlm_cannot_supply_geometry(extra):
    value = appearance(); value[extra] = []
    with pytest.raises(ValueError, match='kinematics forbidden'):
        g.parse(json.dumps(value), ['CAM_F0_f+0'])


def test_actor_prose_quarantined_not_released():
    value = appearance()
    value['setting'] = {'text':'A red car moving left.', 'evidence_ids':['CAM_F0_f+0']}
    value['vegetation'] = {'text':'Mature leafy trees.', 'evidence_ids':['CAM_F0_f+0']}
    _, accepted, bad = g.parse(json.dumps(value), ['CAM_F0_f+0'])
    assert set(accepted) == {'vegetation'}
    assert len(bad)==1 and bad[0]['path']=='setting'


def test_evidence_exact_and_unknown_consistent():
    value = appearance();value['setting']={'text':'Brick buildings.','evidence_ids':['CAM_F0_f+0 t=0']}
    with pytest.raises(ValueError):g.parse(json.dumps(value), ['CAM_F0_f+0'])
    value['setting']['evidence_ids']=[]
    with pytest.raises(ValueError):g.parse(json.dumps(value), ['CAM_F0_f+0'])


def test_decoder_keeps_geometry_and_never_auto_approves(tmp_path):
    _, bundle = raw_and_bundle(tmp_path)
    source=tmp_path/'source.json';client.write(source,bundle)
    geo=tmp_path/'geo.json';client.write(geo,g.compile_geometry(bundle,'video_clip'))
    job={'geometry_file':str(geo),'geometry_sha256':sha256(geo),'bundle':str(source),
         'source_sha256':sha256(source),'scope':'video_clip','evidence':[]}
    response={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(appearance())}}]}
    result=g.result_from_response(job,'fixture',response)
    assert result['trainable'] is False
    assert result['draft_prompts']['retrieval_text'] is None
    assert result['geometry']['sha256']==sha256(geo)
    assert 'fixed' in result['draft_prompts']['video_positive_prompt'].lower()
    geo.write_text('{}')
    with pytest.raises(ValueError,match='changed'):g.result_from_response(job,'fixture',response)



@pytest.mark.parametrize('text', ['Two sedans accelerate.', 'A person runs.', 'Three lanes merge.', 'The signal is red.'])
def test_scope_screen_flags_review_examples(text):
    v=appearance();v['setting']={'text':text,'evidence_ids':['CAM_F0_f+0']}
    _,accepted,warnings=g.parse(json.dumps(v),['CAM_F0_f+0'])
    assert 'setting' not in accepted and warnings


def test_uncertain_terminal_sample_is_not_claimed_observed(tmp_path):
    _,bundle=raw_and_bundle(tmp_path)
    bundle['frames'][-1]['objects'][0]['sample_basis']='previous_observation_held_presence_uncertain'
    geo=g.compile_geometry(bundle,'video_clip')
    a=next(x for x in geo['actor_summaries']if x['actor_id']=='a')
    assert 'last_observed_support_s' not in a
    assert a['uncertain_held_sample_count']==1
    assert 'uncertain' in a['terminal_presence_evidence']
    assert geo['complete_video_condition'] is False
