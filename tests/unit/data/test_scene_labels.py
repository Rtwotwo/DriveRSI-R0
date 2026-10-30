"""Information scope, unknowns, evidence and actor continuity of scene labels."""
import copy
import json
import pytest

from driversi.data.catalog.scene_labels import parse, assemble, measurements, SCOPE


def fixture():
    context={'actor_tracks':[{'actor':'actor_0001','represented_intervals_s':[[0,2],[4,7]]}],
        'ego_motion':{},'map_context':{},'source_quality':{}}
    evidence=[dict(id='CAM_F0_f+0',camera='CAM_F0',relative_time_s=0)]
    claim=dict(text_zh='可见城市道路。',evidence_ids=['CAM_F0_f+0'],fact_ids=[],certainty='supported')
    value=dict(summary=claim,facets=[dict(axis='road_context',label='urban_street',**claim)],
        events=[],distinctive_features=[],unknowns=[dict(field='lane_membership',reason_zh='未提供车道关联。')],proposed_new_tags=[])
    return context,evidence,value


def test_unknown_and_empty_events_are_valid_without_forcing_hard_case():
    c,e,v=fixture()
    assert parse(json.dumps(v),c,e)==v


def test_future_scene_label_is_explicitly_not_history_input_or_released_pool():
    c,e,v=fixture();b=dict(token='a',source_log='train-log',source_split='navtrain',role='source_log')
    out=assemble(b,c,v,e,'test-model')
    assert out['information_scope']==dict(scope=SCOPE,future_included=True,online_policy_input=False)
    assert not out['draft_retrieval']['indexable'] and not out['draft_clustering']['pool_write_allowed']
    assert not out['review']['approved_for_batch']
    assert out['draft_clustering']['cluster_id'] is None


def test_tentative_claims_remain_separate_from_default_tags_and_text():
    c,e,v=fixture();v['facets'][0]['certainty']='tentative'
    v['facets'][0]['text_zh']='可能有道路标线。'
    out=assemble(dict(token='a',source_log='train',source_split='navtrain',role='source_log'),c,v,e,'test-model')
    assert out['draft_retrieval']['tag_keys']==[]
    assert '可能有道路标线' not in out['draft_retrieval']['text_zh']
    assert out['draft_clustering']['semantic_tag_keys']==[]
    assert out['draft_clustering']['tentative_tag_keys']==['road_context:urban_street']


@pytest.mark.parametrize('fault',['image','fact','tag','duplicate_tag','unsupported_field','empty_evidence'])
def test_unbound_references_and_model_failure_metadata_rejected(fault):
    c,e,v=fixture()
    if fault=='image':v['summary']['evidence_ids']=['CAM_R0_f+999']
    elif fault=='fact':v['summary']['fact_ids']=['test_score']
    elif fault=='tag':v['facets'][0]['label']='model_failed_to_yield'
    elif fault=='duplicate_tag':v['facets'].append(copy.deepcopy(v['facets'][0]))
    elif fault=='unsupported_field':v['pdms']=0
    else:v['summary'].update(evidence_ids=[],fact_ids=[])
    with pytest.raises(ValueError):parse(json.dumps(v),c,e)


@pytest.mark.parametrize('span', [[0,7],[-.1,1],[4,8],[4,4]])
def test_event_cannot_bridge_annotation_absence_or_leave_window(span):
    c,e,v=fixture();v['events']=[dict(event_type='relative_gap_closing',actors=['ego','actor_0001'],time_span_s=span,
        text_zh='距离减小。',evidence_ids=[],fact_ids=['ego_motion','track:actor_0001'],certainty='tentative')]
    with pytest.raises(ValueError):parse(json.dumps(v),c,e)


def test_valid_actor_interval_preserved_but_causal_claim_not_automatically_certified():
    c,e,v=fixture();v['events']=[dict(event_type='relative_gap_closing',actors=['ego','actor_0001'],time_span_s=[4,7],
        text_zh='距离减小。',evidence_ids=[],fact_ids=['ego_motion','track:actor_0001'],certainty='tentative')]
    assert parse(json.dumps(v),c,e)['events'][0]['certainty']=='tentative'
    v['events'][0]['fact_ids']=['ego_motion']
    with pytest.raises(ValueError,match='own structured evidence'):parse(json.dumps(v),c,e)


def test_pilot_never_accepts_test_data_or_generated_branch():
    c,e,v=fixture()
    with pytest.raises(ValueError):measurements(dict(role='source_log',source_split='navtest'))
    with pytest.raises(ValueError):measurements(dict(role='executed_branch',source_split='navtrain'))
    with pytest.raises(ValueError):assemble(dict(role='source_log',source_split='navtest'),c,v,e,'test-model')


def test_duplicate_keys_and_nonfinite_json_rejected():
    c,e,v=fixture();s=json.dumps(v)
    with pytest.raises(ValueError):parse(s[:-1]+',"events":[]}',c,e)
    with pytest.raises(ValueError):parse('{"number":NaN}',c,e)


def test_supported_event_needs_actual_temporal_support_not_held_presence():
    c,e,v=fixture()
    c['actor_tracks'][0].update(represented_intervals_s=[[0,.4]],samples=[dict(t_s=0,basis='previous_observation_held_presence_uncertain')])
    v['events']=[dict(event_type='relative_gap_closing',actors=['actor_0001'],time_span_s=[0,.4],
        text_zh='间距减小。',evidence_ids=[],fact_ids=['track:actor_0001'],certainty='supported')]
    with pytest.raises(ValueError,match='temporal track samples'):parse(json.dumps(v),c,e)


def test_event_cannot_support_early_claim_with_only_late_image():
    c,e,v=fixture()
    v['events']=[dict(event_type='ego_forward_progress',actors=['ego'],time_span_s=[0,.4],
        text_zh='自车向前。',evidence_ids=['CAM_F0_f+0'],fact_ids=['ego_motion'],certainty='supported')]
    with pytest.raises(ValueError,match='images do not overlap'):parse(json.dumps(v),c,e)
