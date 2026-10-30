"""Compact labels must retain scope/uncertainty while removing raw state series."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from driversi.data.catalog import scene_label_summary as summary
from driversi.data.catalog import scene_label_jobs as jobs


def full_record():
    return dict(schema_version='driversi.scene_labels.v1',
        scene=dict(token='training-token',split='navtrain',window_s=[0,7]),
        information_scope=dict(scope='retrospective_full_clip',future_included=True,online_policy_input=False),
        labels=dict(summary=dict(text_zh='城市直路，经过路侧车辆。',certainty='supported'),
            facets=[dict(axis='road_geometry',label='straight_segment',certainty='supported'),
                    dict(axis='ego_motion',label='forward_progress',certainty='supported'),
                    dict(axis='environment',label='glass_buildings',certainty='supported'),
                    dict(axis='environment',label='construction_adjacent',certainty='supported'),
                    dict(axis='road_markings',label='white_solid_visible',certainty='tentative')],
            events=[dict(event_type='ego_forward_progress',certainty='supported'),
                    dict(event_type='vehicle_cut_in',certainty='tentative')],
            unknowns=[dict(field='traffic_light_state',reason_zh='看不清')]),
        measured_summary=dict(ego_motion=dict(path_length_xy_m=57.002,annotation_speed_range_mps=[7.914,8.326],
            timeline=[dict(t_s=0,world_xy_m=[0,0],heading_change_deg=0)]),map_context=dict(ego_lane_association='not_computed')),
        review=dict(format_valid=True,status='awaiting_user_review',factual_status='vlm_claims_unverified'),
        support_files=dict(structured_evidence='context.json',source_photos='source-evidence-overview.jpg'))


def test_compact_removes_series_without_mutating_full_evidence():
    full=full_record();before=copy.deepcopy(full)
    value=summary.compact(full,dict(path='audit/full.json',sha256='digest'))
    encoded=json.dumps(value)
    for name in ['world_xy_m','heading_change_deg','timeline','t_s']:
        assert '"'+name+'"' not in encoded
    assert full==before
    assert value['speed_kmh']['mean']==29.32
    assert value['speed_kmh']['range']==[28.49,29.97]
    assert value['online_policy_input'] is False
    assert value['review']['indexable'] is False and value['trainable'] is False


def test_tentative_and_appearance_only_tags_do_not_become_primary_facts():
    value=summary.compact(full_record(),{})
    assert 'white_solid_visible' not in value['visual_context']
    assert value['uncertain']==[dict(axis='road_markings',label='white_solid_visible'),
                               dict(kind='event',label='vehicle_cut_in')]
    assert 'glass_buildings' not in value['visual_context']
    assert 'construction_adjacent' in value['visual_context']
    assert value['interaction']['events']==[]
    assert value['road']['signal_state']=='unknown'


def test_tentative_caption_remains_a_hypothesis():
    full=full_record();full['labels']['summary']['certainty']='tentative'
    value=summary.compact(full,{})
    assert value['caption'] is None
    assert dict(kind='caption',text=full['labels']['summary']['text_zh']) in value['uncertain']


@pytest.mark.parametrize('fault',['nan','negative','zero_duration','online'])
def test_invalid_numeric_or_scope_cannot_make_a_compact_label(fault):
    full=full_record()
    if fault=='nan':full['measured_summary']['ego_motion']['path_length_xy_m']=float('nan')
    elif fault=='negative':full['measured_summary']['ego_motion']['annotation_speed_range_mps']=[-1,2]
    elif fault=='zero_duration':full['scene']['window_s']=[0,0]
    else:full['information_scope']['online_policy_input']=True
    with pytest.raises(ValueError):summary.compact(full,{})


def test_publisher_links_full_audit_and_preserves_old_primary_sample(tmp_path,monkeypatch):
    original=json.dumps(full_record())
    (tmp_path/'sample.json').write_text(original)
    monkeypatch.setattr(jobs,'decode',lambda *a,**kw:dict(scene_label=full_record()))
    job=dict(input_identity='input',source_sha256='source',context_sha256='context')
    jobs.publish(job,'model',dict(raw='cached',request_identity='request'),tmp_path)
    value=json.loads((tmp_path/'sample.json').read_text())
    ref=value['evidence_ref'];full_path=tmp_path/ref['path']
    assert hashlib.sha256(full_path.read_bytes()).hexdigest()==ref['sha256']
    audit=json.loads(full_path.read_text())
    assert audit['measured_summary']['ego_motion']['timeline']
    assert (full_path.parent/audit['support_files']['structured_evidence']).resolve()==tmp_path/'context.json'
    assert any(p.read_text()==original for p in (tmp_path/'superseded').glob('*.json'))
    old_hash=hashlib.sha256(original.encode()).hexdigest()
    metadata_path=tmp_path/'superseded'/(old_hash+'.references.json')
    metadata=json.loads(metadata_path.read_text())
    assert (metadata_path.parent/metadata['reference_base']).resolve()==tmp_path
    prior=full_path.read_bytes()
    jobs.publish(job,'model',dict(raw='cached',request_identity='request'),tmp_path)
    assert full_path.read_bytes()==prior
    assert len(list((tmp_path/'audit').glob('*.json')))==1
