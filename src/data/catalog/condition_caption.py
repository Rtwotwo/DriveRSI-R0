"""Fixed prompt profiles and factual/visual separation for semantic conditions."""
import json
from pathlib import Path

from driversi.data.catalog.store import identity

FIELDS=('road_appearance','surroundings','lighting','occluders','distinctive_details')


def prompt_path(scope):
    if scope not in ('retrieval_history','video_clip'):
        raise ValueError('Unknown caption scope')
    return Path(__file__).resolve().parents[3]/'src/config/data/semantic_conditioning'/f'{scope}.md'


def parse(raw):
    text=raw.strip()
    if text.startswith('```json') and text.endswith('```'):text=text[7:-3].strip()
    def unique(pairs):
        d={}
        for k,v in pairs:
            if k in d:raise ValueError('Duplicate caption key')
            d[k]=v
        return d
    value=json.loads(text,object_pairs_hook=unique)
    if not isinstance(value,dict) or set(value)!=set(FIELDS):raise ValueError('Fixed five appearance fields required')
    if any(not isinstance(v,str) or not v.strip() or len(v.split())>35 or len(v)>500 for v in value.values()):
        raise ValueError('Nonempty bounded caption fields required')
    return value


def assemble(facts, appearance=None, *, reviewed=False):
    scope=facts['scope']
    if facts['future_included'] != (scope=='video_clip'):
        raise ValueError('Caption information scope mismatch')
    counts=', '.join(f'{n} {kind}' for kind,n in facts.get('nearby_class_counts_30m',facts['class_counts']).items()) or 'no annotated actors'
    top=facts['anchor_objects'][:8]
    spatial='; '.join(f"instance {o['instance_id']} {o['semantic_class']} at ego xyz {o['relative_ego_xyz_m']}m, size {o['dimensions_lwh']}m, subtype unknown" for o in top)
    base=f"Annotated anchor within30m: {counts}. Ego net displacement {facts['ego_displacement_m']}m over the described {facts['duration_seconds']}s. {spatial}."
    visual=''
    if appearance and reviewed:
        visual=' '.join(f'{k}: {appearance[k]}' for k in FIELDS if appearance[k].lower()!='unknown')
    return dict(scope=scope,structured_text=base,vlm_observations=appearance,
                vlm_status='reviewed' if reviewed else 'unreviewed_not_ground_truth',
                retrieval_text=(base+(' '+visual if visual else '')) if scope=='retrieval_history' else None,
                generation_prompt=('Follow per-frame whitebox geometry, instance identities, class dictionary, calibrated cameras and timestamps. '
                                   'Semantic colors are not paint colors. Preserve object counts, trajectories and occlusion order across views. '
                                   'Vehicle subtype unknown; do not hallucinate identical vehicle types. '+base+' '+visual) if scope=='video_clip' else None,
                facts_identity=identity(facts),future_included=facts['future_included'],trainable=False)
