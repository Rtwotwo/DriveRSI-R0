"""Build augmentation proposals from the training pool's own measured results.

No test result, test descriptor or test-derived priority is an input. These are
mining scores on pretraining sources, never heldout performance estimates.
"""
import json
from collections import Counter
from pathlib import Path

from driversi.rsi.decisions.evaluation_aggregate import METRICS, number


def build_plan(index, descriptors, scored_rows, checkpoint_identity):
    if index.get('split') != 'navtrain' or not index.get('complete'):
        raise ValueError('Complete official navtrain population required')
    population = {r['token']: r for r in index['records']}
    if len(population) != len(index['records']) or set(population) != set(index['expected_tokens']):
        raise ValueError('Exact unique training population required')
    if any(r['kind'] != 'original_navtrain' for r in population.values()):
        raise ValueError('Only original navtrain anchors may enter mining')
    meta = {r['token']: r for r in descriptors['records']}
    if descriptors.get('kind') != 'navtrain_history_scene_index' or set(meta) != set(population):
        raise ValueError('Training descriptors differ from full population')
    seen, failures, queue, deficits = set(), [], [], Counter()
    bins = dict(zero=0, **{'0-80': 0, '80-95': 0, '95+': 0})
    for row in scored_rows:
        token = row['token']
        if token not in population or token in seen or row['log_name'] != population[token]['source_log']:
            raise ValueError('Scored token/source outside unique training population')
        seen.add(token)
        if row.get('valid') is not True:
            failures.append(dict(token=token, source_log=row['log_name'], reason=row.get('error')))
            continue
        scores = {k: number(row[k]) for k in ('score', *METRICS)}
        shortfalls = [cap for key, cap in METRICS.items() if scores[key] < 1]
        deficits.update(shortfalls)
        score = scores['score']
        bins['zero' if score == 0 else '0-80' if score < .8 else '80-95' if score < .95 else '95+'] += 1
        safety = [c for c in shortfalls if c in
                  {'collision','road_compliance','direction_compliance','interaction'}]
        triggers = (['pdms_below_95'] if score < .95 else []) + (
            ['safety_submetric_shortfall'] if safety else []) + (
            ['comfort_shortfall'] if 'comfort' in shortfalls else [])
        if not triggers:
            continue
        description = meta[token]
        if description['source_log'] != row['log_name']:
            raise ValueError('Training descriptor/source mismatch')
        continuous = description['history'].get('continuous') is True
        queue.append(dict(token=token, source_log=row['log_name'], anchor_index=population[token]['anchor_index'],
            pdms=100 * score, metrics=scores, observed_shortfalls=shortfalls,
            selection_triggers=triggers,
            conditions=description['conditions'], history_continuous=continuous,
            next_action='same_anchor_teacher_and_platform_controls' if continuous else 'repair_history_support',
            teacher_threshold_pdms=95, model_cause_established=False, synthesis_admitted=False,
            trainable=False))
    queue.sort(key=lambda r: (r['pdms'], r['source_log'], r['token']))
    return dict(schema_version=1, kind='navtrain_measured_augmentation_plan',
        evidence_split='navtrain', test_feedback_used=False, checkpoint_identity=checkpoint_identity,
        coverage=dict(expected=len(population), recorded=len(seen), scored=len(seen)-len(failures),
                      runtime_failures=len(failures), missing=len(population)-len(seen),
                      all_recorded=len(seen)==len(population), all_scored=len(seen)==len(population) and not failures),
        bins=bins, submetric_shortfall_counts=dict(deficits), augmentation_candidates=queue,
        selection_policy='pdms_below_95_or_safety_or_comfort_shortfall; mild_progress_only_is_monitoring',
        failures=failures, partial=len(seen)!=len(population),
        interpretation='Measured training shortfalls; causal diagnosis and same-branch admission still required',
        training_started=False, promoted=False)


def read_scored_rows(shard_outputs, expected_identity=None):
    rows, identities = [], []
    for folder in map(Path, shard_outputs):
        identity_path = folder / 'identity.json'
        if not identity_path.exists():
            continue
        identity = json.loads(identity_path.read_text())
        if identity.get('split') != 'navtrain' or identity.get('purpose') != 'training_pool_mining':
            raise ValueError('Test evaluation artifacts cannot enter train-pool mining')
        if expected_identity is not None and {k:v for k,v in identity.items() if k != 'shard'} != expected_identity:
            raise ValueError('Stored training results differ from requested native identity')
        identities.append(identity)
        path = folder / 'scores.jsonl'
        if path.exists():
            # Append-only live files can end with a pending write. A completed
            # JSON line is the only unit counted; no partial JSON is repaired.
            with path.open() as stream:
                for line in stream:
                    if line.endswith('\n'):
                        rows.append(json.loads(line))
    common = [{k: v for k,v in i.items() if k != 'shard'} for i in identities]
    if common and any(i != common[0] for i in common):
        raise ValueError('Training mining checkpoint/protocol identities differ')
    return rows, common[0] if common else None
