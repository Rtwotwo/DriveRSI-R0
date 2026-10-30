"""Bounded skill edits, paired admission and transactional experiment memory."""
import copy
import hashlib
import json
import math
import re
import sqlite3
import statistics
from pathlib import Path

from driversi.rsi.decisions.frozen_skills import METRICS, SAFETY

ARMS = ('native', 'incumbent', 'candidate')


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def acquisition_group(log):
    parts = log.rsplit('_', 2)
    if len(parts) != 3 or not all(x.isdigit() for x in parts[1:]):
        raise ValueError('Unrecognized acquisition log identity')
    return parts[0]


def validate_proposal(value, skill_id, incumbent):
    if not isinstance(value, dict) or set(value) != {'skill_id', 'skill_text', 'rationale', 'expected_effect'}:
        raise ValueError('Doctor response schema mismatch')
    if value['skill_id'] != skill_id or skill_id not in incumbent['skills']:
        raise ValueError('Doctor cannot change the selected router node')
    if any(not isinstance(value[k], str) for k in value):
        raise ValueError('Doctor fields must be strings')
    text = value['skill_text'].strip()
    if not 80 <= len(text) <= 1000 or len(value['rationale']) > 1200 or len(value['expected_effect']) > 600:
        raise ValueError('Doctor response length out of bounds')
    # The reply is data, never code. Reject identity/score targeting and fixed numeric actions.
    if re.search(r'\d|https?://|```|navtest|PDMS|<\||ignore.*instruction', text, re.I):
        raise ValueError('Skill contains identifiers, numeric targets, score or control markup')
    if text == incumbent['skills'][skill_id]:
        raise ValueError('Unchanged proposal')
    candidate = copy.deepcopy(incumbent)
    candidate['skills'][skill_id] = text
    candidate['version'] = 'doctor-' + identity(candidate['skills'])[:16]
    return candidate


def evaluate(rows, members, seeds):
    roles = {m['token']: m['role'] for m in members}
    if len(roles) != len(members) or set(roles.values()) != {'development', 'validation'}:
        raise ValueError('Explicit nonempty development and validation cohorts required')
    expected = {(t, s, a) for t in roles for s in seeds for a in ARMS}
    by = {}
    for row in rows:
        key = (row['token'], row['seed'], row['arm'])
        if key not in expected or key in by:
            raise ValueError('Duplicate or unexpected scored row')
        if any(type(row['metrics'].get(m)) not in (float, int) or
               not math.isfinite(row['metrics'][m]) or not 0 <= row['metrics'][m] <= 1 for m in METRICS):
            raise ValueError('Invalid native score')
        by[key] = row
    if set(by) != expected:
        raise ValueError('Incomplete paired coverage')
    cohorts, reasons = {}, []
    for role in ('development', 'validation'):
        keys = [(t, s) for t in roles if roles[t] == role for s in seeds]
        means = {a: statistics.mean(by[t, s, a]['metrics']['PDMS'] for t, s in keys) for a in ARMS}
        comparisons = {}
        for reference in ('native', 'incumbent'):
            new_zero, regressions = [], []
            for t, s in keys:
                a, b = by[t, s, reference]['metrics'], by[t, s, 'candidate']['metrics']
                if a['PDMS'] > 0 and b['PDMS'] == 0: new_zero.append([t, s])
                bad = [m for m in SAFETY if b[m] < a[m] - 1e-9]
                if bad: regressions.append(dict(token=t, seed=s, metrics=bad))
            comparisons[reference] = dict(mean_delta=means['candidate']-means[reference],
                                          new_zero=new_zero, safety_regressions=regressions)
            if new_zero: reasons.append(role + ':new_zero_vs_' + reference)
            if regressions: reasons.append(role + ':safety_regression_vs_' + reference)
        cohorts[role] = dict(scenes=len(keys)//len(seeds), seed_pairs=len(keys), means=means, comparisons=comparisons)
    if cohorts['development']['comparisons']['incumbent']['mean_delta'] <= 1e-9:
        reasons.append('development:no_improvement_over_incumbent')
    for reference in ('native', 'incumbent'):
        if cohorts['validation']['comparisons'][reference]['mean_delta'] < -1e-9:
            reasons.append('validation:mean_decline_vs_' + reference)
    if cohorts['validation']['comparisons']['incumbent']['mean_delta'] <= 1e-9:
        reasons.append('validation:no_positive_transfer_over_incumbent')
    return dict(accepted=not reasons, reasons=reasons, cohorts=cohorts,
                scope='bounded_adaptation_pilot_not_global_release', statistically_established=False)


def memory(path, initial=None, event=None, expected_revision=None):
    """One transaction commits both experience and active preset, or neither."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=30) as db:
        db.execute('CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, body TEXT NOT NULL)')
        db.execute('BEGIN IMMEDIATE')
        record = db.execute('SELECT body FROM state WHERE id=1').fetchone()
        if record is None:
            if initial is None: raise ValueError('Memory has not been initialized')
            state = dict(revision=0, initial_hash=identity(initial), active=initial, events=[])
        else:
            state = json.loads(record[0])
            if initial is not None and state['initial_hash'] != identity(initial):
                raise ValueError('Initial skill identity changed')
        if event is not None:
            old = next((e for e in state['events'] if e['id'] == event['id']), None)
            if old is not None:
                if old != event: raise ValueError('Conflicting episode replay')
            else:
                if expected_revision != state['revision']: raise ValueError('Stale skill decision')
                if event['incumbent_hash'] != identity(state['active']): raise ValueError('Incumbent changed')
                state['events'].append(event)
                if event['decision']['accepted']: state['active'] = event['candidate']
                state['revision'] += 1
        db.execute('INSERT OR REPLACE INTO state VALUES (1, ?)', (json.dumps(state, ensure_ascii=False),))
    return state


def seed_memory(path, initial, snapshot):
    """Replay preserved events into a new namespace; never modify the old database."""
    if snapshot['initial_hash'] != identity(initial) or snapshot['revision'] != len(snapshot['events']):
        raise ValueError('Invalid inherited memory')
    state = memory(path, initial)
    for revision, event in enumerate(snapshot['events']):
        state = memory(path, event=event, expected_revision=revision)
    if state != snapshot: raise ValueError('Inherited memory replay mismatch')
    return state


def smoke(output):
    """Explicitly synthetic F0 witness; never a driving-result artifact."""
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    initial = dict(version='fixture', skills={'test': 'synthetic fixture'})
    state = memory(out/'memory.sqlite', initial)
    event = dict(id='synthetic-rejection', incumbent_hash=identity(initial), candidate=initial,
                 decision=dict(accepted=False, reasons=['synthetic_fixture']))
    first = memory(out/'memory.sqlite', event=event, expected_revision=0)
    second = memory(out/'memory.sqlite', event=event, expected_revision=0)
    assert first == second and second['active'] == initial and second['revision'] == 1
    report = dict(synthetic_fixture=True, driving_evidence=False, rollback=True, replay_idempotent=True)
    (out/'report.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('output')
    print(json.dumps(smoke(parser.parse_args().output)))
