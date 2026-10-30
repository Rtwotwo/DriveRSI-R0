"""Deterministic hard-case search filters and unchanged safety admission."""
import copy
import math
import statistics
from difflib import SequenceMatcher

from . import grounded_skills as grounded
from .frozen_skills import METRICS, SAFETY, choose_skill
from .skill_evolution import ARMS, evaluate, identity


def _text(value):
    return ' '.join(value.split())


def _gate(preset, node):
    gate = copy.deepcopy(preset.get('eligibility', {}).get(node))
    if gate is not None:
        gate['commands'] = sorted(set(gate['commands']))
        for field in ('speed_min_mps', 'speed_max_mps'):
            gate[field] = float(gate[field])
    return gate


def fingerprint(preset, node):
    """Version and diagnostic prose are not an executable intervention."""
    return identity(dict(text=_text(preset['skills'][node]), gate=_gate(preset, node)))


def _prefixes(preset, members):
    result = {}
    for member in members:
        token = member['token']
        if token in result:
            raise ValueError('Duplicate member in effective signature')
        state = member.get('observed_state', member.get('skill', {}).get('evidence'))
        skill = grounded.apply_gate(choose_skill(state, preset), preset)
        # Keep actual whitespace here: the native tokenizer sees the exact prefix.
        result[token] = ('Planning checklist for this observation:\n' + skill['text']
                         if skill['enabled'] else '')
    return result


def effective_signature(preset, members):
    return identity(_prefixes(preset, members))


def _prior_presets(history):
    for entry in history:
        if not isinstance(entry, dict):
            raise ValueError('Malformed intervention history')
        if 'skills' in entry:
            yield entry
        for name in ('candidate', 'proposed_preset'):
            if isinstance(entry.get(name), dict) and 'skills' in entry[name]:
                yield entry[name]
        for preset in entry.get('variants', {}).values():
            if isinstance(preset, dict) and 'skills' in preset:
                yield preset


def precheck(proposal, incumbent, members, history):
    """Check a schema-validated proposal before any paid GPU evaluation.

    `proposal` has candidate, skill_id, and target_tokens. `grounded.validate`
    is responsible for diagnosis references, text safety and gate schema.
    History entries may be memory events or complete prior presets.
    """
    candidate, node = proposal['candidate'], proposal['skill_id']
    if not members or any(m['role'] != 'development' for m in members):
        raise ValueError('Proposal filters accept development observations only')
    if node not in incumbent['skills'] or node not in candidate['skills']:
        raise ValueError('Unknown skill node')
    targets = set(proposal.get('target_tokens', [m['token'] for m in members if m.get('repair_family')]))
    known = {m['token'] for m in members}
    if not targets or not targets <= known:
        raise ValueError('Explicit nonempty development repair targets required')
    reasons = []
    # Candidate cannot silently alter routing, another skill, or any configuration.
    frozen = copy.deepcopy(candidate)
    frozen['version'] = incumbent['version']
    frozen['skills'][node] = incumbent['skills'][node]
    old_gate = incumbent.get('eligibility', {}).get(node)
    if old_gate is None:
        frozen.get('eligibility', {}).pop(node, None)
        if not frozen.get('eligibility') and 'eligibility' not in incumbent:
            frozen.pop('eligibility', None)
    else:
        frozen.setdefault('eligibility', {})[node] = copy.deepcopy(old_gate)
    if frozen != incumbent:
        reasons.append('change_outside_selected_skill')
    text_changed = _text(candidate['skills'][node]) != _text(incumbent['skills'][node])
    gate_changed = _gate(candidate, node) != _gate(incumbent, node)
    if text_changed == gate_changed:
        reasons.append('require_exactly_one_text_or_gate_edit')
    kind = 'text' if text_changed and not gate_changed else 'gate' if gate_changed and not text_changed else 'invalid'
    before, after = _prefixes(incumbent, members), _prefixes(candidate, members)
    changed = sorted(t for t in targets if before[t] != after[t])
    if before == after:
        reasons.append('effective_no_change')
    if not changed:
        reasons.append('no_target_prompt_change')
    candidate_id = fingerprint(candidate, node)
    effective_id = identity(after)
    matches = []
    for previous in _prior_presets(history):
        if node not in previous['skills']:
            continue
        reason = None
        if candidate_id == fingerprint(previous, node):
            reason = 'repeated_intervention'
        elif (_gate(candidate, node) == _gate(previous, node)
              and SequenceMatcher(None, _text(candidate['skills'][node]).casefold(),
                                  _text(previous['skills'][node]).casefold(), autojunk=False).ratio() > 0.97):
            reason = 'near_repeated_text'
        elif effective_id == effective_signature(previous, members):
            reason = 'repeated_effective_intervention'
        if reason:
            reasons.append(reason)
            matches.append(fingerprint(previous, node))
    if proposal.get('valid') is False:
        reasons.append('invalid_doctor_proposal')
    return dict(passed=not reasons, reasons=list(dict.fromkeys(reasons)),
                change_kind=kind, fingerprint=candidate_id, effective_signature=effective_id,
                changed_target_tokens=changed, matched_history=sorted(set(matches)),
                requires_inference=not reasons)


def _paired(rows, members, seeds):
    tokens = {m['token'] for m in members}
    if not members or len(tokens) != len(members) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Unique nonempty members and seeds required')
    expected = {(t, s, a) for t in tokens for s in seeds for a in ARMS}
    by = {}
    for row in rows:
        key = (row['token'], row['seed'], row['arm'])
        if key not in expected or key in by:
            raise ValueError('Unexpected or duplicate scored row')
        if any(type(row['metrics'].get(k)) not in (int, float)
               or not math.isfinite(row['metrics'][k]) or not 0 <= row['metrics'][k] <= 1 for k in METRICS):
            raise ValueError('Invalid native metrics')
        trajectory = row.get('raw_trajectory')
        if (not isinstance(trajectory, list) or len(trajectory) != 8
                or any(not isinstance(p, list) or len(p) != 3
                       or any(type(v) not in (float, int) or not math.isfinite(v) for v in p) for p in trajectory)):
            raise ValueError('Invalid native raw trajectory')
        by[key] = row
    if set(by) != expected:
        raise ValueError('Incomplete paired coverage')
    return by


def screen(rows, members, seeds, target_tokens):
    """Development-only screen; a repair must improve without safety regression."""
    if any(m['role'] != 'development' for m in members):
        raise ValueError('Screen cannot consume validation outcomes')
    by = _paired(rows, members, seeds)
    tokens = {m['token'] for m in members}; targets = set(target_tokens)
    if not targets or not targets <= tokens:
        raise ValueError('Invalid development targets')
    reasons, comparisons = [], {}
    for reference in ('native', 'incumbent'):
        zeros, bad = [], []
        for token in sorted(tokens):
            for seed in seeds:
                a, b = by[token, seed, reference]['metrics'], by[token, seed, 'candidate']['metrics']
                if a['PDMS'] > 0 and b['PDMS'] == 0:
                    zeros.append([token, seed])
                metrics = [k for k in SAFETY if b[k] < a[k] - 1e-9]
                if metrics:
                    bad.append(dict(token=token, seed=seed, metrics=metrics))
        delta = statistics.mean(by[t, s, 'candidate']['metrics']['PDMS'] - by[t, s, reference]['metrics']['PDMS']
                                for t in tokens for s in seeds)
        comparisons[reference] = dict(mean_delta=delta, new_zero=zeros, safety_regressions=bad)
        if zeros:
            reasons.append('development:new_zero_vs_' + reference)
        if bad:
            reasons.append('development:safety_regression_vs_' + reference)
    if comparisons['incumbent']['mean_delta'] <= 1e-9:
        reasons.append('development:no_improvement_over_incumbent')
    target_gain = statistics.mean(by[t, s, 'candidate']['metrics']['TTC'] - by[t, s, 'incumbent']['metrics']['TTC']
                                  for t in targets for s in seeds)
    changed = any(by[t, s, 'candidate']['raw_trajectory'] != by[t, s, 'incumbent']['raw_trajectory']
                  for t in targets for s in seeds)
    if target_gain <= 1e-9:
        reasons.append('development:no_target_TTC_repair')
    if not changed:
        reasons.append('development:target_action_unchanged')
    return dict(accepted=not reasons, reasons=reasons, target_ttc_delta=target_gain,
                action_changed=changed, comparisons=comparisons,
                scope='development_screen_not_promotion')


def final_decision(rows, members, seeds, target_tokens):
    """Apply original cross-log admission plus explicit TTC family transfer."""
    by = _paired(rows, members, seeds)
    decision = evaluate(rows, members, seeds)
    development = [m for m in members if m['role'] == 'development']
    dev_tokens = {m['token'] for m in development}
    first = screen([r for r in rows if r['token'] in dev_tokens], development, seeds, target_tokens)
    decision['reasons'] += first['reasons']
    groups = {name: {m['token'] for m in members if m['role'] == 'validation' and m.get('validation_group') == name}
              for name in ('family', 'control')}
    if not all(groups.values()) or sum(map(len, groups.values())) != sum(m['role'] == 'validation' for m in members):
        raise ValueError('Fixed nonempty family and control validation cohorts required')
    summaries = {}
    for name, tokens in groups.items():
        summary = {}
        for reference in ('native', 'incumbent'):
            delta = statistics.mean(by[t, s, 'candidate']['metrics']['PDMS'] - by[t, s, reference]['metrics']['PDMS']
                                    for t in tokens for s in seeds)
            summary['pdms_delta_vs_' + reference] = delta
            if delta < -1e-9:
                decision['reasons'].append('validation_' + name + ':mean_decline_vs_' + reference)
        summaries[name] = summary
    transfer = statistics.mean(by[t, s, 'candidate']['metrics']['TTC'] - by[t, s, 'incumbent']['metrics']['TTC']
                               for t in groups['family'] for s in seeds)
    if transfer <= 1e-9:
        decision['reasons'].append('no_cross_log_TTC_transfer')
    decision['reasons'] = list(dict.fromkeys(decision['reasons']))
    decision.update(accepted=not decision['reasons'], development_screen=first,
                    validation_target_ttc_delta=transfer, validation_subgroups=summaries,
                    production_policy_changed=False)
    return decision
