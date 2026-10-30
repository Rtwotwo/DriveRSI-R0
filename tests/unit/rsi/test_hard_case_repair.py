import copy
import math
import pytest

from driversi.rsi.decisions import hard_case_repair as rules


NODE = 'following_speed_check'
TEXT = ('Inspect the visible traffic corridor and preserve the intended route. '
        'Reduce speed smoothly only when the observed history indicates a closing conflict.')
NEW = ('Compare approaching road users across the visible history before progressing. '
       'Preserve lateral continuity while delaying entry into a visibly occupied crossing area.')


def preset():
    return dict(version='v1', low_speed_mps=2,
                skills={NODE: TEXT, 'turn_conflict_check': 'turn', 'low_speed_progress_check': 'slow'})


def member(token, speed=4, role='development', group=None, target=False):
    return dict(token=token, role=role, validation_group=group, repair_family=target,
                skill=dict(evidence=dict(navigation_command='GO STRAIGHT', speed_mps=speed)))


def candidate():
    p = preset(); p['version'] = 'different'; p['skills'][NODE] = NEW
    return p


def proposal(p=None):
    return dict(candidate=p or candidate(), skill_id=NODE, target_tokens=['target'], valid=True)


def metrics(pdms=0.5, ttc=0):
    return dict(PDMS=pdms, TTC=ttc, NC=1, DAC=1, DDC=1, EP=1, C=1)


def rows(members):
    result = []
    for m in members:
        for seed in (42, 43):
            for arm in ('native', 'incumbent', 'candidate'):
                improve = arm == 'candidate' and (m['token'] == 'target' or m.get('validation_group') == 'family')
                result.append(dict(token=m['token'], seed=seed, arm=arm,
                                   metrics=metrics(0.9 if improve else 0.5, 1 if improve else 0),
                                   raw_trajectory=[[float(i) + (0.1 if improve else 0), 0., 0.] for i in range(8)]))
    return result


def test_content_identity_ignores_version_and_gate_order():
    p = preset(); p['eligibility'] = {NODE: dict(commands=['STOP', 'GO STRAIGHT'], speed_min_mps=0,
                                              speed_max_mps=30, exclude_stop=False)}
    q = copy.deepcopy(p); q['version'] = 'v2'; q['eligibility'][NODE]['commands'].reverse()
    q['eligibility'][NODE]['speed_max_mps'] = 30.0
    assert rules.fingerprint(p, NODE) == rules.fingerprint(q, NODE)


def test_repeated_actual_content_skips_inference():
    ms = [member('target', target=True)]
    old = dict(candidate=candidate(), diagnosis='Different narrative')
    check = rules.precheck(proposal(), preset(), ms, [old])
    assert not check['passed'] and not check['requires_inference']
    assert 'repeated_intervention' in check['reasons']


def test_near_duplicate_and_noop_gate_are_rejected():
    ms = [member('target', target=True)]
    p = candidate(); p['skills'][NODE] += ' '
    assert 'repeated_intervention' in rules.precheck(proposal(p), preset(), ms, [candidate()])['reasons']
    p = candidate(); p['skills'][NODE] = NEW.replace('visible', 'visibly', 1)
    assert 'near_repeated_text' in rules.precheck(proposal(p), preset(), ms, [candidate()])['reasons']
    p = preset(); p['eligibility'] = {NODE: dict(commands=['GO STRAIGHT'], speed_min_mps=0, speed_max_mps=10, exclude_stop=True)}
    check = rules.precheck(proposal(p), preset(), ms, [])
    assert 'effective_no_change' in check['reasons']
    assert not check['passed']


def test_valid_atomic_text_and_effective_gate_edit():
    ms = [member('target', target=True), member('guard', speed=8)]
    assert rules.precheck(proposal(), preset(), ms, [])['passed']
    p = preset(); p['eligibility'] = {NODE: dict(commands=['GO STRAIGHT'], speed_min_mps=5, speed_max_mps=10, exclude_stop=True)}
    check = rules.precheck(proposal(p), preset(), ms, [])
    assert check['passed'] and check['change_kind'] == 'gate'
    assert check['changed_target_tokens'] == ['target']


def test_reject_joint_edit_foreign_node_and_route_edit():
    ms = [member('target', target=True)]
    p = candidate(); p['eligibility'] = {NODE: dict(commands=['GO STRAIGHT'], speed_min_mps=5, speed_max_mps=10, exclude_stop=True)}
    assert 'require_exactly_one_text_or_gate_edit' in rules.precheck(proposal(p), preset(), ms, [])['reasons']
    for edit in ('other', 'router'):
        p = candidate()
        if edit == 'other': p['skills']['turn_conflict_check'] = 'new'
        else: p['low_speed_mps'] = 3
        assert 'change_outside_selected_skill' in rules.precheck(proposal(p), preset(), ms, [])['reasons']


def test_effective_signature_ignores_version_and_disabled_text():
    p = preset(); p['eligibility'] = {NODE: dict(commands=['TURN LEFT'], speed_min_mps=0, speed_max_mps=10, exclude_stop=True)}
    q = copy.deepcopy(p); q['version'] = 'v3'; q['skills'][NODE] = NEW
    assert rules.effective_signature(p, [member('target')]) == rules.effective_signature(q, [member('target')])


def test_screen_true_improvement_and_new_zero_protection():
    ms = [member('target', target=True), member('guard')]; data = rows(ms)
    assert rules.screen(data, ms, [42, 43], ['target'])['accepted']
    affected = next(r for r in data if r['token'] == 'guard' and r['arm'] == 'candidate')
    affected['metrics'].update(PDMS=0, NC=0)
    result = rules.screen(data, ms, [42, 43], ['target'])
    assert not result['accepted']
    assert 'development:new_zero_vs_native' in result['reasons']
    assert 'development:safety_regression_vs_incumbent' in result['reasons']


@pytest.mark.parametrize('corruption', ['missing', 'duplicate', 'nan', 'bool', 'trajectory_nan'])
def test_screen_rejects_partial_or_invalid_evidence(corruption):
    ms = [member('target', target=True)]; data = rows(ms)
    if corruption == 'missing': data.pop()
    elif corruption == 'duplicate': data.append(copy.deepcopy(data[0]))
    elif corruption == 'nan': data[0]['metrics']['TTC'] = math.nan
    elif corruption == 'bool': data[0]['metrics']['TTC'] = True
    else: data[0]['raw_trajectory'][0][0] = math.nan
    with pytest.raises(ValueError): rules.screen(data, ms, [42, 43], ['target'])


def test_screen_cannot_read_validation():
    ms = [member('target', role='validation')]
    with pytest.raises(ValueError): rules.screen(rows(ms), ms, [42, 43], ['target'])
    with pytest.raises(ValueError): rules.precheck(proposal(), preset(), ms, [])


def test_final_requires_real_family_transfer_and_control_retention():
    ms = [member('target', target=True), member('guard'), member('family', role='validation', group='family'),
          member('control', role='validation', group='control')]
    data = rows(ms)
    assert rules.final_decision(data, ms, [42, 43], ['target'])['accepted']
    for row in data:
        if row['token'] == 'family' and row['arm'] == 'candidate': row['metrics']['TTC'] = 0
    assert 'no_cross_log_TTC_transfer' in rules.final_decision(data, ms, [42, 43], ['target'])['reasons']
    data = rows(ms)
    for row in data:
        if row['token'] == 'control' and row['arm'] == 'candidate': row['metrics']['PDMS'] = 0.4
    result = rules.final_decision(data, ms, [42, 43], ['target'])
    assert not result['accepted']
    assert 'validation_control:mean_decline_vs_incumbent' in result['reasons']
