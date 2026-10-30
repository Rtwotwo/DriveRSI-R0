"""Release decisions for a fixed, explicitly adaptive driving curriculum."""

import math

SAFETY = ('no_at_fault_collisions', 'drivable_area_compliance',
          'driving_direction_compliance', 'time_to_collision_within_bound', 'comfort')


def scores(report, expected):
    if (report.get('model') != 'chainflow_stage1' or report.get('complete') is not True
            or report.get('score_unit') != '0_to_1'
            or report.get('native_score_unmodified') is not True):
        raise ValueError('Complete native ChainFlow evaluation required')
    rows = report['rows']
    indexed = {r['token']: r['metrics'] for r in rows}
    if len(indexed) != len(rows) or set(indexed) != set(expected):
        raise ValueError('Full frozen evaluation membership required')
    for metrics in indexed.values():
        for key in ('PDMS', *SAFETY):
            v = metrics[key]
            if type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1:
                raise ValueError('Finite native metrics in 0..1 required')
    return indexed


def decide(parent, candidate, control, *, expected, target, retained,
           native_paired_benefit, mastery=0.95, strict_target=False):
    """Observed release only; no claim of blind generalization or universal safety."""
    p, c, b = (scores(r, expected) for r in (parent, candidate, control))
    if target not in p or not set(retained) <= p.keys():
        raise ValueError('Target and retained cases must belong to frozen evaluation')
    mean = lambda rs: sum(r['PDMS'] for r in rs.values()) / len(rs)
    safety_regressions = [t for t in p if any(c[t][k] < p[t][k] for k in SAFETY)]
    from driversi.rsi.decisions.problem_learning import target_success
    forgotten = [t for t in retained if not target_success(c[t]['PDMS'], mastery, strict=strict_target)
                 or any(c[t][k] < p[t][k] for k in SAFETY)]
    target_met = target_success(c[target]['PDMS'], mastery, strict=strict_target)
    new_zero = [t for t in p if p[t]['PDMS'] > 0 and c[t]['PDMS'] == 0]
    checks = dict(native_paired_benefit=native_paired_benefit is True,
                  target_improved=c[target]['PDMS'] > p[target]['PDMS'],
                  target_mastered=target_met,
                  global_pdms_non_decreasing=mean(c) >= mean(p),
                  safety_non_decreasing=not safety_regressions,
                  retained_cases_preserved=not forgotten,
                  paired_benefit=mean(c) > mean(b), no_new_zero=not new_zero)
    return dict(status='accepted' if all(checks.values()) else 'rejected', checks=checks,
                promoted=all(checks.values()), adaptive_evaluation=True,
                target_before=p[target]['PDMS'], target_after=c[target]['PDMS'],
                target_success=target_met,
                target_rule='strictly_positive' if strict_target and mastery == 0 else 'threshold',
                new_zero_count=len(new_zero),
                parent_pdms=100 * mean(p), candidate_pdms=100 * mean(c),
                control_pdms=100 * mean(b), safety_regression_count=len(safety_regressions),
                forgotten_count=len(forgotten), blind_generalization_established=False)
