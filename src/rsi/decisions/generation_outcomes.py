"""Recognize explicit physical termination without treating it as admission."""


def incomplete_physical_branch(native, branch='teacher'):
    row = native.get('branches', {}).get(branch, {})
    steps = row.get('completed_steps')
    events = row.get('events', {})
    return (
        native.get('status') == 'completed'
        and native.get('restore_exact') is True
        and not native.get('error') and not native.get('error_type')
        and bool(native.get('source_files'))
        and type(steps) is int and 0 <= steps < 50
        and row.get('failure') == 'ego_terminated_before_complete_score_window'
        and row.get('trajectory_admitted') is False
        and row.get('score') is None
        and any(type(events.get(k)) is int and events[k] > 0
                for k in ('collision', 'offroad', 'goal', 'route_support_end'))
    )
