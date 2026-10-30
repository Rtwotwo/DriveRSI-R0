"""Current-campaign projection without historical runs or test identities."""
from pathlib import Path


def current_campaign(config, project):
    if config.get('display_mode') != 'current_only':
        return None
    if config.get('current_dynamic_history_campaign'):
        from .dynamic_history import project as dynamic_state
        route = config['current_dynamic_history_campaign']
        route = {'directory': route} if isinstance(route, str) else route
        path = (Path(project) / route['directory']).resolve()
        plan = (Path(project) / route['plan']).resolve() if route.get('plan') else None
        if any(not p.is_relative_to((Path(project) / 'exp').resolve()) for p in (path, plan) if p):
            raise ValueError('Dynamic campaign must belong to experiment storage')
        return dynamic_state(path, plan)
    if config.get('current_skill_campaign'):
        from .skill_evolution import project as skill_state
        path = (Path(project)/config['current_skill_campaign']).resolve()
        if not path.is_relative_to((Path(project)/'exp').resolve()):
            raise ValueError('Skill campaign must belong to experiment storage')
        return skill_state(path)
    from driversi.configuration import read_mapping
    from driversi.rsi.experiments.continual_curriculum import project as project_state
    path = (Path(project) / config['current_campaign'] / 'private/state.json').resolve()
    if not path.is_relative_to((Path(project) / 'exp').resolve()):
        raise ValueError('Current campaign must belong to experiment storage')
    return project_state(read_mapping(path))


def current_config(config):
    """Do not read old experiment receipts when building a current-only view."""
    if config.get('display_mode') != 'current_only':
        return config
    import copy
    c = copy.deepcopy(config)
    for key in ('experiments', 'harness', 'legacy', 'training', 'production', 'scenes', 'whitebox'):
        c[key] = []
    if c.get('clustering'):
        c['clustering'].pop('failure_mapping', None)
        c['clustering'].pop('overlap_summary', None)
    return c
