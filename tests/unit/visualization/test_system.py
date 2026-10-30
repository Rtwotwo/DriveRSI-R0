import json

import pytest

from driversi.visualization.system import system_overview


def fixture(tmp_path):
    path = tmp_path / 'src/config/visualization/architecture.json'
    path.parent.mkdir(parents=True)
    value = {
        'schema_version': 1, 'reviewed_on': '2026-10-26',
        'summary': 'A pilot', 'boundary': 'No promotion claim',
        'layers': [{'id':'one','title':'Evidence','owner':'src/data','status':'pilot',
                    'description':'Aggregate only','outputs':'Candidates','evidence':'experiment:pilot'}],
        'models': [], 'priorities': [], 'contracts': ['No test details'],
    }
    path.write_text(json.dumps(value))
    config = {'architecture':'src/config/visualization/architecture.json',
              'experiments':[{'id':'pilot'}]}
    return config, path, value


def test_architecture_links_only_registered_evidence(tmp_path):
    config, path, value = fixture(tmp_path)
    assert system_overview(config,tmp_path)['layers'][0]['evidence']=='experiment:pilot'
    value['layers'][0]['evidence']='experiment:not-registered'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='unregistered'):
        system_overview(config,tmp_path)


def test_private_content_is_redacted_and_extra_fields_are_rejected(tmp_path):
    config, path, value = fixture(tmp_path)
    value['summary']='/data/private/credential'
    path.write_text(json.dumps(value))
    assert system_overview(config,tmp_path)['summary']=='内部记录（未导出）'
    value['layers'][0]['raw_test_tokens']=['never-publish']
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='declared'):
        system_overview(config,tmp_path)


def test_outside_preset_and_unverified_status_rejected(tmp_path):
    config,path,value=fixture(tmp_path)
    with pytest.raises(ValueError,match='owned'):
        system_overview(dict(config,architecture='/tmp/unknown.json'),tmp_path)
    value['layers'][0]['status']='stable_model_improvement'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='state'):
        system_overview(config,tmp_path)


def test_existing_config_without_architecture_remains_compatible(tmp_path):
    assert system_overview({},tmp_path) is None


def test_current_only_architecture_keeps_design_without_reading_old_runs(tmp_path):
    from driversi.visualization.current import current_config
    config, path, value = fixture(tmp_path)
    value['layers'][0]['evidence'] = ''
    path.write_text(json.dumps(value))
    config['display_mode'] = 'current_only'
    config['experiments'][0]['receipt'] = '/must/not/read/historical.json'
    filtered = current_config(config)
    result = system_overview(filtered, tmp_path)
    assert filtered['experiments'] == []
    assert result['layers'][0]['evidence'] == ''
    # Current-only design cannot silently link a hidden historic experiment.
    value['layers'][0]['evidence'] = 'experiment:pilot'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='unregistered'):
        system_overview(filtered, tmp_path)
