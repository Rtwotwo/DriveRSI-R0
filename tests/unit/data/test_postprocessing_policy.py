import json
from driversi.data.datasets import postprocessing as p, synthetic_publish, simscale_collection


def test_policy_scoped_to_owned_production_not_benchmarks(tmp_path, monkeypatch):
    monkeypatch.setattr(simscale_collection,'project_root',lambda:tmp_path)
    root=tmp_path/'dataset';campaign=tmp_path/'exp/current'
    simscale_collection.initialize(root,campaign)
    monkeypatch.setattr(synthetic_publish,'dataset_root',lambda:root/'synthetic_navsim')
    (root/'postprocessing.json').write_text(json.dumps(dict(schema_version=1,mode='training_only',
        campaign=str(campaign),features='deferred',scene_caption='deferred',classification='deferred',
        training_admission='unchanged')))
    current=dict(output=str(campaign/'wave'))
    assert p.mode(current)=='training_only'
    assert p.mode(current,production=False)=='features_and_classification'
    assert p.mode(dict(output=str(tmp_path/'exp/other')))=='features_and_classification'
