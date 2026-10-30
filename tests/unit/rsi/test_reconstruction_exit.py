import pytest
from driversi.rendering.gaussian import mtgs_reconstruction as native


@pytest.mark.parametrize('code',[0,None])
def test_successful_native_cli_exit_returns_to_receipt_writer(monkeypatch,code):
    def cli(*args,**kwargs):raise SystemExit(code)
    monkeypatch.setattr(native.runpy,'run_module',cli)
    monkeypatch.setattr(native.sys,'argv',[])
    native.native('generate_dense_depth',{'config':'fixture'})


def test_failed_native_cli_exit_is_not_swallowed(monkeypatch):
    def cli(*args,**kwargs):raise SystemExit(2)
    monkeypatch.setattr(native.runpy,'run_module',cli)
    monkeypatch.setattr(native.sys,'argv',[])
    with pytest.raises(SystemExit) as error:native.native('depth',{'config':'fixture'})
    assert error.value.code==2


@pytest.mark.parametrize('pixel', [[255, 0, 10], [0, 0, 9], [0, 31, 65], [0, 0, 0]])
def test_non_native_depth_encoding_is_rejected(pixel):
    import numpy as np
    from driversi.rendering.gaussian.depth_completion import validate_native_depth
    with pytest.raises(ValueError):
        validate_native_depth(np.array([[pixel]], dtype=np.uint8), (1, 1))


def test_native_depth_allows_invalid_zero_and_native_range():
    import numpy as np
    from driversi.rendering.gaussian.depth_completion import validate_native_depth
    validate_native_depth(np.array([[[0, 0, 0], [0, 0, 10], [0, 31, 64]]], dtype=np.uint8), (3, 1))


def test_depth_receipt_io_remedy_requires_specific_settled_failure(tmp_path):
    import json
    from driversi.rsi.services.reconstruction_recovery import remedy, eligible
    root=tmp_path/'native';root.mkdir();out=tmp_path/'recovery'
    job_path=tmp_path/'job.json';job_path.write_text(json.dumps({'output':str(root)}))
    job={'reconstruction_job':str(job_path),'output':str(out)}
    (root/'report.json').write_text(json.dumps({'status':'stage_failed','stage':'depth','returncode':0}))
    (root/'depth.log').write_text('184/184')
    previous=out/'remediations/depth-receipt-1'
    (previous/'completion').mkdir(parents=True)
    (previous/'report.json').write_text(json.dumps({'status':'recovery_failed','completion':{'status':'failed'}}))
    log=previous/'completion/worker.log';log.write_text('arbitrary failure')
    (root/'receipts').mkdir();(root/'receipts/depth.json').write_text('{}')
    assert remedy(job)=='depth-receipt-1'
    log.write_text('FileNotFoundError: /depth-completion/report.json.tmp')
    assert remedy(job)=='depth-receipt-io-1'
    assert eligible(job)
    final=out/'remediations/depth-receipt-io-1';final.mkdir();(final/'launch.json').write_text('{}')
    assert not eligible(job)
    (root/'running.json').write_text('{}')
    assert remedy(job) is None
