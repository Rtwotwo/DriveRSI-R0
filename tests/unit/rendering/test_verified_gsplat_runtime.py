import pytest

from driversi.rendering.gaussian import verified_gsplat_runtime as runtime


def witness(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'project_root', lambda: tmp_path)
    path = tmp_path / '.runtimes/cache/gsplat_cuda.so'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'native binary fixture; never executed')
    abi = {'python': '3.10', 'torch': '2.5.1+cu124', 'cuda': '12.4', 'gsplat': '1.4.0', 'cxx11_abi': False}
    value = {'schema_version': 1, 'kind': 'verified_official_gsplat_runtime', 'checks_passed': True,
             'abi': abi, 'producer_sha256': runtime.digest(runtime.__file__),
             'extension': str(path), 'extension_sha256': runtime.digest(path)}
    return value, abi, path


def test_validated_runtime_identity(tmp_path, monkeypatch):
    value, abi, path = witness(tmp_path, monkeypatch)
    assert runtime.validate_witness(value, abi) == path


@pytest.mark.parametrize('change', ['abi', 'producer', 'binary', 'unverified'])
def test_stale_or_incompatible_runtime_rejected(tmp_path, monkeypatch, change):
    value, abi, path = witness(tmp_path, monkeypatch)
    if change == 'abi':
        abi = dict(abi, torch='different')
    elif change == 'producer':
        value['producer_sha256'] = 'wrong'
    elif change == 'binary':
        path.write_bytes(b'changed binary')
    else:
        value['checks_passed'] = False
    with pytest.raises(ValueError):
        runtime.validate_witness(value, abi)
