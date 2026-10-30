import json
import sqlite3

import pytest

from driversi.data.assets import reconstruction_preparation as mod


@pytest.mark.parametrize('missing', [False, True])
def test_measured_sources_prepare_but_never_grant_asset_readiness(tmp_path, monkeypatch, missing):
    token, name = 'aa'*8, 'training_log'
    frames = []
    root = tmp_path / 'sensors'
    root.mkdir()
    for i in range(23):
        cam = {}
        for c in mod.CAMERAS:
            relative = f'{name}/{c}/{i}.jpg'
            p = root / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'fixture')
            cam[c] = dict(data_path=relative)
        lidar = f'{name}/lidar/{i}.pcd'
        p = root / lidar
        p.parent.mkdir(parents=True, exist_ok=True)
        if not missing or i != 0:
            p.write_bytes(b'fixture')
        frames.append(dict(token=token if i == 6 else str(i), log_name=name,
                           timestamp=i*500000, lidar_path=lidar, cams=cam,
                           ego2global_translation=[i, 0, 0], map_location='us-ma-boston'))
    monkeypatch.setattr(mod, 'load_log', lambda *a, **kw: frames)
    db = tmp_path / (name + '.db')
    with sqlite3.connect(db) as conn:
        conn.execute('create table lidar_pc(token blob)')
        conn.execute('insert into lidar_pc values(?)', (bytes.fromhex(token),))
    index = tmp_path / 'index.json'
    index.write_text(json.dumps(dict(split='navtrain', expected_tokens=[token])))
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(dict(train_index=str(index), candidates=[dict(token=token,source_log=name)],
                                      logs_root=str(tmp_path), db_root=str(tmp_path),
                                      sensor_roots=dict(camera=[str(root)], lidar=[str(root)]))))
    result = mod.prepare(config, tmp_path / 'result')
    row = result['candidates'][0]
    assert row['source_window_complete'] is (not missing)
    assert row['camera_files'] == 184 and row['lidar_files'] == 23-int(missing)
    assert not row['asset_ready'] and not row['reconstruction_started']
    assert not result['full_preprocessor_readiness']
    assert mod.prepare(config, tmp_path / 'result') == result
    index.write_text(json.dumps(dict(split='navtest', expected_tokens=[token])))
    with pytest.raises(ValueError, match='navtrain'):
        mod.prepare(config, tmp_path / 'forbidden')
