"""Delivery contract and durable failure/resume checks; real media tested separately."""
import copy
import json
from pathlib import Path
import zipfile

import pytest

from driversi.rendering.whitebox.delivery import MEMBERS, package_delivery, reviewed_prompt, verify_bundle, stage_finished
from driversi.rendering.whitebox.exporter import sha256

TOKEN = '0123456789abcdef'


@pytest.fixture
def assets(tmp_path):
    bundle = tmp_path / 'bundle'
    renders = bundle / 'source-renders'
    renders.mkdir(parents=True)
    (bundle / "source.json").write_text(json.dumps({"token":TOKEN}))
    for name in MEMBERS[:-1]:
        (renders / name).write_bytes(('encoded-test-content-' + name).encode())
    review = tmp_path / 'review.json'
    review.write_text(json.dumps(dict(token=TOKEN, status='human_reviewed_for_generation_trial',
                                     claims=[dict(text='Pale buildings and trees.', evidence_ids=['CAM_F0_f0'])])))
    return bundle, tmp_path / 'download', review


def test_exact_five_file_delivery_and_no_rewrite_on_resume(assets):
    bundle, output, review = assets
    first = package_delivery(bundle, output, TOKEN, review)
    archive = Path(first['archive'])
    original = archive.stat().st_mtime_ns
    second = package_delivery(bundle, output, TOKEN, review)
    assert second['status'] == 'reused' and second['archive_sha256'] == first['archive_sha256']
    assert archive.stat().st_mtime_ns == original
    with zipfile.ZipFile(archive) as z:
        assert sorted(z.namelist()) == sorted(MEMBERS)
        prompt = z.read(MEMBERS[-1]).decode()
        assert 'Pale buildings and trees.' in prompt
        assert 'NOT physical paint' in prompt and 'walkway, NOT necessarily grass' in prompt
        assert all(z.read(n) == (bundle / 'source-renders' / n).read_bytes() for n in MEMBERS[:-1])


def test_changed_video_does_not_silently_reuse_or_overwrite(assets):
    bundle, output, review = assets
    first = package_delivery(bundle, output, TOKEN, review)
    (bundle / 'source-renders' / MEMBERS[0]).write_bytes(b'changed')
    with pytest.raises(ValueError, match='Existing delivery differs'):
        package_delivery(bundle, output, TOKEN, review)
    assert sha256(first['archive']) == first['archive_sha256']


def test_extra_archive_member_rejected(assets):
    bundle, output, review = assets
    first = package_delivery(bundle, output, TOKEN, review)
    with zipfile.ZipFile(first['archive'], 'a') as z:
        z.writestr('metadata.json', '{}')
    with pytest.raises(ValueError, match='Existing delivery differs'):
        package_delivery(bundle, output, TOKEN, review)


@pytest.mark.parametrize('change', [dict(token='different'), dict(status='unreviewed'),
                                     dict(claims=[dict(text='A red car', evidence_ids=[])]), dict(claims=[])])
def test_wrong_or_unreviewed_appearance_rejected(assets, change):
    _, _, review = assets
    value = json.loads(review.read_text()); value.update(change); review.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        reviewed_prompt(review, TOKEN)


def make_bundle(path, split='navtrain', role='source_log'):
    path.mkdir()
    cameras = [dict(camera_id=c, width=640, height=360) for c in ['CAM_F0', 'CAM_L0', 'CAM_R0', 'CAM_B0']]
    value = dict(token=TOKEN, source_split=split, role=role,
                 timing=dict(interval_seconds=.1, duration_seconds=7.0),
                 frames=[dict(tick=t, cameras=copy.deepcopy(cameras)) for t in range(71)])
    (path / 'source.json').write_text(json.dumps(value))
    (path / 'manifest.json').write_text(json.dumps(dict(files={'source.json': sha256(path / 'source.json')})))


@pytest.mark.parametrize('split,role', [('navtest', 'source_log'), ('navtrain', 'physical_branch')])
def test_test_split_and_branch_cannot_be_certified_as_original_log(tmp_path, split, role):
    bundle = tmp_path / 'bundle'; make_bundle(bundle, split, role)
    with pytest.raises(ValueError, match='original navtrain'):
        verify_bundle(bundle, TOKEN)


def test_changed_source_rejected_even_with_same_token(tmp_path):
    bundle = tmp_path / 'bundle'; make_bundle(bundle)
    assert verify_bundle(bundle, TOKEN)['token'] == TOKEN
    (bundle / 'source.json').write_text('{}')
    with pytest.raises(ValueError, match='Changed bundle file'):
        verify_bundle(bundle, TOKEN)


@pytest.mark.parametrize('text', ['', '{"files":', '{}', '[]'])
def test_interrupted_manifest_is_incomplete_not_finished(tmp_path, text):
    path = tmp_path / 'manifest.json'; path.write_text(text)
    assert not stage_finished(path, ('files', 'producer_sha256', 'frame_count'))
    path.write_text(json.dumps(dict(files={}, producer_sha256='test', frame_count=71)))
    assert stage_finished(path, ('files', 'producer_sha256', 'frame_count'))


def test_planar_delivery_discloses_height_proxy(assets):
    bundle, output, review = assets
    (bundle / 'source.json').write_text(json.dumps({'display_profile':'planar_consistent_v1'}))
    result=package_delivery(bundle,output,TOKEN,review)
    with zipfile.ZipFile(result['archive']) as z:
        prompt=z.read('video-generation-prompt.txt').decode()
        assert 'vertical/pitch/roll motion are not represented' in prompt
        assert len(z.namelist())==5


def test_old_size_sampling_cannot_be_presented_as_new_renders(assets, monkeypatch):
    import sys
    from types import SimpleNamespace
    from driversi.rendering.whitebox.delivery import verify_renders
    # Mismatch must fail before any media decoding, even if all content hashes match.
    monkeypatch.setitem(sys.modules, 'imageio_ffmpeg', SimpleNamespace())
    bundle, _, _ = assets
    (bundle/'source.json').write_text(json.dumps({'dimension_sampling':'source_bracket_linear_v1'}))
    for name in ['map.json','palette.json']:
        (bundle/name).write_text('{}')
    manifest = dict(role='source', frame_count=71, video_frame_count=70, fps=10,
        source_sha256=sha256(bundle/'source.json'), map_sha256=sha256(bundle/'map.json'),
        palette_sha256=sha256(bundle/'palette.json'), dimension_sampling='previous_annotation_hold_v1')
    (bundle/'source-renders/manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='box-size sampling'):
        verify_renders(bundle)
