import json

import pytest

from driversi.rendering.gaussian import source_preflight, source_refinement, source_repair_campaign as campaign


@pytest.mark.parametrize("scores,expected,status", [([25.,25.1],2,"source_threshold_met"), ([24.5,24.5],2,"insufficient_repair_progress"), ([24.6],1,"round_budget_exhausted")])
def test_bounded_repair_retains_source_lineage_and_never_admits(tmp_path, monkeypatch, scores, expected, status):
    asset = {"bundle_root": "original"}
    source = tmp_path / "original.json"
    source.write_text(json.dumps(dict(source_mode="canonical_source_pose_without_motion_v1", asset=asset, mean_psnr_db=24.5)))
    monkeypatch.setattr(campaign, "project_root", lambda: tmp_path)
    seen = []
    def refine(previous, output, **kwargs):
        seen.append(str(previous))
        return {"child_asset": str(output / "child")}
    def inspect(child, sensor, output, **kwargs):
        assert kwargs['minimum_source_psnr_db']==25.
        output.mkdir(parents=True)
        record = dict(mean_psnr_db=scores[len(seen)-1], frames=[dict(tick=5,measured_psnr_db=21.)],asset=child)
        (output / "report.json").write_text(json.dumps(record))
        return record
    monkeypatch.setattr(source_refinement,"run",refine)
    monkeypatch.setattr(source_preflight,"run",inspect)
    result = campaign.run(source,asset,"sensors",tmp_path/"exp"/"new",rounds=len(scores))
    assert len(seen)==expected
    assert result["status"]==status
    assert result["trainable"] is False
    if len(seen)>1:
        assert seen[1]==result["rows"][0]["source_report"]
    assert all(r["repair_holdout_mean_psnr_db"]==21. for r in result["rows"])
    assert json.loads(source.read_text())["asset"]==asset


def test_27_repair_does_not_stop_at_25_or_exactly27(tmp_path,monkeypatch):
    asset={'bundle_root':'original'};p=tmp_path/'source.json'
    p.write_text(json.dumps(dict(source_mode='canonical_source_pose_without_motion_v1',asset=asset,mean_psnr_db=25.04)))
    monkeypatch.setattr(campaign,'project_root',lambda:tmp_path)
    scores=iter([26.5,27.,27.1]);seen=[]
    def refine(previous,output,**kwargs):
        seen.append(str(previous));return {'child_asset':str(output/'child')}
    def inspect(child,sensor,output,**kwargs):
        assert kwargs['minimum_source_psnr_db']==27.
        output.mkdir(parents=True);d=dict(mean_psnr_db=next(scores),frames=[dict(tick=5,measured_psnr_db=25.)])
        (output/'report.json').write_text(json.dumps(d));return d
    monkeypatch.setattr(source_refinement,'run',refine);monkeypatch.setattr(source_preflight,'run',inspect)
    result=campaign.run(p,asset,'sensors',tmp_path/'exp/new',rounds=3,minimum_source_psnr_db=27.)
    assert len(seen)==3 and result['mean_psnr_db']==27.1
    assert result['next_action']=='fresh_branch_admission' and not result['trainable']
