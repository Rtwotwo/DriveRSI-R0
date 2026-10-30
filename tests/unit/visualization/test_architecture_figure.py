import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from driversi.visualization.architecture_figure import STEPS, render_overview
from driversi.visualization.system import system_overview


ROOT = Path(__file__).resolve().parents[3]


def system():
    return json.loads((ROOT/'src/config/visualization/architecture.json').read_text())


def test_overview_keeps_eight_stage_order_and_distinct_feedback_loops():
    value = system()
    tree = ET.fromstring(render_overview(value))
    links = [e for e in tree.iter() if e.attrib.get('data-step')]
    assert [x.attrib['data-step'] for x in links] == list(STEPS)
    flows = [(e.attrib.get('data-from'),e.attrib.get('data-to')) for e in tree.iter() if e.attrib.get('data-edge-type')=='sequence']
    assert flows == list(zip(STEPS[:-1],STEPS[1:]))
    loops = {e.attrib['data-edge-type']:(e.attrib['data-from'],e.attrib['data-to']) for e in tree.iter() if 'feedback' in e.attrib.get('data-edge-type','')}
    assert loops == {'model-feedback':('retrain','evaluate'),'system-feedback':('skill','retrieve'),'utility-feedback':('evaluate','skill')}
    assert all(x.attrib['target']=='_top' and 'index.html#view=architecture&step=' in x.attrib['href'] for x in links)


def test_invalid_flow_order_cannot_silently_mislabel_links():
    value=system();value['steps'][0],value['steps'][1]=value['steps'][1],value['steps'][0]
    with pytest.raises(ValueError,match='eight-stage'):
        render_overview(value)


def test_figure_escapes_labels_and_keeps_metrics_schematic():
    value=system();value['steps'][0]['short']='<script>not executable</script>'
    svg=render_overview(value);tree=ET.fromstring(svg)
    assert '<script>' not in svg
    text=''.join(tree.itertext())
    assert '目标流程 / 非实验结果' in text and '源位姿PSNR：按实验配置' in text
    assert '0 < p < 0.6' in text and 'p > 0.95' in text
    assert '机制示意' in text


def test_stage_projection_is_registered_and_excludes_unstructured_extra_payload(tmp_path):
    value=system(); path=tmp_path/'src/config/visualization/architecture.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps(value))
    cfg=json.loads((ROOT/'src/config/visualization/platform.json').read_text())
    projected=system_overview(cfg,tmp_path)
    assert len(projected['steps'])==8
    assert 'Recall@K' in projected['steps'][3]['acceptance']
    value['steps'][3]['raw_query_tokens']=['must-not-publish'];path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='declared'):
        system_overview(cfg,tmp_path)


def test_builder_still_publishes_legacy_config_without_architecture(tmp_path, monkeypatch):
    from driversi.visualization import build as module
    site=tmp_path/'site'
    cfg=tmp_path/'config.json'
    cfg.write_text(json.dumps({'schema_version':1,'site_root':str(site),'output':str(site/'platform')}))
    monkeypatch.setattr(module,'snapshot',lambda _:{'system':None,'experiments':[],'services':[],'scenes':[],'whitebox':[],'harness':[]})
    result=module.build(cfg)
    assert result['status']=='built'
    assert json.loads((site/'platform/platform.json').read_text())['system'] is None
    assert not (site/'platform/architecture-overview.svg').exists()
    assert '<section id="systemFlow" aria-label="八步 RSI 流程" hidden>' in (site/'platform/index.html').read_text()


def test_timeline_matches_actual_anchor_and_native_supervision_window():
    tree = ET.fromstring(render_overview(system()))
    times = lambda kind: [float(e.attrib['data-time']) for e in tree.iter()
                          if e.attrib.get('data-window') == kind]
    assert times('history') == [.5, 1., 1.5, 2.]
    assert times('future') == [2.5 + .5*i for i in range(10)]
    assert '锚点 2 s' in ''.join(tree.itertext())
    assert 'PDMS评2–6 s' in ''.join(tree.itertext())
