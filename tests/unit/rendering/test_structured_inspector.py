"""Coverage and transfer contracts for the free-view structured inspector."""

import copy
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from driversi.rendering.camera_review.gallery import dump
from driversi.rendering.camera_review.transfer import display_map, optimize

ASSETS = Path(__file__).parents[3] / "src/rendering/camera_review"


def test_display_map_retains_all_identities_properties_and_holes():
    data = {
        "features": [
            {
                "id": "lane",
                "layer": "lanes",
                "properties": {"successors": ["next"]},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[0, 0], [5, 0.001], [10, 0], [10, 10], [0, 10], [0, 0]],
                        [[3, 3], [3, 7], [7, 7], [7, 3], [3, 3]],
                    ],
                },
            },
            {
                "id": "light",
                "layer": "traffic_lights",
                "properties": {},
                "geometry": {"type": "Point", "coordinates": [1, 2]},
            },
        ]
    }
    before = copy.deepcopy(data)
    result = display_map(data)
    assert data == before
    assert result["source_feature_count"] == 2
    assert result["maximum_hausdorff_error_m"] <= 0.02
    for old, new in zip(data["features"], result["features"], strict=True):
        assert {k: v for k, v in old.items() if k != "geometry"} == {
            k: v for k, v in new.items() if k != "geometry"
        }
    assert len(result["features"][0]["geometry"]["coordinates"]) == 2


@pytest.mark.parametrize("value", [-1, 0.03, float("nan")])
def test_invalid_display_tolerance_rejected(value):
    with pytest.raises(ValueError):
        display_map({"features": []}, value)


def test_bootstrap_and_camera_transfer_restore_original_bytes(tmp_path):
    mapping = dump(tmp_path / "map.json", {"features": []})
    chunk = dump(tmp_path / "chunks/front.json", {"frames": [{"tick": 0}]})
    chunk["path"] = "chunks/front.json"
    scene = {
        "map": mapping,
        "cameras": {"front": {"chunks": [chunk]}},
        "frames": [{"tick": 0}],
        "source_frames": [{"timestamp_us": 0}],
    }
    dump(tmp_path / "scene.json", scene)
    result = optimize(tmp_path)
    boot = json.loads((tmp_path / "bootstrap.json").read_text())
    current = json.loads((tmp_path / "scene.json").read_text())
    assert current["frames"] == scene["frames"]
    assert current["source_frames"] == scene["source_frames"]
    for ref in [
        boot["scene"],
        current["map"],
        current["display_map"],
        current["cameras"]["front"]["chunks"][0],
    ]:
        original = (tmp_path / ref["path"]).read_bytes()
        assert gzip.decompress((tmp_path / ref["gzip"]["path"]).read_bytes()) == original
        assert hashlib.sha256(original).hexdigest() == ref["sha256"]
    assert result["feature_count"] == 0
    before = (tmp_path / "bootstrap.json").read_bytes()
    optimize(tmp_path)
    assert (tmp_path / "bootstrap.json").read_bytes() == before


@pytest.fixture
def javascript():
    quickjs = pytest.importorskip("quickjs")
    ctx = quickjs.Context()
    ctx.eval(
        "if(!Array.prototype.at)Array.prototype.at=function(i){return this[i<0?this.length+i:i]}"
    )
    for name in ["structure-geometry.js", "structure.js"]:
        ctx.eval((ASSETS / name).read_text())
    return ctx


def test_all_supported_geometry_types_have_render_primitives(javascript):
    result = json.loads(
        javascript.eval("""JSON.stringify(
    StructuralWhitebox.build([
      {id:'p',layer:'lights',geometry:{type:'MultiPoint',coordinates:[[1,2],[3,4]]}},
      {id:'c',layer:'lines',geometry:{type:'GeometryCollection',geometries:[
        {type:'LineString',coordinates:[[0,0],[1,1]]},
        {type:'Polygon',coordinates:[[[0,0],[1,0],[0,1],[0,0]]]}
      ]}}
    ]))""")
    )
    assert result["count"] == 2
    assert [x["type"] for x in result["items"]].count("point") == 2
    assert {x["type"] for x in result["items"]} == {"polygon", "line", "point"}


def test_absent_world_actor_kept_only_in_source_and_views_are_distinct(javascript):
    result = json.loads(
        javascript.eval("""JSON.stringify((()=>{
    const source={objects:[{id:'a',kind:'pedestrian',pose:[0,0,1,0],dimensions:[1,1,2]}]};
    const world={actors:[{id:'a',present:false},{id:'ego',present:true,position:[1,2],yaw:.3,dimensions:[4,2,2],controlled:true}]};
    const o=StructuralWhitebox.frameObjects({source_frames:[source]},world,source);
    const cameras=['front','chase','orbit','bird'].map(m=>StructuralWhitebox.camera(o.world[0],m,{yaw:1,pitch:.5,distance:30,x:0,y:0}));
    return {o,cameras};
    })())""")
    )
    assert len(result["o"]["source"]) == 1
    assert len(result["o"]["world"]) == 1
    assert result["o"]["source"][0]["kind"] == "pedestrian"
    assert len({json.dumps(c, sort_keys=True) for c in result["cameras"]}) == 4
    assert result["o"]["world"][0]["pose_xyz_yaw"] == [1, 2, 0, 0.3]


def test_front_hides_own_outline_without_dropping_actor_record(javascript):
    result = json.loads(
        javascript.eval("""JSON.stringify((()=>{
      const ctx=new Proxy({}, {get:(o,k)=>o[k]||(()=>{}),set:(o,k,v)=>(o[k]=v,true)});
      const obj={id:'ego',kind:'vehicle',pose_xyz_yaw:[10,0,0,0],dimensions_lwh:[4,2,2]};
      return StructuralWhitebox.render({width:960,height:540,getContext:()=>ctx},StructuralWhitebox.build([]),
        {camera:{x:0,y:0,z:1,yaw:0,pitch:0},objects:[obj],layers:new Set(),kinds:new Set(['vehicle']),hideActor:'ego'});
    })())""")
    )
    assert result["totalObjects"] == 1
    assert result["filteredObjects"] == 1
    assert result["visibleObjects"] == 0
