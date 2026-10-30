import json
from PIL import Image
import pytest
from driversi.visualization.curriculum_candidates import build


def test_train_front_only_and_selected(tmp_path):
    token='0123456789abcdef';folder=tmp_path/'log/CAM_F0';folder.mkdir(parents=True)
    Image.new('RGB',(30,20),(23,42,1)).save(folder/'image.jpg')
    index=tmp_path/'index.json';index.write_text(json.dumps(dict(split='navtrain',complete=True,
        records=[dict(token=token,source_log='log',sensor_root=str(tmp_path),image_files=[str(folder/'image.jpg')])])) )
    case=dict(candidates=[dict(token=token,source_log='log',best_rank=1,max_similarity=.9)],
        source_selected_tokens=[token])
    result=build(case,index,tmp_path/'site');assert result[0]['selected']
    assert result[0]['image']=='current-candidates/'+token+'.jpg'
    assert (tmp_path/'site'/result[0]['image']).exists()
    bad=json.loads(index.read_text());bad['split']='navtest';index.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='navtrain'):build(case,index,tmp_path/'site')


def test_wrong_log_and_wrong_camera_are_rejected(tmp_path):
    token='0123456789abcdef';index=tmp_path/'index.json'
    index.write_text(json.dumps(dict(split='navtrain',complete=True,
        records=[dict(token=token,source_log='log',sensor_root='/',image_files=['/log/CAM_R0/image.jpg'])])))
    with pytest.raises(ValueError,match='front camera'):
        build(dict(candidates=[dict(token=token,source_log='log')]),index,tmp_path/'site')
    with pytest.raises(ValueError,match='absent'):
        build(dict(candidates=[dict(token=token,source_log='different')]),index,tmp_path/'site')
