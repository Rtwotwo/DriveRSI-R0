"""Private train-side candidate front-camera gallery for the current problem."""
import re
from pathlib import Path
from PIL import Image
from driversi.configuration import read_mapping


def build(case, index_path, output):
    """Publish only verified navtrain camera thumbnails, without test identifiers."""
    index=read_mapping(index_path)
    if index.get('split')!='navtrain' or not index.get('complete'):
        raise ValueError('Complete official navtrain source inventory required')
    candidates=case['candidates']
    lookup={(r['token'],r['source_log']):r for r in index['records']
            if r.get('token') in {c['token'] for c in candidates}}
    folder=Path(output)/'current-candidates';folder.mkdir(parents=True,exist_ok=True)
    selected=set(case.get('source_selected_tokens',[]))
    result=[]
    for c in candidates:
        token=c['token'];log=c['source_log']
        if not re.fullmatch('[0-9a-f]{16}',token) or Path(log).name!=log:
            raise ValueError('Invalid training source identity')
        rec=lookup.get((token,log))
        if not rec:raise ValueError('Candidate absent from official train inventory')
        paths=[Path(p) for p in rec['image_files'] if Path(p).parent.name=='CAM_F0']
        root=Path(rec['sensor_root']).resolve()
        if (len(paths)!=1 or paths[0].parent.parent.name!=log
                or not paths[0].resolve().is_relative_to(root)):
            raise ValueError('Unique front camera evidence required')
        target=folder/(token+'.jpg')
        if not target.exists() and paths[0].is_file():
            with Image.open(paths[0]) as img:
                img=img.convert('RGB');img.thumbnail((720,480))
                temp=folder/(token+'.tmp');img.save(temp,format='JPEG',quality=85);temp.replace(target)
        result.append(dict(token=token,source_log=log,rank=c.get('best_rank'),
            similarity=c.get('max_similarity'),selected=token in selected,
            image='current-candidates/'+token+'.jpg' if target.exists() else None))
    return result
