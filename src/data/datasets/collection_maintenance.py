"""Conservative cleanup of orphan empty lock files in the exact project dataset."""
import fcntl
import os
import re
import time
from pathlib import Path

from driversi.agents.backends.native import project_root,digest
from driversi.configuration import read_mapping
from driversi.runtime.budget import write_json


def cleanup(dataset, output, *, execute=False):
    root=Path(dataset)
    expected=project_root().resolve()/'dataset'
    if root.is_symlink() or root.resolve()!=expected:raise ValueError('Only the explicit DriveRSI/dataset directory is allowed')
    output=Path(output)
    if not output.resolve().is_relative_to(project_root().resolve()/'exp'):raise ValueError('Cleanup receipt belongs under exp')
    # Never delete data, admitted references, staging, collection locks or manifests.
    keep={p.name for group in ('samples','admitted','.staging')
          for p in (root/'synthetic_navsim'/group).iterdir()}
    lockroot=root/'synthetic_navsim/.locks'
    candidates=[p for p in lockroot.glob('*.lock') if re.fullmatch(r'[0-9a-f]{64}\.lock',p.name)
        and p.stem not in keep and not p.is_symlink() and p.stat().st_size==0
        and time.time()-p.stat().st_mtime>60]
    before={str(p):digest(p) for p in (root/'manifests').glob('*/*.json')}
    result=dict(root=str(root.resolve()),status='planned',candidates=[str(p) for p in candidates],
        removed=[],synthetic_payloads_removed=0,manifest_hashes_before=before)
    output.parent.mkdir(parents=True,exist_ok=True);write_json(output,result)
    if execute:
        for p in Path('/proc').glob('[0-9]*/cmdline'):
            if int(p.parent.name)==os.getpid():continue
            try:text=p.read_bytes().replace(b'\0',b' ').decode(errors='replace')
            except OSError:continue
            if str(project_root()) in text and any(k in text for k in ('driversi.rsi.experiments','resident_postprocess','navsim_teacher')):
                raise ValueError('Dataset-producing worker is active; postpone cleanup')
        with (root/'.collection.lock').open('a') as guard:
            fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for p in candidates:
                if p.resolve().parent!=lockroot.resolve():raise ValueError('Cleanup target escaped dataset locks')
                with p.open('r+') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    if p.stat().st_size or any((root/'synthetic_navsim'/g/p.stem).exists() for g in ('samples','admitted','.staging')):
                        raise ValueError('Orphan lock acquired new data')
                    p.unlink();result['removed'].append(str(p));write_json(output,result)
        if any(digest(p)!=sha for p,sha in before.items()):raise ValueError('Dataset manifest changed during cleanup')
        result['status']='completed';result['manifest_hashes_unchanged']=True;write_json(output,result)
    return result
