"""Visual descriptors from real, irregular histories; never a temporal repair."""

import hashlib
import io
import pickle
import time
from pathlib import Path

from .corpus import write_report
from .features import VisionEncoder, feature_profile
from .observations import file_hash
from .store import identity
from .vectors import VectorStore

KIND = 'qwen35_observed_irregular_history_v1'


def recovery_profile(config):
    base = feature_profile(config)
    return {**base, 'kind':KIND, 'recovery':{
        'base_profile':identity(base), 'source_kind':'original_observed_images',
        'sampling':'last_four_records_actual_timestamps',
        'synthetic_images':False, 'training_history_repaired':False,
        'implementation':file_hash(Path(__file__))}}


def observed_history(scene, cameras, max_side=512, *, require_irregular=True):
    """Decode the original four records using the native image transform.

    No frames after the anchor are examined. All actual timestamps are retained.
    The opt-out of require_irregular is only useful for native-loader parity tests.
    """
    from PIL import Image

    record, desc = scene['record'], scene['descriptor']
    allowed = {'CAM_F0','CAM_L0','CAM_R0','CAM_B0','CAM_L1','CAM_R1','CAM_L2','CAM_R2'}
    if not cameras or len(set(cameras))!=len(cameras) or not set(cameras)<=allowed:
        raise ValueError('Invalid camera selection')
    if not 128<=max_side<=1920:
        raise ValueError('max_side outside supported bounds')
    with Path(record['log_path']).open('rb') as handle:
        frames=pickle.load(handle)
    anchor=record['anchor_index']
    if type(anchor) is not int or not 3<=anchor<len(frames):
        raise ValueError('Four real historical records required')
    history=frames[anchor-3:anchor+1]
    if history[-1]['token']!=record['token'] or any(f['log_name']!=record['source_log'] for f in history):
        raise ValueError('History identity/source mismatch')
    times=[int(f['timestamp']) for f in history]
    gaps=[b-a for a,b in zip(times,times[1:])]
    if any(g<=0 for g in gaps):
        raise ValueError('Observed history must increase strictly in time')
    if times[-1]!=desc['history']['anchor_timestamp_us']:
        raise ValueError('Anchor timestamp changed')
    if 'actual_intervals_us' in desc['history'] and gaps!=desc['history']['actual_intervals_us']:
        raise ValueError('Actual history differs from indexed intervals')
    regular=all(450000<=g<=550000 for g in gaps)
    if require_irregular and (regular or desc['history']['continuous'] is not False):
        raise ValueError('Recovery is restricted to observed irregular histories')
    root=Path(record['sensor_root']).resolve()
    evidence,images=[],[]
    for i,frame in enumerate(history):
        for camera in cameras:
            path=(root/frame['cams'][camera]['data_path']).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError('Missing image or path outside sensor root')
            data=path.read_bytes()
            with Image.open(io.BytesIO(data)) as image:
                image=image.convert('RGB')
                original=list(image.size)
                image.thumbnail((max_side,max_side),Image.Resampling.LANCZOS)
                images.append(image.copy())
                evidence.append({'id':f'h{i}_{camera}', 'camera':camera, 'history_index':i,
                    'timestamp_us':times[i], 'relative_time_s':(times[i]-times[-1])/1e6,
                    'path':str(path),'sha256':hashlib.sha256(data).hexdigest(),
                    'original_size':original,'input_size':list(image.size),
                    'source_kind':'original_observed_image','synthetic':False,
                    'history_sampling':'regular_2hz' if regular else 'observed_irregular',
                    'temporal_training_admitted':False})
    return evidence,images


def recover_batch(catalog,config,tokens,output,*,backend_factory=VisionEncoder):
    import fcntl

    if not tokens or len(tokens)>1000 or len(tokens)!=len(set(tokens)):
        raise ValueError('Explicit unique bounded recovery tokens required')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with Path(str(catalog.path)+'.feature.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        data=recovery_profile(config)
        store=VectorStore(catalog,writable=True)
        base=data['recovery']['base_profile']
        if store.profile_data(base)!=feature_base(data):
            raise ValueError('Recovery encoder differs from existing base profile')
        # Validate all selected rows before any vector writes.
        for token in tokens:
            scene=catalog.scene(token)
            if scene['record']['kind']!='original_'+catalog.split:
                raise ValueError('Recovery scene split mismatch')
            old=catalog.db.execute('SELECT status,error FROM vectors WHERE profile=? AND token=?',(base,token)).fetchone()
            if old is None or old['status']!='failed' or old['error']!='ValueError: Irregular history is not silently padded':
                raise ValueError('Only explicit prior irregular-history failures may be recovered')
        profile=store.profile(data)
        write_report(output/'profile.json',data)
        report={'split':catalog.split,'base_profile':base,'profile':profile,'requested':len(tokens),
                'ready':0,'reused':0,'failed':0,'complete':False,'synthetic_images':0,
                'training_history_repaired':False,'trainable':False,'rows':[],'status':'running'}
        backend=None;start=time.monotonic()
        try:
            for token in tokens:
                evidence=[]
                try:
                    evidence,images=observed_history(catalog.scene(token),config['cameras'],config['max_side'])
                except Exception as exc:
                    error=type(exc).__name__+': '+str(exc)
                    report['failed']+=1
                    report['rows'].append({'token':token,'status':'failed','error':error})
                    store.put(token,profile,[],None,{},error)
                    write_report(output/'report.json',report)
                    continue
                if store.completed(token,profile,evidence):
                    status='reused'
                else:
                    try:
                        if backend is None:backend=backend_factory(config)
                        values,cost=backend(images)
                        store.put(token,profile,evidence,values,cost)
                    except Exception as exc:
                        error=type(exc).__name__+': '+str(exc)
                        report['failed']+=1
                        report['rows'].append({'token':token,'status':'failed','error':error})
                        store.put(token,profile,evidence,None,{},error)
                        raise
                    status='ready'
                report[status]+=1
                report['rows'].append({'token':token,'status':status,
                    'actual_history_times_s':[e['relative_time_s'] for e in evidence if e['camera']==config['cameras'][0]]})
                report['elapsed_seconds']=time.monotonic()-start
                write_report(output/'report.json',report)
            report['complete']=True
            report['status']='complete_with_failures' if report['failed'] else 'complete'
        except Exception as exc:
            report['error']=type(exc).__name__+': '+str(exc)
            report['status']='failed'
            raise
        finally:
            report['elapsed_seconds']=time.monotonic()-start
            write_report(output/'report.json',report)
        return report


def feature_base(data):
    result={k:v for k,v in data.items() if k!='recovery'}
    result['kind']='qwen35_native_vision_features_v1'
    return result


def enable_recovery_view(catalog,base_profile,recovered_profile):
    """Read-connection-only overlay; original tables/profile IDs stay intact.

    Native ready rows win. Supplemental rows are exposed under the base search
    namespace only in this temporary view, with provenance returned separately.
    """
    store=VectorStore(catalog)
    base=store.profile_data(base_profile)
    recovered=store.profile_data(recovered_profile)
    if recovered.get('kind')!=KIND or recovered.get('recovery',{}).get('base_profile')!=base_profile or feature_base(recovered)!=base:
        raise ValueError('Recovery profile is not encoder-compatible with base')
    if any(len(x)!=64 or any(c not in '0123456789abcdef' for c in x) for x in (base_profile,recovered_profile)):
        raise ValueError('Stored profile identities required')
    if catalog.db.execute("SELECT 1 FROM sqlite_temp_master WHERE name='vectors'").fetchone():
        if getattr(catalog, '_history_recovery_view', None) == (base_profile, recovered_profile):
            return
        raise ValueError('A recovery view is already active on this connection')
    b,s=base_profile,recovered_profile
    catalog.db.execute(f'''CREATE TEMP VIEW vectors AS
        SELECT v.* FROM main.vectors v
        WHERE NOT (v.profile='{b}' AND v.status!='ready' AND EXISTS (
            SELECT 1 FROM main.vectors z WHERE z.profile='{s}' AND z.token=v.token AND z.status='ready'))
        UNION ALL
        SELECT '{b}' AS profile,v.token,v.input_id,v.status,v.evidence,v.cost,v.error,
               v.dimension,v.views,v.centroid,v.image_vectors
        FROM main.vectors v WHERE v.profile='{s}' AND v.status='ready' AND NOT EXISTS (
            SELECT 1 FROM main.vectors z WHERE z.profile='{b}' AND z.token=v.token AND z.status='ready')''')
    catalog._history_recovery_view = (base_profile, recovered_profile)


def retrieval_provenance(catalog,report,base_profile,recovered_profile):
    """Expose real storage identities and avoid claims of aligned time histories."""
    def source(token):
        row=catalog.db.execute("SELECT profile FROM main.vectors WHERE token=? AND status='ready' AND profile IN (?,?) ORDER BY CASE WHEN profile=? THEN 0 ELSE 1 END LIMIT 1",(token,base_profile,recovered_profile,base_profile)).fetchone()
        if row is None:raise ValueError('No native or recovered vector for selected token')
        return {'storage_profile':row[0],'observed_irregular_history':row[0]==recovered_profile,
                'temporal_training_admitted':False}
    report['history_recovery_profile']=recovered_profile
    report['query_feature_provenance']=source(report['query_token'])
    for row in report['results']:
        row['feature_provenance']=source(row['token'])
        row['nominal_2hz_pair']=not (row['feature_provenance']['observed_irregular_history'] or report['query_feature_provenance']['observed_irregular_history'])
    report['view_comparison_semantics']='ordered_real_visual_views_not_temporal_alignment'
    return report
