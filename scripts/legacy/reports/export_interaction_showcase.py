"""Build an evidence-linked gallery dataset from sealed server-side rollouts.

No policy, simulation, or rendering runs here. Source arrays and pixels are read
without modification. Images are hard-linked (or copied across filesystems).
"""
from __future__ import annotations
import argparse
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STATE_COLUMNS = ['x','y','z','yaw','vx','vy','vz','goal_x','goal_y','stopped','collision','offroad','reached_goal']
KINDS = {1:'vehicle',2:'pedestrian',3:'cyclist',10:'other_object'}
CASE_DEFINITIONS = [
 dict(id='pittsburgh-recovery', title='Pittsburgh · 教师避碰恢复', city='Pittsburgh', kind='teacher_recovery',
      subtitle='同一到达状态，扰动延续与 DriveRL TTS 分支对照',
      summary='扰动对照最终碰撞；教师分支完成评分窗口，反事实 PDM 为 98.28 / 100。轨迹准入通过，视觉仍未准入。',
      execution='exp/worldengine-rsi-v1/tasks/worldengine-sealed-render-v2/reuse/execution',
      render='exp/worldengine-rsi-v1/tasks/worldengine-sealed-render-v2/render/render', backend='worldengine', branch='teacher',
      source_images='outputs/demonstrations/worldengine-rsi-v1/pittsburgh-new-branch',
      comparison='outputs/demonstrations/worldengine-rsi-v1/pittsburgh-matched-source-v2',
      warnings=['新视角画质与地面高程未独立验收。','物理世界中两处路障缺外观绑定；另有重建资产不在物理对象表。','实拍与新分支位姿不同，不能用逐像素差证明新视角真实。']),
 dict(id='boston-control-search',title='Boston · 控制搜索纠正',city='Boston',kind='control_search',
      subtitle='9 个有界控制候选，选中 candidate-07',
      summary='扰动延续 94.74 → 搜索分支 95.18 / 100；对应四路相机重放与逐步 BEV。轨迹通过，视觉拒绝。',
      execution='exp/navsim-trajectory-first-v1/tasks/travel6-search-gs-export-v4/teacher/execution',
      render='exp/navsim-trajectory-first-v1/tasks/travel6-search-gs-export-v4/render/render/session', backend='mtgs', branch='teacher',
      warnings=['重建节点与当前日志对象身份匹配数为 0；相机移动不能证明交通外观完整响应。','分支图片保留模糊、残影与配准问题，不能进入合格训练集。']),
 dict(id='boston-teacher-failure',title='Boston · 教师失败也保留',city='Boston',kind='teacher_failure',
      subtitle='教师候选不会因名称而自动成为正确监督',
      summary='扰动对照 97.80、DriveRL 候选 92.47 / 100；方向与背景接触另行检查，均未获得训练准入。',
      execution='exp/navsim-teacher-loop-v1/tasks/boston-teacher-gs-export-v1/teacher/execution',
      render='exp/navsim-teacher-loop-v1/tasks/boston-teacher-gs-export-v1/render/render/session',backend='mtgs',branch='teacher',
      warnings=['教师低于 95 分且存在背景接触记录；不能按高质量正例展示。','高分对照仍因方向 / 接触等独立条件不合格。','重建与物理对象完整性未验收。']),
 dict(id='boston-asset-transport',title='Boston · 动态资产诊断',city='Boston',kind='asset_diagnostic',
      subtitle='同一物理分支，比较资产修正前后的成像',
      summary='展示实际扰动分支与资产绑定修正前后；该分支 90.83 / 100，尚无合格教师分支。',
      execution='exp/gs-asset-loop-v2/tasks/travel1-asset-search-v1/teacher/execution',
      render='outputs/demonstrations/gs-asset-loop-v2/after',comparison='outputs/demonstrations/gs-asset-loop-v2/before',
      backend='mtgs',branch='baseline',
      warnings=['这里渲染的是扰动 baseline，不能标成教师优化轨迹。','before / after 复用同一物理状态，只比较渲染资产处理。','参与者资产与新视角质量尚未全部验收。']),
 dict(id='boston-braking-search',title='Boston · 制动纠正轨迹',city='Boston',kind='trajectory_only',
      subtitle='完整执行轨迹补充：当前没有对应完整 RGB 序列',
      summary='同一到达状态搜索后，反事实 PDM 94.43 → 95.32 / 100；轨迹准入通过。',
      execution='exp/navsim-trajectory-first-v1/tasks/travel2-braking-control-search-v1/teacher/execution',
      backend='none',branch='teacher',
      warnings=['本案例仅展示结构化执行证据；没有借用其他场景的相机图片。','轨迹通过不等于视觉数据或训练收益通过。']),
]

def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def write_json(p,value):
    """Publish one JSON atomically; never reuse another writer's temporary file."""
    p.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n'
    name=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=p.parent,
                                         prefix='.'+p.name+'-',suffix='.tmp',delete=False) as stream:
            name=stream.name;stream.write(raw);stream.flush();os.fsync(stream.fileno())
        os.replace(name,p)
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


def checked_case_path(output, relative):
    """Catalog links are local to this gallery, never arbitrary files or URLs."""
    if (not isinstance(relative,str) or not relative or '\\' in relative
            or '?' in relative or '#' in relative or ':' in relative):
        raise ValueError('Invalid case data path')
    path=Path(relative)
    if path.is_absolute() or '..' in path.parts or path.suffix!='.json':
        raise ValueError('Case data must be a relative JSON path')
    resolved=(output/path).resolve()
    if not resolved.is_relative_to(output.resolve()):
        raise ValueError('Case data escaped gallery root')
    return resolved


def validate_case_identity(case):
    if not isinstance(case,dict):raise ValueError('Catalog case must be an object')
    cid=case.get('id');source=case.get('source_log')
    if not isinstance(cid,str) or not cid or cid.strip()!=cid:
        raise ValueError('Catalog case ID must be a nonempty string')
    if not isinstance(source,str) or not source or source.strip()!=source:
        raise ValueError(f'Case {cid!r} has no valid source_log')
    return cid


def assert_same_source(existing, incoming):
    cid=validate_case_identity(incoming)
    if existing is not None:
        if existing.get('id')!=cid or existing.get('source_log')!=incoming['source_log']:
            raise ValueError(f'Case ID {cid!r} cannot replace a different source_log')
        if (existing.get('scenario_id') and incoming.get('scenario_id')
                and existing['scenario_id']!=incoming['scenario_id']):
            raise ValueError(f'Case ID {cid!r} cannot replace a different scenario_id')


def read_catalog(output):
    """Fail closed on malformed entries instead of silently dropping them."""
    path=output/'data/cases.json'
    if not path.exists():return {'schema_version':1,'cases':[]}
    catalog=read_json(path)
    if (not isinstance(catalog,dict) or type(catalog.get('schema_version')) is not int
            or catalog['schema_version']!=1 or not isinstance(catalog.get('cases'),list)):
        raise ValueError('Unsupported or malformed case catalog')
    seen=set();paths=set()
    for case in catalog['cases']:
        cid=validate_case_identity(case)
        if cid in seen:raise ValueError(f'Duplicate catalog case ID {cid!r}')
        seen.add(cid);data_path=checked_case_path(output,case.get('data'))
        if data_path in paths:raise ValueError('Two case IDs share the same data file')
        paths.add(data_path)
        if not data_path.is_file():raise ValueError(f'Missing data for registered case {cid!r}')
        data=read_json(data_path)
        if not isinstance(data,dict) or data.get('id')!=cid:
            raise ValueError(f'Case data identity mismatch for {cid!r}')
        assert_same_source(case,data.get('metadata'))
    return catalog


def merge_catalog(catalog, updates):
    """Update IDs in place, append new IDs, preserve every unrelated entry/field."""
    old={validate_case_identity(c):c for c in catalog['cases']}
    if len(old)!=len(catalog['cases']):raise ValueError('Duplicate existing case ID')
    incoming={}
    for case in updates:
        cid=validate_case_identity(case)
        if cid in incoming:raise ValueError(f'Duplicate update case ID {cid!r}')
        assert_same_source(old.get(cid),case);incoming[cid]=case
    merged=[]
    for case in catalog['cases']:
        update=incoming.pop(case['id'],None)
        merged.append({**case,**update} if update is not None else dict(case))
    merged.extend(dict(c) for c in incoming.values())
    return {**catalog,'cases':merged}

def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def contiguous_points(points,present):
    """Never draw an artificial connecting line across an absent object interval."""
    out=[];segment=[]
    for xy,ok in zip(points,present):
        if ok:segment.append([float(xy[0]),float(xy[1])])
        elif segment:
            out.append(segment);segment=[]
    if segment:out.append(segment)
    return out

class Exporter:
    def __init__(self, root, output):
        self.root=root.resolve();self.output=output.resolve();self.media=[];self.files={}
        self.output.mkdir(parents=True,exist_ok=True)
    def path(self,rel):
        p=(self.root/rel).resolve()
        if not p.is_relative_to(self.root):raise ValueError('Source outside project')
        return p
    def link(self,src,case_id,name):
        src=Path(src).resolve()
        if not src.is_relative_to(self.root) or not src.is_file():raise ValueError(f'Invalid media source {src}')
        dest=self.output/'media'/case_id/name;dest.parent.mkdir(parents=True,exist_ok=True)
        sha=digest(src)
        if dest.exists():
            if digest(dest)!=sha:raise ValueError(f'Refusing to replace existing different media {dest}')
            mode='existing'
        else:
            try:os.link(src,dest);mode='hardlink'
            except FileExistsError:
                if digest(dest)!=sha:raise ValueError(f'Concurrent different media {dest}')
                mode='existing'
            except OSError as exc:
                if exc.errno not in (errno.EXDEV,errno.EPERM,errno.EACCES,errno.ENOTSUP):raise
                with src.open('rb') as reader, dest.open('xb') as writer:shutil.copyfileobj(reader,writer)
                mode='copy'
        if digest(dest)!=sha:raise ValueError('Media hash mismatch')
        relative=dest.relative_to(self.output).as_posix()
        self.media.append({'path':relative,'source':str(src),'sha256':sha,'bytes':src.stat().st_size,'mode':mode})
        return relative
    def preflight_catalog(self):
        """Check identity conflicts before touching any media or per-case JSON."""
        catalog=read_catalog(self.output);existing={c['id']:c for c in catalog['cases']}
        expected=[]
        for cfg in CASE_DEFINITIONS:
            scene_path=self.path(cfg['execution'])/'scene.npz'
            with np.load(scene_path,allow_pickle=False) as scene:
                sm=json.loads(str(scene['metadata']))
            identity={'id':cfg['id'],'source_log':sm['source_log'],'scenario_id':sm['scenario_id']}
            assert_same_source(existing.get(cfg['id']),identity)
            target=self.output/'data'/f"{cfg['id']}.json"
            if target.exists():
                prior=read_json(target)
                if not isinstance(prior,dict) or prior.get('id')!=cfg['id']:
                    raise ValueError(f'Refusing to overwrite unrelated case file {target}')
                assert_same_source(prior.get('metadata'),identity)
            expected.append(identity)
        merge_catalog(catalog,expected)
        return catalog

    def export(self,cfg):
        execution=self.path(cfg['execution']);scene_path=execution/'scene.npz';roll_path=execution/(cfg['branch']+'.npz')
        scene=npz(scene_path);roll=npz(roll_path)
        sm=json.loads(str(scene['metadata']));rm=json.loads(str(roll['metadata']));report=read_json(execution/'report.json')
        ids=[str(x) for x in scene['actor_ids']];control={str(x) for x in roll['controlled_ids']}
        if ids!=[str(x) for x in roll['actor_ids']]:raise ValueError('Actor identity/order mismatch')
        states=roll['states'];presence=roll['presence'];dt=float(rm['dt_seconds'])
        assert states.shape[2]==13 and states.shape[:2]==presence.shape and np.isfinite(states).all()
        assert np.allclose(scene['times'][:len(states)],np.arange(len(states))*dt,atol=1e-10)
        assert sm['coordinate_frame']=='map_axes_translated_from_initial_ego'
        render=None;render_dir=None;render_report=None
        if cfg['backend']!='none':
            render_dir=self.path(cfg['render'])
            render_report=render_dir/('manifest.json' if cfg['backend']=='mtgs' else 'report.json')
            render=read_json(render_report)
            assert render['source_log']==sm['source_log']
            if cfg['backend']=='mtgs':
                assert render['native_replay']['source_rollout']==digest(roll_path)
                assert render['inputs']['scene']['sha256']==digest(scene_path)
                assert render['dt_seconds']==dt
            else:
                assert render['inputs'][str(roll_path)]==digest(roll_path)
                assert render['inputs'][str(scene_path)]==digest(scene_path)
        ticks=[int(f['tick']) for f in render['frames']] if render else list(range(0,min(len(states)-1,60)+1,5))
        if len(set(ticks))!=len(ticks) or ticks!=sorted(ticks):raise ValueError('Duplicate/out-of-order render ticks')
        if not ticks or min(ticks)<0 or max(ticks)>=len(states):raise ValueError('Render outside physics')
        cid=cfg['id'];frames=[];exact_rows=0;camera_ids=set()
        source_start=int(sm['source_start_timestamp_us'])
        for fi,tick in enumerate(ticks):
            rf=render['frames'][fi] if render else None;actors=[];cameras=[]
            for j,aid in enumerate(ids):
                st=states[tick,j];pr=bool(presence[tick,j])
                a={'id':aid,'x':float(st[0]),'y':float(st[1]),'yaw':float(st[3]),'speed':float(math.hypot(st[4],st[5])),
                   'length':float(scene['dimensions'][j,0]),'width':float(scene['dimensions'][j,1]),'kind':KINDS.get(int(scene['kinds'][j]),'other_object'),
                   'controlled':aid in control,'present':pr,'alive':bool(pr and st[9]==0) if aid in control else None,
                   'collision':bool(pr and st[10]),'offroad':bool(pr and st[11])}
                actors.append(a)
            if rf and cfg['backend']=='mtgs':
                assert abs(rf['time_seconds']-tick*dt)<1e-10
                by_id={a['id']:a for a in rf['actors']};assert set(by_id)==set(ids)
                for a in actors:
                    b=by_id[a['id']]
                    assert a['present']==b['present']
                    assert np.allclose([a['x'],a['y'],a['yaw'],a['speed']],b['position']+[b['yaw'],b['speed']],atol=1e-8,rtol=0)
                    exact_rows+=1
                for c in rf['cameras']:
                    cam={'id':c['camera_id'],'owner_actor_id':c['owner_actor_id'],'virtual':c['virtual'],
                         'image':self.link(render_dir/c['image'],cid,f"{tick:04d}-{c['camera_id']}-render.png"),
                         'tick':tick,'time_seconds':tick*dt,'simulation_timestamp_us':int(rf['simulation_timestamp_us']),
                         'camera_timestamp_us':c.get('camera_timestamp_us'),'source_camera_timestamp_us':c.get('source_camera_timestamp_us'),
                         'warnings':c.get('quality',{}).get('reasons',[]),'support':c.get('support',{}),
                         'camera_to_world':c.get('camera_to_world'),'K':c.get('K'),'distortion':c.get('distortion')}
                    if c.get('source'):cam['source']=self.link(render_dir/c['source'],cid,f"{tick:04d}-{c['camera_id']}-source.png")
                    if c.get('valid'):cam['valid_mask']=self.link(render_dir/c['valid'],cid,f"{tick:04d}-{c['camera_id']}-valid.png")
                    if cfg.get('comparison'):
                        before_dir=self.path(cfg['comparison']);before=read_json(before_dir/'manifest.json')
                        bf=next(x for x in before['frames'] if x['tick']==tick);bc=next(x for x in bf['cameras'] if x['camera_id']==c['camera_id'])
                        assert before['native_replay']['source_rollout']==render['native_replay']['source_rollout']
                        cam['comparison']=self.link(before_dir/bc['image'],cid,f"{tick:04d}-{c['camera_id']}-before.png")
                        cam['comparison_label']='资产处理修正前（同一物理分支）'
                    cameras.append(cam);camera_ids.add(cam['id'])
            elif rf:
                for key,c in rf['cameras'].items():
                    stamp=int(c['timestamp_us']);residual=stamp-(source_start+round(tick*dt*1e6))
                    assert residual==int(rf['capture_residual_us'])
                    cam={'id':key,'owner_actor_id':'ego','virtual':False,'image':self.link(render_dir/c['file'],cid,f'{tick:04d}-{key}-render.png'),
                         'tick':tick,'time_seconds':tick*dt,'simulation_timestamp_us':source_start+round(tick*dt*1e6),
                         'source_timestamp_us':stamp,'capture_residual_us':residual,'source_token':rf['source_token'],
                         'warnings':['新视角质量未准入','采样使用帧时间而非逐相机快门时间'],
                         'camera_to_world':c.get('camera_to_world'),'K':c.get('distorted_intrinsic'),'distortion':c.get('distortion'),
                         'missing_appearance':rf.get('physical_missing_appearance',[])}
                    source=self.path(cfg['source_images'])/f'{tick:04d}-{key}-source.jpg'
                    if source.is_file():cam['source']=self.link(source,cid,f'{tick:04d}-{key}-source.jpg')
                    comparison=self.path(cfg['comparison'])/f'{tick:04d}-{key}.png'
                    if comparison.is_file():
                        cam['comparison']=self.link(comparison,cid,f'{tick:04d}-{key}-source-pose.png')
                        cam['comparison_label']='原日志位姿 GS（不是同一反事实位姿）'
                    cameras.append(cam);camera_ids.add(key)
            frame={'tick':tick,'time_seconds':round(tick*dt,8),'actors':actors,'cameras':cameras,
                   'phase':'initial' if tick==0 else ('history' if tick<=report.get('arrival_tick',20) else 'future'),
                   'simulation_timestamp_us':source_start+round(tick*dt*1e6)}
            frames.append(frame)
        roads=[scene['road_points'][a:b,:2].tolist() for a,b in zip(scene['road_offsets'][:-1],scene['road_offsets'][1:])]
        routes=[{'actor_id':a,'points':v['route']['points']} for a,v in sm.get('driving_tasks',{}).get('actors',{}).items() if v.get('route',{}).get('points')]
        trajectories=[]
        # All coordinates are actual scene-map positions, never anchor-relative labels.
        for j,aid in enumerate(ids):
            for segment in contiguous_points(states[:max(ticks)+1,j,:2],presence[:max(ticks)+1,j]):
                trajectories.append({'actor_id':aid,'role':'executed','points':segment})
        ego=ids.index('ego')
        for segment in contiguous_points(scene['poses'][:max(ticks)+1,ego,:2],scene['presence'][:max(ticks)+1,ego]):
            trajectories.append({'actor_id':'ego','role':'reference','points':segment})
        base_path=execution/'baseline.npz';baseline=npz(base_path)
        assert [str(a) for a in baseline['actor_ids']]==ids
        for segment in contiguous_points(baseline['states'][:max(ticks)+1,ego,:2],baseline['presence'][:max(ticks)+1,ego]):
            trajectories.append({'actor_id':'ego','role':'baseline','points':segment})
        branches=report.get('branches',{});before=branches.get('baseline',{}).get('score');after=branches.get('teacher',{}).get('score') if cfg['branch']=='teacher' else None
        warnings=['非学习型对抗 Self-Play：固定扰动 / 规则交通 / 教师或搜索候选。','图片用于分支录制；并未参与本组策略决策。',*cfg['warnings']]
        score={'label':'反事实 PDM ×100','before':None if before is None else before['score']*100,'after':None if after is None else after['score']*100,
               'protocol':'navsim-v1-native-math-executed-counterfactual-v1','official_benchmark':False}
        metadata={k:cfg[k] for k in ['id','title','subtitle','city','kind','summary']}
        metadata.update(source_log=sm['source_log'],scenario_id=sm['scenario_id'],duration_seconds=max(ticks)*dt,camera_count=len(camera_ids),frame_count=len(frames),
                        trainable=False,learned_self_play=False,warnings=warnings,score=score,data=f'data/{cid}.json',source_type='existing_sealed_artifacts',
                        executed_branch=cfg['branch'],generator=report.get('teacher') if cfg['branch']=='teacher' else report.get('arrival_method'))
        if frames[0]['cameras']:metadata['thumbnail']=frames[min(4,len(frames)-1)]['cameras'][0]['image']
        evidence={'execution_directory':str(execution),'render_directory':str(render_dir) if render_dir else None,
                  'report_path':str(execution/'report.json'),'render_report_path':str(render_report) if render_report else None,
                  'physics_dt_seconds':dt,'state_columns':STATE_COLUMNS,'coordinate_frame':sm['coordinate_frame'],'origin_xyz':sm['origin_xyz'],
                  'join':'exact recorded tick; actor ID and source rollout hashes checked','render_actor_rows_checked':exact_rows,
                  'image_policy':'original pixels; no inpainting, recoloring, resizing, or cross-scene substitution',
                  'score_protocol':score['protocol'],'logged_benchmark':False,'trainable':False,
                  'rendered_rgb_used_for_actions':False,'learned_self_play':False,'report_fields':{k:report.get(k) for k in ['source_log','token','arrival_tick','history_ticks','label_ticks','teacher','teacher_backend','arrival_method','history_xy_deviation_m','restore_exact','qualified_teacher_branch','charged_joint_steps']},
                  'branches':{k:{x:v.get(x) for x in ['completed_steps','failure','events','score','trajectory_admitted','object_contacts']} for k,v in branches.items() if k in ['baseline','teacher']},
                  'files':{str(p):digest(p) for p in [scene_path,roll_path,base_path,execution/'report.json']}}
        if render_report:evidence['files'][str(render_report)]=digest(render_report)
        data={'id':cid,'metadata':metadata,'roads':roads,'routes':routes,'frames':frames,'trajectories':trajectories,'evidence':evidence}
        write_json(self.output/'data'/f'{cid}.json',data)
        return metadata
    def run(self):
        self.preflight_catalog()
        cases=[self.export(cfg) for cfg in CASE_DEFINITIONS]
        # Re-read immediately before publication so unrelated additions and
        # catalog extension fields survive even a long export operation.
        catalog=merge_catalog(read_catalog(self.output),cases)
        write_json(self.output/'data/cases.json',catalog)
        report={'schema_version':1,'kind':'sealed_showcase_export','cases':len(cases),'catalog_cases':len(catalog['cases']),'preserved_case_ids':[c['id'] for c in catalog['cases'] if c['id'] not in {x['id'] for x in cases}],'rendered_case_count':sum(c['camera_count']>0 for c in cases),
                'distinct_logs':len({c['source_log'] for c in cases}),'frames':sum(c['frame_count'] for c in cases),
                'media_files':len(self.media),'media_bytes':sum(x['bytes'] for x in self.media),'new_rendered_images':0,'new_simulation_steps':0,
                'source_pixels_preserved':True,'all_trainable_false':all(not c['trainable'] for c in cases),'media':self.media}
        write_json(self.output/'data/export-verification.json',report)
        print(json.dumps({k:v for k,v in report.items() if k!='media'},ensure_ascii=False))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();output=args.output or args.root/'viz'
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.export.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        Exporter(args.root,output).run()

if __name__=='__main__':main()
