"""Read-only, bounded-memory candidate diagnosis; never used as training data."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('CHAINFLOW_ENABLE_RAY', '0')
os.environ.setdefault('CHAINFLOW_USE_RAY_SCORE', '0')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import sys
if not hasattr(np, '_core'):
    import numpy.core as nc
    sys.modules['numpy._core'] = nc
    for sub in ('numeric','multiarray','umath','_multiarray_umath','_dtype','_dtype_ctypes','_internal','_methods','fromnumeric','arrayprint','records','getlimits'):
        try:
            sys.modules['numpy._core.'+sub] = __import__('numpy.core.'+sub, fromlist=[sub])
        except ImportError:
            pass
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from navsim.common.dataloader import SceneLoader, MetricCacheLoader
from navsim.common.dataclasses import SensorConfig, Trajectory
from navsim.evaluate.pdm_score import pdm_score, transform_trajectory, get_trajectory_as_array
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex as MM, WeightedMetricIndex as WM
from navsim.agents.chainflow_fpd_online_student.training.privileged_inputs import PrivilegedInputBuilder
from navsim.agents.chainflow_fpd_online_student.training.teacher_supervisor import TeacherSupervisor


def rows(path):
    with open(path) as f:
        return {r['token']:r for r in csv.DictReader(f)
                if len(r.get('token',''))==16 and r.get('valid','').lower()=='true'}


def safe_row(r):
    return all(float(r[k]) >= 1-1e-6 for k in ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound'))


def pool_metrics(mc,poses,simulator,scorer):
    ego=mc.ego_state;sampling=simulator.proposal_sampling
    trajectories=[mc.trajectory]+[transform_trajectory(Trajectory(p),ego) for p in poses]
    states=np.stack([get_trajectory_as_array(t,sampling,ego.time_point) for t in trajectories])
    simulated=simulator.simulate_proposals(states,ego)
    scorer.score_proposals(simulated,mc.observation,mc.centerline,mc.route_lane_ids,mc.drivable_area_map)
    multi=scorer._multi_metrics.prod(axis=0)
    raw=scorer._progress_raw*multi
    # Each candidate is compared with the SAME reference, not with the best of
    # the other 63 candidates. The latter changes the official progress label.
    denominator=np.maximum(raw[0],raw[1:])
    progress=np.ones(len(poses),dtype=np.float64)
    moving=denominator>scorer._config.progress_distance_threshold
    progress[moving]=raw[1:][moving]/denominator[moving]
    progress[(~moving)&(multi[1:]==0)]=0
    weighted=scorer._weighted_metrics[:,1:].copy()
    weighted[WM.PROGRESS]=progress
    weights=scorer._config.weighted_metrics_array
    score=multi[1:]*(weighted*weights[:,None]).sum(0)/weights.sum()
    return np.stack([scorer._multi_metrics[MM.NO_COLLISION,1:],scorer._multi_metrics[MM.DRIVABLE_AREA,1:],
                     progress,weighted[WM.TTC],weighted[WM.COMFORTABLE],weighted[WM.DRIVING_DIRECTION],score],axis=-1)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--eval-config',required=True)
    ap.add_argument('--train-config',required=True)
    ap.add_argument('--student-csv',required=True)
    ap.add_argument('--teacher-csv',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--uniform',type=int,default=64)
    ap.add_argument('--failure',type=int,default=64)
    ap.add_argument('--memory-gib',type=float,default=6)
    a=ap.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    c=OmegaConf.load(a.eval_config);tr=OmegaConf.load(a.train_config)
    sr,te=rows(a.student_csv),rows(a.teacher_csv)
    if set(sr)!=set(te):raise ValueError('Evaluation token universes differ')
    order=lambda t:hashlib.sha256(('candidate-audit-20261010:'+t).encode()).hexdigest()
    uniform=sorted(sr,key=order)[:a.uniform]
    failures=sorted([t for t in sr if safe_row(te[t]) and not safe_row(sr[t])],key=order)[:a.failure]
    tokens=list(dict.fromkeys(uniform+failures))
    meta={'uniform_tokens':uniform,'failure_tokens':failures,'evaluation_only':True,
          'sampling':'fixed hash uniform plus separately reported teacher-safe/student-unsafe subset',
          'student_checkpoint':str(c.agent.checkpoint_path),'teacher_checkpoint':str(tr.agent.config.teacher_checkpoint),
          'inference_batch_size':1,'precision':'bf16-mixed','memory_limit_gib':a.memory_gib}
    mf=out/'manifest.json'
    if mf.exists() and json.loads(mf.read_text())!=meta:raise ValueError('Output manifest mismatch')
    mf.write_text(json.dumps(meta,indent=2))
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cudnn.benchmark=False
    free,total=torch.cuda.mem_get_info()
    if free < (a.memory_gib+2)*2**30:raise RuntimeError('Insufficient GPU slack; diagnostic does not wait or evict training')
    torch.cuda.set_per_process_memory_fraction(a.memory_gib*2**30/total)
    agent=instantiate(c.agent);agent.initialize();agent.requires_grad_(False);agent.cuda().eval()
    privileged=PrivilegedInputBuilder(tr.agent.config)
    teacher=TeacherSupervisor(tr.agent.config.teacher_checkpoint,tr.agent.config.teacher_sha256,tr.agent.config)
    fb=agent.get_feature_builders()[0]
    scene_filter=instantiate(c.train_test_split.scene_filter)
    scene_filter.tokens=tokens
    scene_filter.log_names=list(dict.fromkeys(sr[t]['log_name'] for t in tokens))
    sc=SensorConfig(cam_f0=[3,7],cam_b0=[3,7],cam_l0=[3,7],cam_r0=[3,7],cam_l1=[],cam_l2=[],cam_r1=[],cam_r2=[],lidar_pc=[])
    loader=SceneLoader(sensor_blobs_path=Path(c.sensor_blobs_path),data_path=Path(c.navsim_log_path),scene_filter=scene_filter,sensor_config=sc)
    cache=MetricCacheLoader(Path(c.metric_cache_path))
    simulator,scorer=instantiate(c.simulator),instantiate(c.scorer)
    results=[]
    for i,t in enumerate(tokens):
        dest=out/(t+'.json')
        if dest.exists():results.append(json.loads(dest.read_text()));continue
        started=time.monotonic()
        scene=loader.get_scene_from_token(t)
        feat={k:v.unsqueeze(0).cuda() for k,v in fb.compute_features(scene.get_agent_input()).items()}
        priv=privileged.compute_scene_features(scene)
        target={'teacher_future_image':priv['future_image'].unsqueeze(0).cuda(),
                'teacher_future_metadata':priv['future_metadata'].unsqueeze(0).cuda()}
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            pred=agent(feat)
            pool=pred['proposals'].detach().float()
            ts=teacher.score_pool(pool,teacher.context(feat,target))
        mc=cache.get_from_token(t)
        metrics=pool_metrics(mc,pool[0].cpu().numpy(),simulator,scorer)
        scores=pred['pdm_score'][0].float().cpu().numpy()
        teacher_scores=ts['pdm_score'][0].float().cpu().numpy()
        selected=int(scores.argmax());teach_idx=int(teacher_scores.argmax());oracle=int(metrics[:,-1].argmax())
        safe=(metrics[:,[0,1,3]]>=1-1e-6).all(-1)
        # Standard scorer is called separately for selected/oracle/teacher choices,
        # so progress normalization matches official one-trajectory evaluation.
        official={}
        for idx in dict.fromkeys([selected,teach_idx,oracle]):
            val=pdm_score(mc,Trajectory(pool[0,idx].cpu().numpy()),simulator.proposal_sampling,simulator,scorer)
            official[str(idx)]=asdict(val)
        result={'token':t,'log_name':sr[t]['log_name'],'student_idx':selected,'teacher_idx':teach_idx,'oracle_idx':oracle,
                'student_score':float(metrics[selected,-1]),'teacher_same_pool_score':float(metrics[teach_idx,-1]),
                'oracle_score':float(metrics[oracle,-1]),'student_safe':bool(safe[selected]),'teacher_safe':bool(safe[teach_idx]),
                'has_safe':bool(safe.any()),'best_safe_score':float(metrics[safe,-1].max()) if safe.any() else None,
                'csv_student_score':float(sr[t]['score']),'csv_teacher_score':float(te[t]['score']),
                'candidate_metrics':metrics.tolist(),'student_rank':scores.tolist(),'teacher_rank':teacher_scores.tolist(),
                'official':official,'seconds':time.monotonic()-started,
                'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30}
        for prefix,obj in [('student',pred),('teacher',ts)]:
            result[prefix+'_probabilities']={k:torch.sigmoid(v[0].float()).cpu().tolist() for k,v in obj['pred_logit'].items()}
        np.savez_compressed(out/(t+'.npz'),proposals=pool[0].cpu().numpy())
        dest.write_text(json.dumps(result,indent=2));results.append(result)
        print(json.dumps({k:result[k] for k in ('token','student_score','teacher_same_pool_score','oracle_score','student_safe','has_safe','seconds','peak_gpu_gib')}),flush=True)
        del feat,target,pred,pool,ts
        summarize(out,results,uniform,failures)
    summarize(out,results,uniform,failures)
    (out/'completed.json').write_text(json.dumps({'scenes':len(results),'requested':len(tokens)}))


def summarize(out,results,uniform,failures):
    summary={}
    for name,cohort in [('uniform',uniform),('failure_diagnostic',failures)]:
        rr=[r for r in results if r['token'] in cohort]
        if not rr:continue
        summary[name]={'n':len(rr),'requested':len(cohort)}
        for k in ('student_score','teacher_same_pool_score','oracle_score','student_safe','teacher_safe','has_safe','csv_student_score','csv_teacher_score'):
            summary[name][k]=float(np.mean([r[k] for r in rr]))
        summary[name]['batched_official_max_delta']=max(abs(r['candidate_metrics'][int(j)][-1]-v['score']) for r in rr for j,v in r['official'].items())
        summary[name]['csv_student_mean_abs_delta']=float(np.mean([abs(r['official'][str(r['student_idx'])]['score']-r['csv_student_score']) for r in rr]))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':main()
