"""Owned NAVSIM v2 navtest replay; report scene and EC coverage separately."""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import fields
from datetime import datetime
import hashlib
import json
import lzma
import multiprocessing
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import OmegaConf
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import relative_to_absolute_poses

from navsim.common.dataclasses import PDMResults
from navsim.common.enums import SceneFrameType
from .pickle_compat import load
from navsim.evaluate.pdm_score import pdm_score
from . import one_stage, two_stage

from driversi.agents.backends.native import project_root
ROOT = project_root()
CONFIG = ROOT / 'src/config/evaluation/epdms'
STATE = None

def read_index(path):
    result = {}
    for csv in sorted((path/'metadata').glob('*.csv')):
        for raw in csv.read_text().splitlines()[1:]:
            value = raw.strip().strip('"')
            if not value:
                continue
            item = Path(value)
            if not item.is_absolute():
                item = path/item
            token = item.parent.name
            if token in result and result[token] != str(item):
                raise ValueError(f'Conflicting cache index for {token}')
            result[token] = str(item)
    if not result:
        raise ValueError(f'No metric-cache index under {path}')
    return result

def read_cache(path):
    with lzma.open(path,'rb') as f:
        cache = load(f)
    for field in ['scene_type','map_parameters','past_human_trajectory','current_tracked_objects']:
        if not hasattr(cache,field):
            raise ValueError(f'{path} is not a NAVSIM v2 metric cache: missing {field}')
    return cache

def benchmark_tokens(benchmark):
    if benchmark == 'navtest':
        tokens = set(OmegaConf.load(ROOT/'src/vendor/driving/navsim-v2.2/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml').tokens)
        return tokens, set(), {}
    raw = OmegaConf.load(CONFIG/'navhard_mapping.yaml').reactive_all_mapping
    mappings = {(a,b): [tuple(pair) for pair in pairs] for a,b,pairs in raw}
    first = {t for pair in mappings for t in pair}
    second = {t for pairs in mappings.values() for pair in pairs for t in pair}
    if first & second:
        raise ValueError('Overlapping benchmark stages')
    return first,second,mappings

def predictions(path, benchmark, first, second):
    with path.open('rb') as f:
        data=load(f)
    names = ['predictions'] if benchmark=='navtest' else ['first_stage_predictions','second_stage_predictions']
    merged={}
    for name,expected in zip(names,[first,second]):
        values=data.get(name)
        if not isinstance(values,list) or len(values)!=1 or not isinstance(values[0],dict):
            raise ValueError(f'{name} must be a one-seed list containing a token→Trajectory dictionary')
        actual=set(values[0])
        if actual != expected:
            raise ValueError(f'{name}: missing {len(expected-actual)}, extra {len(actual-expected)}; expected {len(expected)}')
        for token,t in values[0].items():
            poses=np.asarray(t.poses)
            if poses.shape!=(8,3) or not np.isfinite(poses).all():
                raise ValueError(f'Invalid trajectory {token}: expected finite [8,3]')
            if not np.isclose(t.trajectory_sampling.interval_length,0.5):
                raise ValueError(f'Wrong trajectory sampling: {token}')
        merged.update(values[0])
    return merged

def initialize(config):
    global STATE
    cfg=OmegaConf.create(config)
    simulator=instantiate(cfg.simulator)
    scorer=instantiate(cfg.scorer)
    policy=instantiate(cfg.traffic_agents_policy[cfg.traffic_agents],simulator.proposal_sampling)
    assert simulator.proposal_sampling==scorer.proposal_sampling
    STATE=(simulator,scorer,policy)

def score_batch(tasks):
    simulator,scorer,policy=STATE
    rows=[]
    for token,trajectory,path in tasks:
        cache=read_cache(path)
        row,states=pdm_score(cache,trajectory,simulator.proposal_sampling,simulator,scorer,policy)
        end=StateSE2(*map(float,trajectory.poses[-1]))
        endpoint=relative_to_absolute_poses(cache.ego_state.rear_axle,[end])[0]
        row['valid']=True
        row['token']=token
        row['log_name']=cache.log_name
        row['frame_type']=cache.scene_type
        row['start_time']=cache.timepoint.time_s
        row['endpoint_x']=endpoint.x
        row['endpoint_y']=endpoint.y
        row['start_point_x']=cache.ego_state.rear_axle.x
        row['start_point_y']=cache.ego_state.rear_axle.y
        row['ego_simulated_states']=[states]
        rows.append(row)
    return rows

def score_all(tasks,cfg,workers):
    config=OmegaConf.to_container(cfg,resolve=True)
    batches=iter([tasks[i:i+4] for i in range(0,len(tasks),4)])
    rows=[]
    if workers==1:
        initialize(config)
        for batch in batches:
            rows.extend(score_batch(batch))
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize,initargs=(config,)) as pool:
            pending=set()
            for _ in range(2*workers):
                batch=next(batches,None)
                if batch is not None:pending.add(pool.submit(score_batch,batch))
            while pending:
                done,pending=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    rows.extend(future.result())  # Fail rather than averaging only successful scenes.
                    batch=next(batches,None)
                    if batch is not None:pending.add(pool.submit(score_batch,batch))
                print(f'EPDMS: {len(rows)}/{len(tasks)} scenes',flush=True)
    return pd.concat(rows,ignore_index=True).sort_values('token').reset_index(drop=True)

def aggregate(raw,benchmark,mappings,sampling):
    if benchmark=='navtest':
        adjacent=one_stage.infer_start_adjacent_mapping(raw)
        if adjacent:
            table=one_stage.create_scene_aggregators(adjacent,raw.copy(),sampling)
        else:
            table=raw.drop(columns=['ego_simulated_states']).copy()
            table['two_frame_extended_comfort']=np.nan
        table=one_stage.compute_final_scores(table)
    else:
        table=two_stage.create_scene_aggregators(mappings,raw.copy(),sampling)
        table=two_stage.compute_final_scores(table)
    cols=[c for c in table if (any(f.name in c for f in fields(PDMResults)) or c in ['two_frame_extended_comfort','score']) and c!='pdm_score']
    if not np.isfinite(table['score']).all():
        raise ValueError('Non-finite final scene scores')
    if benchmark=='navtest':
        output=table[['token','valid']+cols].copy()
        avg=table[cols].mean()
        output.loc[len(output)]={'token':'average_all_frames','valid':True,**avg.to_dict()}
        return output,float(avg['score'])
    combined,stage1,stage2=two_stage.calculate_individual_mapping_scores(table[cols+['token','weight']],mappings)
    if not np.isfinite([combined['score'],stage1['score'],stage2['score']]).all():
        raise ValueError('Incomplete two-stage aggregation')
    output=table[['token','valid','score']].copy()
    for col in cols:
        if col=='score':continue
        output[col+'_stage_one']=table[col].where(table.frame_type==SceneFrameType.ORIGINAL)
        output[col+'_stage_two']=table[col].where(table.frame_type==SceneFrameType.SYNTHETIC)
    for tag,values in [('stage_one',stage1),('stage_two',stage2),('combined',combined)]:
        row={'token':'extended_pdm_score_'+tag,'valid':True,'score':float(values['score'])}
        for col in cols:
            if col=='score':continue
            if tag in ['stage_one','combined']:row[col+'_stage_one']=stage1[col]
            if tag in ['stage_two','combined']:row[col+'_stage_two']=stage2[col]
        output.loc[len(output)]=row
    return output,float(combined['score'])

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--benchmark',choices=['navtest'],required=True)
    ap.add_argument('--submission',type=Path)
    ap.add_argument('--metric-cache',type=Path)
    ap.add_argument('--map-root',type=Path)
    ap.add_argument('--output-dir',type=Path)
    ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--check',action='store_true',help='Check official tokens/cache and optional submission, without scoring')
    ap.add_argument('--smoke',action='store_true',help='Diagnostic subset, never a full benchmark score')
    ap.add_argument('--save-raw',action='store_true',help='Retain simulated states for parity auditing')
    args=ap.parse_args()
    if args.workers<1:ap.error('--workers must be positive')
    cfg=OmegaConf.load(CONFIG/(args.benchmark+'.yaml'))
    if args.metric_cache:cfg.metric_cache_path=str(args.metric_cache)
    if args.map_root:cfg.map_root=str(args.map_root)
    first,second,mappings=benchmark_tokens(args.benchmark)
    expected=first|second
    assert len(expected)==cfg.expected_scenes,(len(expected),cfg.expected_scenes)
    index=read_index(Path(cfg.metric_cache_path))
    if expected-set(index):raise ValueError(f'Missing {len(expected-set(index))} cache tokens')
    with ThreadPoolExecutor(max_workers=32) as pool:
        missing=[p for p,ok in zip([index[t] for t in sorted(expected)],pool.map(lambda p:Path(p).is_file(),[index[t] for t in sorted(expected)])) if not ok]
    if missing:raise FileNotFoundError(f'{len(missing)} missing cache files; first: {missing[0]}')
    read_cache(index[sorted(first)[0]])
    if second:read_cache(index[sorted(second)[0]])
    pred=None
    if args.submission:pred=predictions(args.submission,args.benchmark,first,second)
    print(f'Preflight OK: {args.benchmark}, {len(first)} original + {len(second)} synthetic scenes',flush=True)
    if args.check:return
    if pred is None:ap.error('--submission is required for scoring')
    selected=expected
    if args.smoke:
        if args.benchmark=='navhard':
            key=next(iter(mappings));mappings={key:mappings[key]}
            selected=set(key)|{t for pair in mappings[key] for t in pair}
        else:
            first_path=Path(index[sorted(first)[0]])
            same_log=[t for t in sorted(first) if Path(index[t]).parents[2]==first_path.parents[2]][:32]
            ordered=sorted(same_log,key=lambda t:read_cache(index[t]).timepoint.time_s)
            selected=set(ordered[:16])
    output=args.output_dir or ROOT/'exp/epdms'/args.benchmark/datetime.now().strftime('%Y.%m.%d.%H.%M.%S')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'summary.json').exists():raise FileExistsError('Use a new output directory; existing results are not overwritten')
    OmegaConf.save(cfg,output/'effective_config.yaml')
    try:
        raw=score_all([(t,pred[t],index[t]) for t in sorted(selected)],cfg,args.workers)
        assert len(raw)==len(selected) and set(raw.token)==selected and not raw.token.duplicated().any()
        if args.save_raw:raw.to_pickle(output/'raw_scores.pkl')
        table,score=aggregate(raw,args.benchmark,mappings,instantiate(cfg.proposal_sampling))
        csv=output/(datetime.now().strftime('%Y.%m.%d.%H.%M.%S')+'.csv')
        table.to_csv(csv,index=False)
        summary={'benchmark':args.benchmark,'protocol':'NAVSIM v2 EPDMS','scope':'smoke' if args.smoke else 'full',
                 'full_benchmark_coverage':not args.smoke,'online_submission':False,'scenes':len(selected),'failed':0,'score':score,'epdms':score*100,
                 'traffic_agents':cfg.traffic_agents,'csv':str(csv),'submission':str(args.submission.resolve()),
                 'submission_sha256':hashlib.sha256(args.submission.read_bytes()).hexdigest(),
                 'metric_cache':cfg.metric_cache_path,'model_inference_run':False}
        if args.benchmark == 'navtest':
            scenes = table[table.token.isin(selected)]
            computed = int(scenes.two_frame_extended_comfort.notna().sum())
            summary['extended_comfort'] = dict(
                computed=computed, missing=len(selected) - computed,
                all_scenes_have_complete_metrics=computed == len(selected),
                missing_policy='upstream_one_stage_omit_metric_and_weight',
                usable_for_generated_data_admission=False)
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary,indent=2),flush=True)
    except Exception:
        (output/'failure.txt').write_text(traceback.format_exc())
        raise

if __name__=='__main__':main()
