"""Harness-pushed current snapshot; no independent monitor or public deployment."""
import math
import time
import threading
from contextlib import contextmanager
from pathlib import Path

from driversi.agents.backends.native import project_root
from driversi.configuration import read_mapping
from driversi.runtime.budget import write_json

_last = {}
_lock = threading.Lock()


def refresh(campaign, *, force=False):
    if not _lock.acquire(blocking=False):return
    try:return _refresh(campaign,force=force)
    finally:_lock.release()


def _refresh(campaign, *, force=False):
    campaign=Path(campaign)
    if not (campaign/'operations-policy.json').exists():return
    if not force and time.monotonic()-_last.get(str(campaign),0)<30:return
    _last[str(campaign)]=time.monotonic()
    try:
        from driversi.data.datasets.curriculum_lineage import build,public
        from driversi.rsi.experiments.continual_curriculum import project,state_path
        value=build(project_root()/'dataset',campaign)
        state=read_mapping(state_path(campaign));current=project(state)
        operations=public(value)
        if state['policy'].get('cohort_mode')=='global_50_min1_v1':
            operations['cohort_contract']=dict(total=50,minimum_per_source=1,scope='current_problem')
        case=next((c for c in state['cases'] if c['id']==state['active']),None)
        if case:
            search=case.get('source_search',{})
            operations['source_search']=dict(limit=search.get('limit',case.get('selected_sources')),
                selected_sources=case.get('selected_sources'),pending_assets=len(case.get('pending_source_tokens',[])),
                levels=[3,5,8,10],reason=search.get('reason'),
                fresh_attempts=search.get('fresh_attempts'),fresh_yield=search.get('fresh_yield'),
                duplicate_fraction=search.get('duplicate_fraction'),original_visual_ranks=search.get('original_visual_ranks',[]),
                expansions=len(case.get('source_expansions',[])),location_used=False)
        if case and case.get('current_round'):
            folder=Path(case['current_round']);cfg=read_mapping(folder/'generation-config.json')
            model=read_mapping(cfg['training_model_config'])
            from driversi.rsi.decisions.training_recipe import choose
            import yaml
            lr=yaml.safe_load((project_root()/'agents/chainflow/native/navsim/planning/script/config/common/agent/chainflow_vla_stage1.yaml').read_text())['lr_args']
            recipe=choose(case['target'],case['round'],model['settings']['distributed_training']['devices'],cfg['max_pair_optimizer_steps'])
            recipe['scaled_base_lr']=float(lr['base_lr'])*math.sqrt(recipe['global_batch']/int(lr['base_batch_size']))
            recipe['qualification']='planned_until_per_source_cohort_and_native_training_receipts_pass'
            operations['planned_recipe']=recipe
            operations['generation_resources']=dict(physical_workers=cfg.get('physical_prefetch_workers',1),
                render_gpus=cfg.get('resident_postprocessing',{}).get('gpus',[]),
                native_gpu=cfg.get('production_gpu'),features='deferred',captions='deferred')
            progress=folder/'waves/wave-0001/production-status.json'
            if progress.exists():
                p=read_mapping(progress)
                operations['phase']={k:p.get(k) for k in ('status','candidate','qualified_samples','prepared_trajectories','attempted','rendered','error_type')}
            wave=folder/'waves/wave-0001'
            queue=wave/'trajectory-queue.json'
            if queue.exists():
                q=read_mapping(queue)
                operations['trajectory_collection']={k:q.get(k) for k in ('stage','prepared','admitted','target')}
            plan=wave/'trajectory-family-plan.json'
            if plan.exists():
                profiles={}
                for a in read_mapping(plan)['assignments']:
                    p=a.get('adaptive_diversity_policy',{})
                    if p.get('version')!='scene_conditioned_diversity_v2':continue
                    profiles[a['token']]=dict(token=a['token'],cluster=p['scene_context'].get('cluster',{}).get('label'),
                        skill=p['skill_id'],action=p['action'],revision=p['revision'],
                        ade_m=p['minimum_ade_m'],path_m=p['minimum_path_ade_m'],speed_mps=p['minimum_speed_mae_mps'],
                        speed_duration_s=p['minimum_speed_duration_s'],version=p['version'])
                operations['scene_diversity']=list(profiles.values())
        current['operations']=operations
        write_json(campaign/'operations.json',current)
        cfg=read_mapping(project_root()/'src/config/visualization/platform.json')
        if cfg.get('display_mode')=='current_only' and Path(cfg.get('current_campaign','')).resolve()==campaign.resolve():
            output=Path(cfg['output']).resolve();site=Path(cfg['site_root']).resolve()
            if output!=site/'platform':raise ValueError('Invalid existing platform destination')
            if case:
                try:
                    from driversi.visualization.curriculum_candidates import build
                    index=read_mapping(campaign/'asset-recovery-policy.json')['factory']['train_index']
                    operations['candidate_gallery']=build(case,index,output)
                except (OSError,ValueError,KeyError) as exc:
                    operations['candidate_gallery_error']=type(exc).__name__
            write_json(output/'current-operations.json',dict(schema_version=1,read_only=True,
                generated_at=value['generated_at'],current=current))
    except Exception as exc:
        write_json(campaign/'operations-error.json',dict(error_type=type(exc).__name__,reason=str(exc),time=time.time()))


def pulse(wave):
    campaign=next((p for p in Path(wave).parents if (p/'private/state.json').exists()),None)
    if campaign:refresh(campaign)


@contextmanager
def publishing(campaign):
    """Lifetime-bound progress publication, not a recurring desktop monitor."""
    stop=threading.Event()
    def tick():
        while not stop.wait(30):refresh(campaign,force=True)
    thread=threading.Thread(target=tick,daemon=True,name='harness-progress-publication')
    thread.start()
    try:yield
    finally:
        stop.set();thread.join(timeout=2);refresh(campaign,force=True)
