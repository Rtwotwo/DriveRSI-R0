"""Epoch-aware DDP transport around the released ChainFlow model and loss.

Invoked by the existing owned-process lifecycle. Each rank consumes a disjoint
slice of one global batch. World-size changes on resume are explicitly rejected.
"""
import gc
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

from driversi.agents.backends.native import digest, path_at, project_root
from driversi.configuration import read_mapping
from driversi.runtime.budget import write_json
from driversi.training.orchestration.state import deterministic, fingerprint, rng_state, restore_rng
from driversi_agents.chainflow.sampling import GlobalEpochCursor, distributed_settings, remaining_updates


def _cpu(value):
    import torch
    if torch.is_tensor(value):return value.detach().cpu()
    if isinstance(value,dict):return {k:_cpu(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(_cpu(v) for v in value)
    return value


def run(plan):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import default_collate
    from driversi_agents.chainflow import training as native
    from driversi_agents.chainflow.data import NativeScenes, validate_manifest

    cfg=plan['model_config'];settings=distributed_settings(cfg['settings'])
    rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK','0'))
    world=int(os.environ.get('WORLD_SIZE','1'))
    if world!=len(settings['devices']):raise ValueError('Launcher and declared world size differ')
    torch.cuda.set_device(local)
    fraction=cfg['settings'].get('gpu_memory_fraction',.20)
    if not 0<fraction<=.9:raise ValueError('Explicit GPU memory fraction required')
    torch.cuda.set_per_process_memory_fraction(fraction,local)
    dist.init_process_group('nccl',timeout=timedelta(seconds=300))
    root=project_root();out=Path(plan['output']);rank_out=out/f'rank-{rank}';rank_out.mkdir(exist_ok=True)
    sys.path[:0]=[str(root/cfg['repository']),str(root/'src/vendor/driving/nuplan-devkit')]
    os.environ.update(CHAINFLOW_ENABLE_RAY='0',CHAINFLOW_USE_RAY_SCORE='0',NUPLAN_MAPS_ROOT=cfg['paths']['maps'])
    from navsim.common.numpy_pickle_compat import install_numpy_2_pickle_compat
    install_numpy_2_pickle_compat()
    req=plan['resolved']['request'];seed=req['pilot']['seed'];batch_size=req['pilot']['batch_size']
    dataset=validate_manifest(plan['data']['manifest'],groups=plan['source_groups'])
    if dataset['schema_version'] not in (2,3):raise ValueError('DDP requires the complete official navtrain population')
    rows=dataset['splits']['train'];synthetics=[r for r in rows if r['kind']=='synthetic_navtrain']
    validation=native._verify_inputs(synthetics,rank_out/'scoring-budget.json',req['budget']['max_scoring_agent_steps']//world)
    deterministic(seed)
    local_plan={**plan,'output':str(rank_out)}
    agent,native_cfg=native._agent(local_plan,path_at(root,cfg['weights']['checkpoint']))
    agent.num_gpus=world;native_cfg.num_gpus=world
    # Native get_optimizers owns global-batch scaling and schedule equations.
    recipe=native._training_recipe(native_cfg,cfg['settings'],batch_size*world)
    recipe.update(batch_size_per_device=batch_size,world_size=world,
        sample_order='global_shuffled_original_epoch_with_synthetic_insertions',
        sampling_seed=seed,synthetic_every=settings['synthetic_every'],
        epoch_boundary='all_original_samples_then_explicit_distributed_padding',
        recovery_schema_version=3,gradient_reduction='DDP_native_mean',
        batch_normalization='native_rank_local_buffers',training_population='complete_official_navtrain')
    if settings['synthetic_every'] == 1:
        recipe.update(sample_order='shuffled_synthetic_epochs_or_original_only_control',
            epoch_boundary='selected_population_then_explicit_distributed_padding',
            training_population='synthetic_only_when_present_original_only_control')
    agent.scheduler_args=native_cfg.scheduler_args
    from omegaconf import OmegaConf
    OmegaConf.save(native_cfg, rank_out/'native-config.yaml')
    optimizers,schedules=agent.get_optimizers();optimizer=optimizers[0];scheduler=schedules[0]['scheduler']
    if len(optimizers)!=1 or len(schedules)!=1 or schedules[0].get('interval')!='step':
        raise ValueError('Native optimizer protocol changed')
    scenes=NativeScenes(rows,agent,native_original_windows=True)
    spent=native._score_function(agent,rows,rank_out/'scoring-budget.json',
        req['budget']['max_scoring_agent_steps']//world,validation_steps=validation)
    cursor=GlobalEpochCursor(rows,seed,batch_size,world,settings['synthetic_every'])
    start=0;trace=[];previous=read_mapping(plan['previous']) if plan['previous'] else None
    recovered=None
    if previous:
        recovered=torch.load(previous['recovery_checkpoint'],map_location='cpu',weights_only=False)
        if recovered.get('context')!=plan['context'] or recovered.get('recipe')!=recipe or recovered.get('recovery_schema_version')!=3:
            raise ValueError('DDP recovery data/software/recipe/world size differs')
        if len(recovered['ranks'])!=world:raise ValueError('DDP recovery rank coverage differs')
        agent.load_state_dict(recovered['model'],strict=True);optimizer.load_state_dict(recovered['optimizer']);scheduler.load_state_dict(recovered['scheduler'])
        cursor.load_state_dict(recovered['cursor']);start=recovered['total_updates'];trace=recovered['sample_trace']
        if scheduler.last_epoch!=start or cursor.draws!=start*world*batch_size or len(trace)!=start:
            raise ValueError('DDP recovery optimizer/scheduler/cursor disagree')

    class NativeLoss(torch.nn.Module):
        def __init__(self,model):super().__init__();self.agent=model
        def forward(self,features,targets):
            prediction=self.agent.forward(features,targets)
            losses=self.agent.compute_loss(features,targets,prediction)
            return losses['loss'],{k:v.detach() for k,v in losses.items() if torch.is_tensor(v) and v.numel()==1}

    agent.train()
    wrapper=DistributedDataParallel(NativeLoss(agent),device_ids=[local],broadcast_buffers=False,find_unused_parameters=True)
    if recovered:
        with torch.no_grad():
            for name,buffer in agent.named_buffers():buffer.copy_(recovered['ranks'][rank]['buffers'][name])
        restore_rng(recovered['ranks'][rank]['rng'])
    else:deterministic(seed+rank)

    def snapshot(total):
        local_state=dict(rank=rank,rng=rng_state(),buffers=_cpu(dict(agent.named_buffers())))
        ranks=[None]*world;dist.all_gather_object(ranks,local_state)
        if rank:return None
        return dict(recovery_schema_version=3,context=plan['context'],recipe=recipe,
            model=_cpu(agent.state_dict()),optimizer=_cpu(optimizer.state_dict()),scheduler=scheduler.state_dict(),
            ranks=ranks,cursor=cursor.state_dict(),sample_trace=list(trace),total_updates=total)

    def state_hash(state):return fingerprint({k:v for k,v in state.items() if k not in ('context','total_updates')})
    initial=snapshot(start);restored=state_hash(initial) if rank==0 else None
    before=native._parameter_fingerprint(agent) if rank==0 else None
    updates_to_run=remaining_updates(recipe['scheduler']['total_updates'], start, req['budget']['max_optimizer_steps'])
    losses=[];last=None
    if rank==0:write_json(out/'effective-training-recipe.json',recipe)
    for update in range(updates_to_run):
        indices,position=cursor.next_batch();local_indices=indices[rank*batch_size:(rank+1)*batch_size]
        features,targets=native._move(default_collate([scenes[i] for i in local_indices]),f'cuda:{local}')
        learning_rates=[g['lr'] for g in optimizer.param_groups];scheduler_before=scheduler.last_epoch
        optimizer.zero_grad(set_to_none=True)
        loss,details=wrapper(features,targets)
        finite=torch.isfinite(loss).to(torch.int32);dist.all_reduce(finite,op=dist.ReduceOp.MIN)
        if not finite.item():raise ValueError('Nonfinite native loss on a rank')
        loss.backward()
        finite=torch.tensor(int(all(p.grad is None or torch.isfinite(p.grad).all() for p in agent.parameters())),device=local)
        dist.all_reduce(finite,op=dist.ReduceOp.MIN)
        if not finite.item():raise ValueError('Nonfinite native gradient on a rank')
        optimizer.step();scheduler.step()
        finite=torch.tensor(int(all(torch.isfinite(p).all() for p in agent.parameters())),device=local)
        dist.all_reduce(finite,op=dist.ReduceOp.MIN)
        if not finite.item():raise ValueError('Nonfinite updated native parameters on a rank')
        for value in details.values():dist.all_reduce(value);value.div_(world)
        losses.append({k:float(v.cpu()) for k,v in details.items()})
        trace.append(dict(optimizer_step=start+update+1,sample_ids=[rows[i]['sample_id'] for i in indices],
            rank_sample_ids=[[rows[i]['sample_id'] for i in indices[j*batch_size:(j+1)*batch_size]] for j in range(world)],
            learning_rates_before=learning_rates,native_scheduler_step_before=scheduler_before,**position))
        total=start+update+1
        if rank==0:
            write_json(out/'training-progress.json',dict(updates=update+1,total_updates=total,cursor=cursor.state_dict(),losses=losses[-20:]))
            print('DDP update',total,'loss',losses[-1]['loss'],flush=True)
        if total%settings['checkpoint_interval']==0 or update+1==updates_to_run:
            last=snapshot(total)
            if rank==0:
                folder=out/'checkpoints'/f'step-{total:08d}';folder.mkdir(parents=True)
                recovery=folder/'recovery.pt';model=folder/'model.ckpt'
                torch.save(last,folder/'recovery.tmp');os.replace(folder/'recovery.tmp',recovery)
                torch.save({'state_dict':{'agent.'+k:v for k,v in last['model'].items()}},folder/'model.tmp');os.replace(folder/'model.tmp',model)
                write_json(out/'latest-checkpoint.json',dict(recovery_checkpoint=str(recovery),checkpoint=str(model),
                    total_updates=total,updates=update+1,world_size=world,cursor=cursor.state_dict(),context=plan['context'],
                    checkpoint_files={str(p):digest(p) for p in (model,recovery)},
                    state_fingerprint=state_hash(last),effective_training_recipe=recipe,
                    sample_trace=list(trace),recovery_schema_version=3))
            dist.barrier()
        del features,targets,loss,details
    costs=[None]*world;dist.all_gather_object(costs,spent['reserved_agent_steps'])
    result=None
    if rank==0:
        after=native._parameter_fingerprint(agent)
        result=dict(status='completed',reason=None,checkpoint=str(model),recovery_checkpoint=str(recovery),
            checkpoint_files={str(p):digest(p) for p in (model,recovery)},updates=len(losses),total_updates=start+len(losses),
            losses=losses,sample_trace=trace,state_fingerprint=state_hash(last),restored_fingerprint=restored,
            resume_from=previous['recovery_checkpoint'] if previous else None,
            checkpoint_state=['model','optimizer','scheduler','rank_rng','rank_buffers','global_epoch_cursor'],
            native_loss_unchanged=True,native_optimizer_scheduler_algorithm_unchanged=True,
            effective_training_recipe=recipe,recovery_schema_version=3,
            training_population_size=len(rows),actual_training_sample_count=len({s for item in trace for s in item['sample_ids']}),
            data_cursor=cursor.state_dict(),world_size=world,
            parameter_update_evidence=dict(before_sha256=before,after_sha256=after,trainable_parameters_changed=before!=after),
            scope='epoch_capable_backend_bounded_run_not_full_training_reproduction',
            consumed={'environment_steps':0,'scoring_agent_steps':sum(costs)})
    del wrapper,agent,scenes;gc.collect();torch.cuda.empty_cache();dist.destroy_process_group()
    return result
