"""Bounded data-size/exposure policy; native optimizer and loss stay unchanged."""
import copy
import math
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.rsi.experiments.continuation import frozen_json


def choose(samples, round_index, devices, max_updates=256):
    if (type(samples) is not int or samples<10 or type(round_index) is not int or round_index<0
            or not devices or any(type(d) is not int or not 0<=d<=7 for d in devices)
            or len(set(devices))!=len(devices)):
        raise ValueError('Admitted data and authorized device pool required')
    # Keep small cohorts on one device; scale only within the configured pool.
    count=min(len(devices),max(1,math.ceil(samples/128)))
    batch=2*count
    wanted=min(8,4+2*round_index)
    epochs=min(wanted,max_updates//math.ceil(samples/batch))
    if epochs<4:raise ValueError('Insufficient explicit SFT budget for four exposures')
    from .problem_learning import sft_updates
    updates=sft_updates(samples,epochs,batch)
    if updates%2:updates+=1
    if updates>max_updates:raise ValueError('SFT update budget exceeded')
    return dict(version='bounded_exposure_recipe_v1',samples=samples,epochs=epochs,
        devices=list(devices[:count]),gpu_count=count,batch_per_device=2,global_batch=batch,
        optimizer_updates_per_arm=updates,min_exposures=epochs,
        max_exposures=math.ceil(updates*batch/samples),training_population='synthetic_only',
        selection_reason='increase_exposure_after_unresolved_round_within_update_budget',
        learning_rate_policy='native_get_optimizers_batch_scaling',learned_optimal=False)


def prepare_recipe(root,case,config,summary):
    import yaml
    folder=Path(case['current_round']);path=folder/'training-config.json'
    model=copy.deepcopy(read_mapping(config['training_model_config']))
    recipe=choose(len(summary['sample_reports']),case['round'],
        model['settings']['distributed_training']['devices'],config['max_pair_optimizer_steps'])
    native=project_root()/'agents/chainflow/native/navsim/planning/script/config/common/agent/chainflow_vla_stage1.yaml'
    lr=yaml.safe_load(native.read_text())['lr_args']
    recipe.update(native_config_sha256=digest(native),
        native_base_lr=float(lr['base_lr']),native_base_batch_size=int(lr['base_batch_size']),
        scaled_base_lr=float(lr['base_lr'])*math.sqrt(recipe['global_batch']/int(lr['base_batch_size'])),
        warmup_fraction=.1,precision='32-true',optimizer_initialization='parent_weights_new_native_optimizer',
        actual_training_started=False,problem=case['id'],round=case['round'])
    model['settings']['distributed_training']['devices']=recipe['devices']
    model['settings']['training_recipe'].update(scheduler_dataset_size=recipe['samples'],scheduler_num_epochs=recipe['epochs'])
    frozen_json(folder/'training-model.json',model)
    updated=dict(config,training_model_config=str(folder/'training-model.json'),sft_epochs=recipe['epochs'],
        training_capability_plan=dict(path=str(folder/'training-cohort-plan.json'),sha256=digest(folder/'training-cohort-plan.json')),
        training_family_summary=dict(path=str(folder/'training-cohort.json'),sha256=digest(folder/'training-cohort.json')))
    frozen_json(folder/'training-recipe.json',recipe)
    frozen_json(path,updated)
    return path
