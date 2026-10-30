"""Frozen native AUGUstus prefix intervention. No scorer, teacher or optimizer."""

import json
import sys
import time
import types
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.runtime.budget import write_json
from driversi.rsi.decisions.frozen_skills import ARMS, compile_prompts


MODEL_INPUTS = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw",
                "pixel_values_videos", "video_grid_thw")


def skill_metadata(member, arm, evolution):
    if not evolution:
        return member['skill']
    return None if arm == 'native' else member['arm_skills'][arm]


def evolution_prompts(base, skills):
    return {'native': base, **{
        arm: (base.rstrip() + '\n\nPlanning checklist for this observation:\n' + skills[arm]['text']
              if skills[arm].get('enabled',True) else base)
        for arm in skills}}


def run(prepared_path, output):
    import numpy as np
    import torch
    import yaml
    from driversi.training.orchestration.state import deterministic
    from driversi_agents.augustus.adapter import merge_recipe
    from driversi_agents.augustus.runner import strict_native_evaluation
    from driversi_agents.augustus.normalization import frozen_loader

    prepared = json.loads(Path(prepared_path).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "calls").mkdir(exist_ok=True)
    if (output / "report.json").exists():
        old = json.loads((output / "report.json").read_text())
        if old["prepared_sha256"] != digest(prepared_path):
            raise ValueError("Frozen inference input changed")
        for name, sha in old["files"].items():
            if digest(name) != sha: raise ValueError("Frozen inference artifact changed")
        return old
    root = project_root()
    sft = root / "agents/augustus/native/AUGUstus_SFT"
    sys.path[:0] = [str(sft), str(sft / "third_party/navsim"), str(sft / "third_party/nuplan-devkit")]
    import model.utils as utils
    import test_qwen3p5 as native_eval
    import train_qwen3p5 as native

    plan = prepared["plan"]
    arms = tuple(plan.get('arms', ARMS))
    repair=plan.get('skill_repair_protocol')=='targeted_ablation_v1'
    if repair and arms!=('native','incumbent','text_only','gate_only','combined'):
        raise ValueError('Targeted repair requires declared factorial arms')
    evolution = repair or arms == ('native', 'incumbent', 'candidate')
    if not evolution and arms != ARMS:
        raise ValueError('Unknown frozen comparison mode')
    parent = Path(plan["checkpoint"])
    if digest(parent / "model.safetensors") != plan["checkpoint_sha256"]:
        raise ValueError("Frozen parent changed")
    for name, sha in prepared["cache_files"].items():
        if digest(name) != sha: raise ValueError("Prepared observations changed")
    cfg = merge_recipe(yaml.safe_load((sft / "configs/navsim_qwen3p5.yaml").read_text()),
                       yaml.safe_load((parent / "config.yaml").read_text()))
    if cfg["cot"].get("use_action_first") is not True:
        raise ValueError("This experiment requires the declared action-first checkpoint")
    cfg["model"].update(qwen_model_name=str(parent), skip_backbone_weight_load=True)
    cfg["normalization"]["stats_file"] = str(parent / "action_norm_stats.json")
    cfg["data"].update(train_data_path=prepared["cache"], val_data_path=prepared["cache"],
                       prompt_file=str(sft / "prompts/default.txt"), metric_cache_test=None,
                       metric_cache_train=None)
    cfg["training"].update(per_device_eval_batch_size=1, dataloader_num_workers=0)
    utils.load_or_compute_action_norm_stats = frozen_loader(parent / "action_norm_stats.json")
    memory_fraction=plan.get('gpu_memory_fraction',0.60)
    if type(memory_fraction) not in (int,float) or not 0.05<=memory_fraction<=0.60:
        raise ValueError('Frozen inference memory fraction outside declared bounds')
    torch.cuda.set_per_process_memory_fraction(memory_fraction)
    deterministic(plan["seeds"][0])
    started = time.monotonic()
    write_json(output / "progress.json", dict(stage="loading_frozen_parent", completed=0))
    model = strict_native_evaluation(native_eval, cfg, parent, device="cuda")
    model.eval().requires_grad_(False)
    model.processor.video_processor.do_sample_frames = False
    versions = {n: p._version for n, p in model.named_parameters()}
    _, raw_loader = native.build_loaders(cfg, model.processor)
    dataset, collator = raw_loader.dataset, raw_loader.collate_fn
    base_template = dataset.prompt_template
    load_seconds = time.monotonic() - started
    skill_map = {r["token"]: r["skill"] for r in prepared["members"]}
    member_map = {r['token']: r for r in prepared['members']}
    prompt_map = {}
    for row in dataset.hf_dataset:
        base = dataset._fill_prompt_fields(base_template, row)
        if evolution:
            skills = member_map[row['frame_id']]['arm_skills']
            prompt_map[row['frame_id']] = evolution_prompts(base, skills)
        else:
            prompt_map[row["frame_id"]] = compile_prompts(base, skill_map[row["frame_id"]], model.processor.tokenizer)
    current_arm = "native"

    def build_prompt(self, row):
        return prompt_map[row["frame_id"]][current_arm]

    dataset._build_prompt = types.MethodType(build_prompt, dataset)
    rows = []
    frozen_versions = lambda: all(p._version == versions[n] and not p.requires_grad
                                 and p.grad is None for n, p in model.named_parameters())

    def predict(batch, seed, explain=False):
        deterministic(seed)
        network = {k: v.to("cuda") for k, v in batch.items()
                   if k in MODEL_INPUTS and torch.is_tensor(v)}
        # Labels, future actions and ground-truth CoT never reach the model.
        torch.cuda.synchronize()
        t0 = time.monotonic()
        with torch.no_grad():
            pred, cot, _, _ = model.generate_trajectory(**network, generate_cot=explain, return_normed=True)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - t0
        raw = pred.detach().float().cpu().numpy()
        rounded = (torch.round(pred * 100.0) / 100.0).detach().float().cpu().numpy()
        if raw.shape != (1, 8, 3) or not np.isfinite(raw).all():
            raise ValueError("Native trajectory contract failed")
        return raw[0], rounded[0], elapsed, int(cot.numel()) if cot is not None else 0

    # The first real scene witnesses post-hoc text omission and deterministic replay.
    first = collator([dataset[0]])
    a, _, ta, _ = predict(first, plan["seeds"][0], explain=False)
    b, _, tb, _ = predict(first, plan["seeds"][0], explain=False)
    c, _, tc, cot_tokens = predict(first, plan["seeds"][0], explain=True)
    witness = dict(repeated_raw_actions_equal=bool(np.array_equal(a, b)),
                   explanation_raw_actions_equal=bool(np.array_equal(a, c)),
                   maximum_repeat_delta=float(np.abs(a-b).max()),
                   maximum_explanation_delta=float(np.abs(a-c).max()),
                   posthoc_tokens=cot_tokens, wall_seconds=ta+tb+tc,
                   parameters_frozen=frozen_versions(), optimizer_updates=0)
    write_json(output / "witness.json", witness)
    if not all(witness[k] for k in ("repeated_raw_actions_equal", "explanation_raw_actions_equal", "parameters_frozen")):
        raise RuntimeError("Native frozen inference witness failed")
    for i, member in enumerate(prepared["members"]):
        if dataset.hf_dataset[i]["frame_id"] != member["token"]:
            raise ValueError("Native row order changed")
        processed = {}
        for current_arm in arms:
            t0 = time.monotonic()
            processed[current_arm] = (collator([dataset[i]]), time.monotonic()-t0)
        tokens = {arm: int(v[0]["attention_mask"].sum().item()) for arm, v in processed.items()}
        if not evolution and tokens["matched_skill"] != tokens["neutral_control"]:
            raise ValueError("Complete native prompt length control mismatch")
        for seed in plan["seeds"]:
            baseline = None
            for arm in arms:
                raw, pred, elapsed, _ = predict(processed[arm][0], seed)
                if arm == "native": baseline, baseline_raw = pred.copy(), raw.copy()
                delta = pred - baseline
                row = dict(token=member["token"], source_log=member["source_log"], seed=seed, arm=arm,
                           skill=skill_metadata(member, arm, evolution),
                           selection_stratum=member["selection_stratum"],
                           prompt=prompt_map[member["token"]][arm], input_token_count=tokens[arm],
                           inference_seconds=elapsed, preprocessing_seconds=processed[arm][1],
                           delta_from_native=dict(xy_ade_m=float(np.linalg.norm(delta[:,:2], axis=1).mean()),
                                                  xy_fde_m=float(np.linalg.norm(delta[-1,:2]))),
                           trajectory=pred.tolist(), raw_trajectory=raw.tolist())
                if evolution and arm!='native' and not member['arm_skills'][arm].get('enabled',True):
                    if not np.array_equal(pred,baseline) or not np.array_equal(raw,baseline_raw):
                        raise RuntimeError('Disabled skill failed exact native fallback')
                    row['native_fallback_verified']=True
                rows.append(row)
                write_json(output / "progress.json", dict(stage="frozen_inference", completed=len(rows),
                           expected=len(prepared["members"])*len(plan["seeds"])*len(arms)))
                # Durable per-call receipts permit auditing partial inference without score-based selection.
                write_json(output / "calls" / f"{len(rows)-1:04d}.json", row)
    if not frozen_versions() or digest(parent / "model.safetensors") != plan["checkpoint_sha256"]:
        raise ValueError("Weights changed during frozen inference")
    write_json(output / "predictions.json", rows)
    result = dict(status="completed", rows=len(rows), prepared_sha256=digest(prepared_path),
                  checkpoint_sha256=plan["checkpoint_sha256"], parameters_frozen=True, optimizer_updates=0,
                  model_load_seconds=load_seconds, wall_seconds=time.monotonic()-started,
                  max_gpu_memory_bytes=torch.cuda.max_memory_allocated(), witness=witness,
                  model_input_keys=list(MODEL_INPUTS), sealed_before_scoring=True,
                  files={str(output / n): digest(output / n) for n in ("predictions.json", "witness.json")})
    write_json(output / "report.json", result)
    return result
