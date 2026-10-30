"""Bounded local VLM jobs. No future supervision, student scores or remote calls."""

import os
import sys
import time
from pathlib import Path

from .observations import file_hash, history_evidence
from .prompts import annotation_prompt, parse_annotation_response
from .store import encode, identity


def annotator_profile(config):
    if config["backend"] not in ("augustus_sft_backbone", "qwen3_5"):
        raise ValueError("Unsupported local annotator backend")
    if not 128 <= config["max_new_tokens"] <= 2048:
        raise ValueError("Output-token cap outside pilot bounds")
    if not 1 <= config["max_seconds_per_scene"] <= 300:
        raise ValueError("Scene generation time cap outside pilot bounds")
    model = Path(config["model_path"]).resolve()
    files = [p for p in model.iterdir() if p.suffix in (".json", ".yaml", ".jinja", ".safetensors")]
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("Local safetensors weights required")
    profile = {
        "config": config, "prompt": annotation_prompt(config), "schema": 1,
        "checkpoint_files": {p.name: file_hash(p) for p in sorted(files)},
        "decoding": "greedy_no_thinking", "input_scope": "four_history_frames_only",
        "claim": "unverified_observation_not_diagnosis_or_supervision",
    }
    if config["backend"] == "augustus_sft_backbone":
        root = Path(config["native_sft"]).resolve()
        files = [root / "train_qwen3p5.py", *sorted((root / "model").rglob("*.py"))]
        profile["native_source"] = {str(p.relative_to(root)): file_hash(p) for p in files}
        profile["conditioning"] = "native_time_zero_text_prefix"
    import torch
    import transformers

    profile["runtime"] = {"torch": torch.__version__, "transformers": transformers.__version__}
    profile["adapter"] = {
        name: file_hash(Path(__file__).with_name(name))
        for name in ("labeling.py", "observations.py", "prompts.py")
    }
    if config.get("prompt_version") in ("camera_grounded_v6", "camera_grounded_v7"):
        profile["input_scope"] = "three_current_images_independent_camera_calls"
        profile["adapter"]["grounding.py"] = file_hash(Path(__file__).with_name("grounding.py"))
    if config.get("prompt_version") == "history_state_v8":
        if config["backend"] != "qwen3_5":
            raise ValueError("History profile requires the local Qwen backbone")
        profile["input_scope"] = "four_real_images_per_camera_plus_measured_historical_ego"
        profile["camera_batch_size"] = len(config["cameras"])
        profile["adapter"]["history_caption.py"] = file_hash(Path(__file__).with_name("history_caption.py"))
        import PIL, importlib.metadata
        profile["adapter"]["history_recovery.py"] = file_hash(Path(__file__).with_name("history_recovery.py"))
        profile["runtime"].update(pillow=PIL.__version__, pyquaternion=importlib.metadata.version("pyquaternion"))
        profile["factual_quality_release"] = False
    return profile


class LocalAnnotator:
    def __init__(self, config):
        import torch

        self.config, self.owner, self.cond = config, None, None
        if not torch.cuda.is_available():
            raise RuntimeError("The pilot requires an explicitly allocated CUDA device")
        model_path = Path(config["model_path"]).resolve()
        if config["backend"] == "augustus_sft_backbone":
            import yaml

            root = Path(config["native_sft"]).resolve()
            sys.path[:0] = [str(root), str(root / "third_party/navsim"),
                           str(root / "third_party/nuplan-devkit")]
            import train_qwen3p5 as native

            from driversi_agents.augustus.adapter import merge_recipe
            from driversi_agents.augustus.runner import strict_load

            cfg = merge_recipe(yaml.safe_load((root / "configs/navsim_qwen3p5.yaml").read_text()),
                               yaml.safe_load((model_path / "config.yaml").read_text()))
            cfg["model"].update(qwen_model_name=str(model_path), skip_backbone_weight_load=True)
            cfg["normalization"]["stats_file"] = str(model_path / "action_norm_stats.json")
            if not Path(cfg["normalization"]["stats_file"]).is_file():
                raise ValueError("Native normalization must exist; no recalculation")
            self.owner = strict_load(native.build_model(cfg), model_path).eval().to("cuda")
            self.model, self.processor = self.owner.qwen_model, self.owner.processor
            with torch.inference_mode():
                self.cond = self.owner._compute_adarms_cond(
                    torch.zeros(1, device="cuda", dtype=torch.float32), torch.bfloat16
                )
            # HF generate() does not expose the native AdaRMS text argument.
            # Forward it at the native text module boundary, identically for
            # prefill and decode, without changing checkpoint or model source.
            def native_text_condition(module, args, kwargs):
                kwargs["adarms_cond"] = self.cond
                return args, kwargs

            self.condition_hook = self.owner._text_model.register_forward_pre_hook(
                native_text_condition, with_kwargs=True
            )
        else:
            from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

            self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_path, local_files_only=True, dtype=torch.bfloat16,
                attn_implementation="sdpa",
            ).eval().to("cuda")

    def __call__(self, evidence, images):
        import torch

        content = [{"type": "text", "text": annotation_prompt(self.config)}]
        if self.config.get("prompt_version") in ("camera_grounded_v6", "camera_grounded_v7"):
            camera = evidence[0]["camera"]
            extra = ("Front camera. Describe static placement; no driving/walking/approaching claims. "
                     if camera == "CAM_F0" else
                     "Side camera: use vehicle_to_side_visible, never vehicle_ahead_visible or ego-turn direction tags. ")
            content.append({"type": "text", "text": camera + ". " + extra + "No sign transcription. End after the JSON object."})
        for row, image in zip(evidence, images, strict=True):
            content += [
                {"type": "text", "text": f'Image {row["id"]}, t={row["relative_time_s"]:.3f}s'},
                {"type": "image", "image": image},
            ]
        if self.config.get("prompt_version") in ("retrieval_v3", "retrieval_v4", "retrieval_v5"):
            content.append({"type": "text", "text": (
                'Return JSON with all three keys: unknown, caption, observations. '
                'At most four distinct tags; cite the exact h3 camera image where the fact is visible. '
                'Describe visible road geometry and actor placement, not guessed motion.'
            )})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", enable_thinking=False,
        ).to("cuda")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        options = {}
        if self.config.get("prompt_version") in ("camera_grounded_v6", "camera_grounded_v7"):
            import json

            from transformers import StoppingCriteria, StoppingCriteriaList
            processor, prefix = self.processor, inputs["input_ids"].shape[-1]
            class CompleteObject(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    text = processor.decode(input_ids[0, prefix:], skip_special_tokens=True).strip()
                    if text.startswith("```json\n"):
                        text = text[8:]
                    try:
                        value, _ = json.JSONDecoder().raw_decode(text)
                        return isinstance(value, dict)
                    except json.JSONDecodeError:
                        return False
            options["stopping_criteria"] = StoppingCriteriaList([CompleteObject()])
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=self.config["max_new_tokens"],
                max_time=self.config["max_seconds_per_scene"], **options,
            )
        torch.cuda.synchronize()
        generated = output[0, inputs["input_ids"].shape[-1]:]
        return self.processor.decode(generated, skip_special_tokens=True), {
            "input_tokens": int(inputs["input_ids"].shape[-1]),
            "output_tokens": len(generated), "images": len(images),
            "generation_seconds": time.monotonic() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "generation_budget_exhausted": len(generated) >= self.config["max_new_tokens"],
        }


def label_batch(catalog, config, tokens, output, *, max_scenes=16, max_seconds=1200):
    """Single-writer annotation with durable per-scene success/failure receipts."""
    import fcntl

    if not 1 <= max_scenes <= 1000 or len(tokens) > max_scenes or len(tokens) != len(set(tokens)):
        raise ValueError("Invalid or unbounded annotation token batch")
    if not 1 <= max_seconds <= 3600:
        raise ValueError("Explicit bounded job time required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with Path(str(catalog.path) + ".label.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        profile_data = annotator_profile(config)
        profile = catalog.profile(profile_data)
        report = {"schema_version": 1, "split": catalog.split, "profile": profile,
                  "requested": len(tokens), "rows": [], "trainable": False,
                  "scope": "unverified_visual_observations", "complete": False}
        (output / "profile.json").write_text(encode(profile_data))
        backend = None
        for token in tokens:
            if time.monotonic() - started >= max_seconds:
                break
            evidence, raw, observation, error, cost = [], None, None, None, {}
            step = time.monotonic()
            reusing_history_attempt = False
            try:
                scene = catalog.scene(token)
                context = None
                if config.get("prompt_version") == "history_state_v8":
                    from .history_caption import load_history
                    evidence, images, context = load_history(scene, config)
                else:
                    evidence, images = history_evidence(scene, config["cameras"], config["max_side"])
                if context is not None:
                    old = catalog.db.execute("SELECT id,input_id,status,raw FROM attempts WHERE token=? AND profile=? ORDER BY id DESC LIMIT 1",(token,profile)).fetchone()
                    if old is not None and old["input_id"] == identity(evidence) and old["status"] == "validated":
                        reusing_history_attempt = True
                        review = catalog.db.execute("SELECT reviewer,verdict,note,created FROM reviews WHERE attempt=? ORDER BY id DESC LIMIT 1",(old["id"],)).fetchone()
                        export_history_sidecar(output,old["id"],token,catalog.split,profile,old["raw"],evidence,config,review=dict(review) if review else None)
                        report["rows"].append({"token":token,"attempt":old["id"],"status":"cached_rejected" if review and review["verdict"]=="rejected" else "reused"})
                        continue
                elif catalog.completed(token, profile, identity(evidence)):
                    report["rows"].append({"token": token, "status": "reused"})
                    continue
                if backend is None:
                    if config.get("prompt_version") == "history_state_v8":
                        from .history_caption import HistoryAnnotator
                        backend = HistoryAnnotator(config)
                    elif config.get("prompt_version") in ("camera_grounded_v6", "camera_grounded_v7"):
                        from .grounding import CameraAnnotator
                        backend = CameraAnnotator(config)
                    else:
                        backend = LocalAnnotator(config)
                raw, cost = (backend(evidence, images, context) if context is not None else backend(evidence, images))
                observation = parse_annotation_response(raw, evidence, config)
            except Exception as exc:
                if reusing_history_attempt:
                    # Export/storage errors must not replace a validated inference receipt.
                    raise
                # Local-only backend: errors contain no remote credentials or request headers.
                error = type(exc).__name__ + ": " + str(exc)[:2000]
            cost["total_seconds"] = time.monotonic() - step
            attempt = catalog.record_attempt(token, profile, evidence, raw, cost,
                                             observation=observation, error=error)
            if observation is not None and config.get("prompt_version") == "history_state_v8":
                export_history_sidecar(output,attempt,token,catalog.split,profile,raw,evidence,config)
            report["rows"].append({"token": token, "attempt": attempt,
                                   "status": "validated" if observation else "failed", "error": error})
            report["elapsed_seconds"] = time.monotonic() - started
            temporary = output / "report.tmp"
            temporary.write_text(encode(report))
            os.replace(temporary, output / "report.json")
        report["complete"] = len(report["rows"]) == len(tokens)
        report["elapsed_seconds"] = time.monotonic() - started
        report["stats"] = catalog.stats()
        (output / "report.json").write_text(encode(report))
        return report


def export_history_sidecar(output, attempt, token, split, profile, raw, evidence, config, *, review=None):
    """Reconstructible export; committed DB raw/evidence remains authoritative."""
    from .history_caption import parse_bundle
    from .corpus import write_report
    _, structured = parse_bundle(raw,evidence,config)
    structured["assessment"] = review["verdict"] if review else "unreviewed"
    structured["review"] = review
    write_report(Path(output)/f"attempt-{attempt}.json",{
        "token":token,"split":split,"profile":profile,"attempt":attempt,
        "structured":structured,"image_evidence":evidence})
