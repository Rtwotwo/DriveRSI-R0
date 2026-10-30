"""Integration contracts: no substitute model, reward fallback or future-history swap."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from driversi.data.datasets.tto_audit import COMPONENTS, ROW_DTYPE, sha256
from driversi.rsi.plugins import augustus as aug
from driversi.rsi.plugins.tto import collect, prepare_candidates
from driversi_agents.augustus.adapter import build
from driversi_agents.augustus.budget import charge, metered
from driversi_agents.augustus.data import manifest

ROOT = Path(__file__).resolve().parents[3]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def bank(tmp_path):
    token = "0123456789abcdef"
    p = tmp_path / "bank/scenes/train" / token
    p.mkdir(parents=True)
    rows = np.zeros(3, dtype=ROW_DTYPE)
    rows["trajectories"][:, :, 0] = np.arange(8) + 1
    rows["trajectories"][1, :, 1] = 0.1
    rows["trajectories"][2, :, 1] = 0.2
    rows["labels"][:] = 1
    rows["labels"][2] = 0
    np.save(p / "candidates.npy", rows)
    dump(
        p / "ALL_SCORES.json",
        dict(
            format="all_scored_candidates_v1",
            passed=True,
            token=token,
            split="train",
            candidates=3,
            data_digest=sha256(p / "candidates.npy"),
        ),
    )
    dump(p / "state.json", {"count": 3})
    identity = dict(
        token=token,
        split="train",
        log_name="train-log",
        component_names=COMPONENTS,
        expert_gt=rows["trajectories"][0].tolist(),
        timestamp_us=100,
        ego_pose=[1.0, 2.0, 0.0],
    )
    contract = dict(
        identity=identity,
        protocol={
            "constructor": {
                "num_poses": 8,
                "interval_s": 0.5,
                "coordinate_frame": "current_ego_x_forward_y_left",
            },
            "labeling": {
                "protocol": "navsim_v1_official_single_candidate",
                "evaluation_group": "reference_plus_one_candidate",
                "score_unit": "raw_0_to_1",
            },
        },
    )
    dump(p / "contract.json", contract)
    cases = dump(tmp_path / "cases.json", [{"split": "train", "token": token}])
    protect = dump(tmp_path / "protected.json", [])
    config, _ = prepare_candidates(
        dict(bank=str(p.parents[2]), cases=str(cases), protected_logs=str(protect), max_scenes=3),
        tmp_path,
        tmp_path,
    )
    return config, p, identity


def test_tto_role_and_evidence(bank):
    report = collect(bank[0])
    assert report["accepted"][0]["candidate_row"] == 1
    assert report["role"] == "trajectory_proposals_only" and report["new_score_queries"] == 0
    assert not report["trainable"] and not report["reactive_trajectory_admitted"]
    assert collect(bank[0], ["train-log"])["rejected"][0]["reason"] == "protected_source"


def test_no_high_labels_are_not_filled_or_repeated(bank):
    config, p, _ = bank
    rows = np.load(p / "candidates.npy")
    rows["labels"][:] = 0
    np.save(p / "candidates.npy", rows)
    pub = json.loads((p / "ALL_SCORES.json").read_text())
    pub["data_digest"] = sha256(p / "candidates.npy")
    dump(p / "ALL_SCORES.json", pub)
    report = collect(config)
    assert not report["accepted"] and report["rejected"][0]["table_rows"] == 3


def test_mutated_or_wrong_score_context_is_rejected(bank):
    config, p, _ = bank
    ct = json.loads((p / "contract.json").read_text())
    ct["protocol"]["labeling"]["protocol"] = "counterfactual"
    dump(p / "contract.json", ct)
    with pytest.raises(ValueError, match="score context"):
        collect(config)
    Path(config["cases"]).write_text("[]")
    with pytest.raises(ValueError, match="changed"):
        collect(config)


def data_manifest(tmp_path, identity):
    value = {
        "schema_version": 1,
        "kind": "augustus_original_cache",
        "dataset_version": "fixture",
        "parent_training_logs": [],
        "protected_logs": ["official-test"],
        "splits": {},
    }
    for split, log, token in [
        ("train", "train-log", identity["token"]),
        ("validation", "val-log", "val"),
        ("development", "dev-log", "dev"),
    ]:
        value["splits"][split] = {
            "cache": str(tmp_path / split),
            "records": [
                dict(frame_id=token, log_name=log, timestamp_us=100, ego_pose=[1.0, 2.0, 0.0])
            ],
        }
    return dump(tmp_path / "manifest.json", value)


@pytest.mark.parametrize("change", ["train_dev", "parent_dev", "protected"])
def test_log_isolation(tmp_path, bank, change):
    p = data_manifest(tmp_path, bank[2])
    v = json.loads(p.read_text())
    if change == "train_dev":
        v["splits"]["development"]["records"][0]["log_name"] = "train-log"
    if change == "parent_dev":
        v["parent_training_logs"] = ["dev-log"]
    if change == "protected":
        v["protected_logs"] = ["train-log"]
    dump(p, v)
    with pytest.raises(ValueError, match="overlap|Protected"):
        manifest(p)


def metrics(score=0.9, nc=1):
    return {"samples": [dict(frame_id="dev", PDMS=score, NC=nc, DAC=1, TTC=1, EP=1, C=1, DDC=1)]}


def test_safety_regression_vetoes_mean_improvement():
    result = aug.compare(metrics(0.8), metrics(0.95, 0.5), [{"frame_id": "dev"}])
    assert result["mean_pdms_delta"] > 0 and not result["observed_benefit"]
    assert not result["promoted"]


@pytest.mark.parametrize("bad", ["missing", "duplicate", "nan", "percentage"])
def test_independent_evaluation_requires_full_finite_raw_scores(bad):
    candidate = metrics()
    if bad == "missing":
        candidate["samples"][0].pop("NC")
    if bad == "duplicate":
        candidate["samples"] *= 2
    if bad == "nan":
        candidate["samples"][0]["PDMS"] = float("nan")
    if bad == "percentage":
        candidate["samples"][0]["PDMS"] = 95
    with pytest.raises(ValueError):
        aug.compare(metrics(), candidate, [{"frame_id": "dev"}])


def test_meter_reserves_before_call_and_never_skips_failed_scores(tmp_path):
    path = tmp_path / "budget.json"
    calls = []

    def score(preds, ids, **kw):
        calls.append(kw)
        return {"count": len(ids), "PDMS": 0.9}

    run = metered(score, path, 160)
    assert run([0], ["one"]) == {"PDMS": 0.9}
    assert run([0], ["two"], return_count=True)["count"] == 1
    with pytest.raises(RuntimeError, match="exhausted"):
        run([0], ["three"])
    assert len(calls) == 2 and json.loads(path.read_text())["reserved_agent_steps"] == 160
    failure = metered(lambda *a, **k: {}, tmp_path / "failed.json", 80)
    with pytest.raises(RuntimeError, match="Incomplete"):
        failure([0], ["one"])
    assert json.loads((tmp_path / "failed.json").read_text())["reserved_agent_steps"] == 80


def test_shared_budget_is_not_per_rank(tmp_path):
    from concurrent.futures import ProcessPoolExecutor

    path = tmp_path / "budget.json"
    with ProcessPoolExecutor(4) as pool:
        list(pool.map(_charge, [(str(path), i) for i in range(8)]))
    assert json.loads(path.read_text())["reserved_agent_steps"] == 640


def _charge(arg):
    return charge(arg[0], 640, 1)


@pytest.mark.parametrize("inference_only", [False, True])
def test_native_recipe_keeps_training_math_and_denoising(tmp_path, inference_only):
    cfg = yaml.safe_load((ROOT / "src/config/agents/augustus/native.yaml").read_text())
    parent = tmp_path / "parent"
    parent.mkdir()
    recipe = yaml.safe_load(
        (ROOT / "agents/augustus/native/AUGUstus_SFT/configs/navsim_qwen3p5.yaml").read_text()
    )
    recipe["diffusion"]["num_inference_steps"] = 3
    published = json.loads(json.dumps(recipe))
    if inference_only:
        published.pop("training")
        published.pop("resume")
    (parent / "config.yaml").write_text(yaml.safe_dump(published))
    paths = {k: str(tmp_path / k) for k in cfg["paths"]}
    weights = {"parent": str(parent), "backbone": str(tmp_path / "backbone")}
    launch = build(
        cfg,
        ROOT,
        ROOT / cfg["repository"],
        "/missing/python",
        weights,
        paths,
        ROOT / "outputs/fixture",
        "train",
    )
    native = yaml.safe_load(launch["generated_files"]["native.yaml"])
    for key in ("diffusion", "trajectory", "cot"):
        assert native[key] == recipe[key]
    for key in ("learning_rate", "num_epochs", "per_device_train_batch_size", "mixed_precision"):
        assert native["training"][key] == recipe["training"][key]
    assert native["normalization"]["stats_file"] == str(parent / "action_norm_stats.json")
    assert "--resume_from" not in launch["argv"] and "ray" not in " ".join(launch["argv"])
    assert str(parent / "config.yaml") in launch["files"]


def test_harness_exposes_typed_native_chain():
    from driversi.rsi.workflow.providers import builtin_providers

    providers = builtin_providers()
    assert providers.get("drivetto_candidates").output == "trajectory_proposals"
    assert dict(providers.get("native_model_dataset").inputs)["proposals"] == "trajectory_proposals"
    assert providers.get("native_model_rsi_training").output == "native_rsi_student"
    assert (
        dict(providers.get("native_model_rsi_evaluation").inputs)["student"] == "native_rsi_student"
    )
    assert providers.get("native_model_learning_feedback").output == "native_learning_feedback"


def test_missing_native_cache_is_explicit_failure_not_training(tmp_path, bank):
    p = data_manifest(tmp_path, bank[2])
    cfg = {
        "manifest": str(p),
        "sha256": sha256(p),
        "python": "/missing/python",
        "max_wall_seconds": 1,
    }
    result = aug.run_dataset(cfg, {}, tmp_path / "out", tmp_path / "registry")
    assert result["status"] == "failed" and result["reason"] == "native_cache_or_runtime_missing"
    assert result["sim_agent_steps"] == 0


@pytest.mark.parametrize("label_source", ["original", "drivetto"])
def test_optional_real_hf_relabel_preserves_video_and_native_prediction(tmp_path, bank, label_source):
    datasets = pytest.importorskip("datasets")
    av = pytest.importorskip("av")
    import io

    from driversi_agents.augustus.data import relabel

    config, _, identity = bank
    source = data_manifest(tmp_path, identity)
    b = io.BytesIO()
    with av.open(b, mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=2)
        stream.width = 2048
        stream.height = 512
        stream.pix_fmt = "yuv420p"
        for i in range(4):
            frame = av.VideoFrame.from_ndarray(
                np.full((512, 2048, 3), i * 30, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    video = b.getvalue()
    base = dict(
        id="fixture",
        video=video,
        scene_token="original",
        driving_command="GO FORWARD",
        ego_dynamic_state="vx: 1.0 m/s",
        historical_x_y_angle=json.dumps([[0, 0, 0]] * 4),
        historical_vx_vy=json.dumps([[1, 0]] * 4),
        future_x_y_angle=json.dumps(identity["expert_gt"]),
        reasoning_trace="old ego action",
        solution_planning="old plan",
        solution_prediction="logged other-object future",
        solution_perception="unchanged",
    )
    value = manifest(source)
    for split, block in value["splits"].items():
        row = {**base, "frame_id": block["records"][0]["frame_id"]}
        datasets.Dataset.from_list([row]).save_to_disk(str(Path(block["cache"]) / "batch_0"))
    proposals = dump(tmp_path / "proposals.json", collect(config))
    report = relabel(source, proposals, tmp_path / "relabel", label_source)
    out = datasets.load_from_disk(str(tmp_path / "relabel/train/batch_0"))[0]
    assert out["video"] == video and out["historical_x_y_angle"] == base["historical_x_y_angle"]
    assert out["solution_prediction"] == base["solution_prediction"]
    changed = label_source == "drivetto"
    assert out["solution_planning"] == ("" if changed else base["solution_planning"])
    assert np.array_equal(json.loads(out["future_x_y_angle"]), identity["expert_gt"]) != changed
    assert report["replaced_rows"] == int(changed) and not report["on_policy_rl"]
    assert report["label_source"] == label_source
    from driversi_agents.augustus.data import cache_files
    assert report["source_cache_files"] == {p: sha256(p) for p in cache_files(value)}

    # Load our actual output with the unmodified upstream dataset class.
    import importlib.util
    from types import SimpleNamespace

    import torch

    spec = importlib.util.spec_from_file_location(
        "native_augustus_dataset", ROOT / "agents/augustus/native/AUGUstus_SFT/model/dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Processor:
        chat_template = ""
        tokenizer = SimpleNamespace(pad_token_id=0, encode=lambda text, **kwargs: [7, 8])

        def apply_chat_template(self, *args, **kwargs):
            return "history only"

        def __call__(self, **kwargs):
            assert kwargs["video_metadata"][0]["fps"] == 2
            assert len(kwargs["videos"][0]) == 4
            return {"input_ids": torch.tensor([[1, 2, 3]])}

    ds = module.TransfusionHFCacheDataset(
        str(tmp_path / "relabel/train"),
        Processor(),
        str(ROOT / "agents/augustus/native/AUGUstus_SFT/prompts/default.txt"),
        enable_cot=True,
        cot_components=["prediction"],
    )
    item = ds[0]
    assert item["actions"].shape == (8, 3) and item["cot_ids"].tolist() == [7, 8]
    assert np.array_equal(
        item["actions"].numpy(), np.asarray(json.loads(out["future_x_y_angle"]), dtype=np.float32)
    )


def test_normalization_cannot_recompute_or_modify_parent(tmp_path):
    from driversi_agents.augustus.normalization import frozen_loader, read_stats

    path = tmp_path / "stats.json"
    value = {
        "q01": [0] * 3,
        "q99": [1] * 3,
        "q01_per_t": [[0] * 3] * 8,
        "q99_per_t": [[1] * 3] * 8,
        "action_repr": "absolute",
    }
    dump(path, value)
    before = path.read_bytes()
    load = frozen_loader(path)
    assert load(str(path), "/not-used", 3, "absolute") == value and path.read_bytes() == before
    wrong = dict(value)
    wrong.pop("q01_per_t")
    dump(path, wrong)
    with pytest.raises(ValueError, match="incomplete"):
        read_stats(path)
    with pytest.raises(ValueError, match="changed"):
        load(str(path))
    wrong = dict(value, action_repr="delta")
    dump(path, wrong)
    with pytest.raises(ValueError, match="representation"):
        read_stats(path)


@pytest.mark.parametrize("bad", ["external", "duplicate_metadata", "wrong_log", "duplicate_token"])
def test_native_metric_metadata_matches_actual_read_paths(tmp_path, bad):
    from driversi_agents.augustus.metric_cache import scoring_files

    root = tmp_path / "cache"
    meta = root / "metadata"
    meta.mkdir(parents=True)
    target = root / "log/unknown/token/metric_cache.pkl"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"test")
    file = meta / "index.csv"
    file.write_text("file_path\n" + str(target) + "\n")
    records = [{"frame_id": "token", "log_name": "log"}]
    assert scoring_files(root, records) == [str(file), str(target)]
    if bad == "external":
        file.write_text("file_path\n" + str(tmp_path / "outside/token/metric_cache.pkl") + "\n")
    if bad == "duplicate_metadata":
        (meta / "other.csv").write_text(file.read_text())
    if bad == "wrong_log":
        records[0]["log_name"] = "other"
    if bad == "duplicate_token":
        file.write_text(file.read_text() + str(target) + "\n")
    with pytest.raises(ValueError):
        scoring_files(root, records)


def test_optional_strict_parent_load_preserves_weights_and_rejects_partial(tmp_path):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    from driversi_agents.augustus.runner import strict_load

    source = torch.nn.Linear(3, 2)
    st.save_file(source.state_dict(), str(tmp_path / "model.safetensors"))
    target = torch.nn.Linear(3, 2)
    strict_load(target, tmp_path)
    assert all(torch.equal(v, target.state_dict()[k]) for k, v in source.state_dict().items())
    st.save_file({"weight": source.weight}, str(tmp_path / "model.safetensors"))
    with pytest.raises(RuntimeError, match="Missing key"):
        strict_load(target, tmp_path)


def test_dataset_content_change_between_stages_is_rejected(tmp_path, bank):
    from driversi_agents.augustus.data import cache_files

    source = data_manifest(tmp_path, bank[2])
    value = manifest(source)
    for block in value["splits"].values():
        file = Path(block["cache"]) / "batch_0/data.arrow"
        file.parent.mkdir(parents=True)
        file.write_bytes(b"sealed-cache")
    report = dump(
        tmp_path / "report.json", {"cache_files": {p: sha256(p) for p in cache_files(value)}}
    )
    inputs = {"artifacts": {"report": str(report), "manifest": str(source)}}
    Path(cache_files(value)[0]).write_bytes(b"changed labels")
    with pytest.raises(ValueError, match="content changed"):
        aug.resolved({}, inputs, tmp_path / "out", "train")


def test_native_stage_contract_chain_retains_regression_feedback(tmp_path, bank, monkeypatch):
    # Process outputs are fixtures here; this tests artifact flow, not AUGUstus learning.
    from driversi_agents.augustus.data import cache_files

    monkeypatch.setattr(aug, "project_root", lambda: tmp_path)
    test_filter = (
        tmp_path
        / "src/vendor/driving/navsim-v1.1/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
    )
    test_filter.parent.mkdir(parents=True)
    test_filter.write_text("log_names: [official-test]")
    source = data_manifest(tmp_path, bank[2])
    value = manifest(source)
    for block in value["splits"].values():
        file = Path(block["cache"]) / "batch_0/data.arrow"
        file.parent.mkdir(parents=True)
        file.write_bytes(b"cache fixture")
    data_report = dump(
        tmp_path / "dataset-report.json",
        {"cache_files": {p: sha256(p) for p in cache_files(value)}},
    )
    dataset = {
        "artifacts": {"manifest": str(source), "report": str(data_report)},
        "summary": {
            "artifact_reference": {"identity": "dataset-id"},
            "proposal_identity": "proposal-id",
            "replaced_rows": 1,
        },
    }
    parent = tmp_path / "weights/parent"
    parent.mkdir(parents=True)
    (parent / "weights.fixture").write_text("parent")
    model = dump(
        tmp_path / "model.yaml",
        {
            "model": "augustus",
            "weights": {"parent": str(parent)},
            "paths": {},
            "settings": {"gpu": 0, "gpus": [0]},
        },
    )
    config = {
        "model_config": str(model),
        "config_sha256": sha256(model),
        "max_wall_seconds": 10,
        "max_sim_agent_steps": 160,
    }
    candidate = tmp_path / "weights/candidate"

    def prepare_fixture(path, operation, output):
        cfg = yaml.safe_load(Path(path).read_text())
        return {
            "ready_to_attempt": True,
            "operation": operation,
            "output": str(output),
            "plan_sha256": "fixture",
            "source": {"revision": "fixture"},
            "files": {},
            "parent": cfg["weights"]["parent"],
        }

    def execute_fixture(plan, **kwargs):
        output = Path(plan["output"])
        output.mkdir(parents=True)
        if plan["operation"] == "train":
            candidate.mkdir()
            (candidate / "weights.fixture").write_text("candidate")
            dump(
                output / "candidate.json",
                {"parent": str(parent), "checkpoint": str(candidate), "updates": 1},
            )
        else:
            report = metrics(0.8) if plan["parent"] == str(parent) else metrics(0.95, 0.5)
            dump(output / "eval_detailed.json", report)
            dump(output / "scoring-budget.json", {"reserved_agent_steps": 80})
        return {"status": "completed", "wall_seconds": 0.01, "reserved_gpu_seconds": 0.01}

    monkeypatch.setattr(aug, "prepare", prepare_fixture)
    monkeypatch.setattr(aug, "execute", execute_fixture)
    student = aug.run_training(
        config, {"dataset": dataset}, tmp_path / "train-job", tmp_path / "registry"
    )
    student["summary"]["artifact_reference"] = {"identity": "student-id"}
    evaluation = aug.run_evaluation(
        config,
        {"dataset": dataset, "student": student},
        tmp_path / "eval-job",
        tmp_path / "registry",
    )
    feedback = aug.run_feedback(
        {},
        {"dataset": dataset, "student": student, "evaluation": evaluation},
        tmp_path / "feedback",
        tmp_path / "registry",
    )
    assert feedback["status"] == "completed"
    assert feedback["summary"]["outcome"] == "no_benefit_or_regression"
    assert feedback["summary"]["proposal_identity"] == "proposal-id"
    assert (
        evaluation["sim_agent_steps"] == 160
        and not feedback["summary"]["next_round"]["eligible_for_review"]
    )


def test_preallocated_download_is_not_a_complete_checkpoint(tmp_path):
    from driversi_agents.augustus.adapter import complete_assets

    (tmp_path / "model.safetensors").write_bytes(b"partial")
    (tmp_path / "model.safetensors.aria2").write_bytes(b"download map")
    files = complete_assets(tmp_path)
    assert files == [str(tmp_path / "model.safetensors.aria2.PENDING_DOWNLOAD")]
    assert not Path(files[0]).exists()


def test_dataset_launch_preserves_virtual_environment(tmp_path, bank):
    import subprocess
    import venv

    source = data_manifest(tmp_path, bank[2])
    environment = tmp_path / "isolated"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    interpreter = environment / "bin/python"
    assert interpreter.is_symlink()
    config, _ = aug.prepare_dataset(
        {"manifest": str(source), "python": str(interpreter), "max_wall_seconds": 10},
        tmp_path,
        tmp_path / "registry",
    )
    prefix = subprocess.check_output(
        [config["python"], "-c", "import sys; print(sys.prefix)"], text=True
    ).strip()
    assert Path(prefix) == environment


def test_original_drive_shards_cannot_cross_native_splits(tmp_path, bank):
    path = data_manifest(tmp_path, bank[2])
    value = json.loads(path.read_text())
    value["splits"]["train"]["records"][0]["log_name"] = "drive_00001_00100"
    value["splits"]["development"]["records"][0]["log_name"] = "drive_00200_00300"
    dump(path, value)
    with pytest.raises(ValueError, match="overlap"):
        manifest(path)


def test_protected_original_drive_applies_to_all_shards(tmp_path, bank):
    path = data_manifest(tmp_path, bank[2])
    value = json.loads(path.read_text())
    value["splits"]["train"]["records"][0]["log_name"] = "drive_00001_00100"
    value["protected_logs"] = ["drive_00200_00300"]
    dump(path, value)
    with pytest.raises(ValueError, match="Protected"):
        manifest(path)


@pytest.mark.parametrize("operation", ["train", "evaluate"])
@pytest.mark.parametrize("saved_candidate", [False, True])
def test_full_parent_uses_own_metadata_without_a_second_backbone(tmp_path, operation, saved_candidate):
    cfg = yaml.safe_load((ROOT / "src/config/agents/augustus/native.yaml").read_text())
    parent = tmp_path / "parent"
    parent.mkdir()
    recipe = yaml.safe_load(
        (ROOT / "agents/augustus/native/AUGUstus_SFT/configs/navsim_qwen3p5.yaml").read_text()
    )
    recipe["diffusion"]["num_inference_steps"] = 3
    if saved_candidate:
        recipe["model"]["skip_backbone_weight_load"] = True
    (parent / "config.yaml").write_text(yaml.safe_dump(recipe))
    launch = build(
        cfg, ROOT, ROOT / cfg["repository"], "/missing/python",
        {"parent": str(parent)}, {k: str(tmp_path / k) for k in cfg["paths"]},
        ROOT / "outputs/fixture", operation,
    )
    native = yaml.safe_load(launch["generated_files"]["native.yaml"])
    assert native["model"]["qwen_model_name"] == str(parent)
    assert native["model"]["skip_backbone_weight_load"] is True
    for key in ("diffusion", "trajectory", "cot"):
        assert native[key] == recipe[key]
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "processor_config.json"):
        assert str(parent / name) in launch["files"]  # Missing metadata blocks launch.
    assert not any("backbones/" in p for p in launch["files"])


@pytest.mark.parametrize("bundled", [False, True])
@pytest.mark.parametrize("failure", [None, "constructor", "strict"])
def test_eval_load_preserves_native_arguments_and_restores_constructor(monkeypatch, bundled, failure):
    from types import SimpleNamespace

    from driversi_agents.augustus import runner

    calls = []
    model = object()

    def constructor(**kwargs):
        calls.append(kwargs)
        if failure == "constructor":
            raise RuntimeError("constructor failure")
        return model

    native = SimpleNamespace(TransfusionQwen3p5=constructor)

    def original(cfg, checkpoint, device):
        assert device == "cpu" and checkpoint == "parent"
        return native.TransfusionQwen3p5(num_inference_steps=3, freeze_llm=True)

    def strict(value, checkpoint):
        assert value is model and checkpoint == "parent"
        if failure == "strict":
            raise RuntimeError("strict failure")
        calls.append("strict")
        return value

    monkeypatch.setattr(runner, "strict_load", strict)
    native.load_model = original
    cfg = {"model": {"skip_backbone_weight_load": bundled}}
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            runner.strict_native_evaluation(native, cfg, "parent", "cpu")
    else:
        assert runner.strict_native_evaluation(native, cfg, "parent", "cpu") is model
        assert calls[-1] == "strict"
    assert native.TransfusionQwen3p5 is constructor
    expected = {"num_inference_steps": 3, "freeze_llm": True}
    if bundled:
        expected["skip_backbone_weight_load"] = True
    assert calls[0] == expected


@pytest.mark.parametrize("corruption", [None, "truncated", "trailing", "gap", "bad_header"])
def test_bundle_inventory_rejects_partial_weights_without_download_markers(tmp_path, corruption):
    import struct

    from driversi_agents.augustus.adapter import complete_assets, safetensors_extent_complete

    model = tmp_path / "model.safetensors"
    header = {"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    if corruption == "gap":
        header["weight"]["data_offsets"] = [4, 8]
    raw = json.dumps(header).encode()
    content = struct.pack("<Q", len(raw)) + raw + b"12345678"
    if corruption == "truncated":
        content = content[:-1]
    elif corruption == "trailing":
        content += b"0"
    elif corruption == "bad_header":
        content = b"invalid"
    model.write_bytes(content)
    assert safetensors_extent_complete(model) == (corruption is None)
    paths = complete_assets(tmp_path)
    expected = str(model) + (".INCOMPLETE_SAFETENSORS" if corruption else "")
    assert expected in paths
