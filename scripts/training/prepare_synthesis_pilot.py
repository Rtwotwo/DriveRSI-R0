"""Build a small real-data Harness study, without starting training."""

import argparse
import json
import subprocess
from pathlib import Path

import yaml

from driversi.agents.backends.native import digest, project_root
from driversi.runtime.budget import write_json
from driversi_agents.augustus.data import manifest, read_cache

r = project_root()
parser = argparse.ArgumentParser(description=__doc__)
for option in (
    "output",
    "manifest",
    "backend",
    "reference-contract",
    "official-cache-root",
    "evaluator",
    "scorer-python",
):
    parser.add_argument("--" + option, required=True, type=Path)
parser.add_argument("--task-id", required=True)
parser.add_argument("--samples", type=int, default=4, choices=range(1, 17))
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()
if args.gpu < 0:
    parser.error("GPU must be nonnegative")
o = args.output.resolve()
o.relative_to((r / "exp").resolve())
o.mkdir(parents=True, exist_ok=False)
m = manifest(args.manifest.resolve(), allow_smoke=True)
train = read_cache(m["splits"]["train"]).select(range(args.samples))
train.save_to_disk(str(o / "cache/train/batch_0"))
tokens = set(train["frame_id"])
m["splits"]["train"]["records"] = [
    x for x in m["splits"]["train"]["records"] if x["frame_id"] in tokens
]
m["splits"]["train"]["cache"] = str(o / "cache/train")
m["dataset_version"] += ":bounded-synthesis-" + str(args.samples)
write_json(o / "dataset.json", m)
b = args.backend.resolve()
reference = json.loads(args.reference_contract.read_text())
cases = []
for record in m["splits"]["train"]["records"]:
    row = next(x for x in train if x["frame_id"] == record["frame_id"])
    identity = {
        **record,
        "token": record["frame_id"],
        "split": "train",
        "expert_gt": json.loads(row["future_x_y_angle"]),
        "official_cache_path": str(
            args.official_cache_root.resolve()
            / record["log_name"]
            / "unknown"
            / record["frame_id"]
            / "metric_cache.pkl"
        ),
        "pdm_config": reference["identity"]["pdm_config"],
        "component_names": reference["identity"]["component_names"],
    }
    assert Path(identity["official_cache_path"]).is_file()
    # Trusted native cache is read in its own dependency environment. Preserve
    # both source and scorer headings; the worker transforms coordinates exactly.
    probe = """import json,lzma,pickle,sys
with lzma.open(sys.argv[1],'rb') as f:c=pickle.load(f)
p=c.ego_state.rear_axle
print(json.dumps({'pose':[p.x,p.y,p.heading],'timestamp_us':c.ego_state.time_point.time_us}))"""
    scoring = json.loads(
        subprocess.check_output(
            [str(args.scorer_python.resolve()), "-c", probe, identity["official_cache_path"]],
            cwd=b,
            text=True,
        )
    )
    if scoring["timestamp_us"] != identity["timestamp_us"]:
        raise ValueError("Native cache timestamp differs")
    identity["scoring_ego_pose"] = scoring["pose"]
    path = o / (record["frame_id"] + "-contract.json")
    write_json(path, {"identity": identity, "protocol": reference["protocol"]})
    cases.append({"contract": str(path)})
write_json(o / "cases.json", cases)
write_json(
    o / "protected.json",
    m["protected_logs"]
    + [x["log_name"] for s in ("validation", "development") for x in m["splits"][s]["records"]],
)
e = json.loads(args.evaluator.read_text())
if e["dataset_manifest_sha256"] != digest(args.manifest):
    raise ValueError("Evaluator belongs to another source dataset")
e["dataset_manifest_sha256"] = digest(o / "dataset.json")
write_json(o / "evaluator.json", e)
data = json.loads((r / "src/config/training/lifecycle/augustus-data.json").read_text())
data.update(manifest=str(o / "dataset.json"), metric_cache=e["metric_cache"])
write_json(o / "data.json", data)
cfg = yaml.safe_load((r / "src/config/agents/augustus/lifecycle.yaml").read_text())
cfg["settings"].update(gpu=args.gpu, gpus=[args.gpu])
(o / "model.yaml").write_text(yaml.safe_dump(cfg))
req = json.loads((r / "src/config/training/lifecycle/augustus.json").read_text())
req.update(id=args.task_id, model_config=str(o / "model.yaml"), data=str(o / "data.json"))
req["budget"]["max_optimizer_steps"] = 1
write_json(o / "request.json", req)
steps = []


def add(id, provider, inputs, config, wall=1800):
    steps.append(
        dict(id=id, provider=provider, inputs=inputs, config=config, max_wall_seconds=wall)
    )


add(
    "search",
    "drivetto_search_skill",
    {},
    dict(
        backend=str(b),
        python=str(args.scorer_python.resolve()),
        cases=str(o / "cases.json"),
        protected_logs=str(o / "protected.json"),
        initial_keys=[[150, 0], [-150, 0], [0, 150], [0, -150]],
        refinement_keys=[[50, 0], [-50, 0], [0, 50], [0, -50]],
        max_queries=11 * args.samples,
        max_wall_seconds=600,
    ),
    660,
)
add(
    "admit",
    "drivetto_independent_admission",
    {"search": "search"},
    dict(max_scenes=args.samples, max_wall_seconds=300),
    330,
)
add(
    "history",
    "synthesis_history",
    {"admission": "admit"},
    dict(manifest=str(o / "dataset.json")),
    120,
)
for arm, labels in [("control", "original"), ("treatment", "drivetto")]:
    add(
        arm + "-data",
        "synthesis_native_dataset",
        {"sensors": "history"},
        dict(
            python=str(r / ".venvs/augustus/bin/python"), max_wall_seconds=300, label_source=labels
        ),
        330,
    )
    add(
        arm + "-train",
        "native_dataset_training",
        {"dataset": arm + "-data"},
        dict(request=str(o / "request.json")),
    )
    add(
        arm + "-resume",
        "native_dataset_resume",
        {"student": arm + "-train"},
        dict(max_sim_agent_steps=640),
    )
    if arm == "treatment":
        add(
            "replay",
            "native_dataset_resume",
            {"student": arm + "-train"},
            dict(max_sim_agent_steps=640),
        )
    add(
        arm + "-eval",
        "native_dataset_evaluation",
        {"student": arm + "-resume"},
        dict(evaluator=str(o / "evaluator.json"), max_sim_agent_steps=640),
    )
add(
    "feedback",
    "paired_synthesis_feedback",
    dict(
        control_student="control-resume",
        treatment_student="treatment-resume",
        control_evaluation="control-eval",
        treatment_evaluation="treatment-eval",
        replay="replay",
        search="search",
    ),
    {},
    120,
)
spec = dict(
    schema_version=2,
    id=args.task_id,
    label="New DriveTTO search to matched native learning; current-run holdout only",
    budget=dict(max_wall_seconds=15000, max_sim_agent_steps=12 * args.samples * 80 + 7 * 640),
    steps=steps,
)
write_json(o / "harness.json", spec)
print(o / "harness.json")
