"""Training semantics are separate from a model's input/output family."""

import argparse
from pathlib import Path

from driversi.agents.backends.native import catalog, project_root
from driversi.configuration import read_mapping
from driversi.runtime.budget import write_json

MODES = ("supervised", "offline_reward", "online_rl")
# Only implemented and audited learners belong here. A paper capability or
# inference checkpoint is not an executable learner.
BACKENDS = {
    ("augustus", "supervised"): {
        "id": "augustus_sft",
        "objective": "trajectory_and_language",
        "data_contract": "native_augustus_history_trajectory_prediction_cot",
        "native_protocol": "augustus_native_sft",
    },

}
LIFECYCLE_BACKENDS = {
    **BACKENDS,
    ("chainflow_stage1", "supervised"): {
        "id": "chainflow_stage1_native_lifecycle",
        "objective": "native_trajectory_and_live_pdm_proposal_scoring",
        "data_contract": "four_history_ten_future_native_scene_with_branch_pdm_cache",
        "origin": "official_model_loss_optimizer_with_bounded_platform_loop",
    },
    ("driverl", "online_rl"): {
        "id": "driversi_driverl_paper_ppo",
        "objective": "native_beta_decomposed_ppo",
        "origin": "paper_derived_not_official_training_release",
    },
}

NATIVE_LEARNERS = {}
for (model, mode), backend in BACKENDS.items():
    NATIVE_LEARNERS.setdefault(model, {})[mode] = backend["native_protocol"]


def inventory(root=None):
    rows = []
    for cfg in catalog(root):
        modes = NATIVE_LEARNERS.get(cfg["model"], {})
        folder = Path(root or project_root()) / "src/config/agents" / cfg["model"]
        training_configs = [
            p
            for p in sorted(folder.glob("*.yaml"))
            if read_mapping(p).get("capabilities", {}).get("train")
        ]
        rows.append(
            {
                "training_configs": list(map(str, training_configs)),
                "model": cfg["model"],
                "family": cfg["family"],
                "training_modes": {m: bool(m in modes and training_configs) for m in MODES},
                "lifecycle_modes": {m: (cfg["model"], m) in LIFECYCLE_BACKENDS for m in MODES},
                "lifecycle_config": str(
                    Path(root or project_root())
                    / "src/config/training/lifecycle"
                    / (cfg["model"] + ".json")
                ),
                "native_protocols": modes,
                "config": str(
                    Path(root or project_root()) / "src/config/agents" / cfg["model"] / "native.yaml"
                ),
                "readiness": "requires_native_preflight",
                "performance_verified": False,
            }
        )
    return {
        "schema_version": 1,
        "models": rows,
        "online_coordinator": "backend_injected",
        "native_online_learner_connected": False,
        "injected_online_learners": {
            "driverl": {
                "implementation": "driversi-driverl-paper-ppo-v1",
                "entrypoint": "driversi_agents.driverl.learning.runner",
                "origin": "paper_derived_not_official_training_release",
                "scope": "bounded_single_world_pilot",
                "max_episode_decisions": 110,
                "native_evaluator_unchanged": True,
            }
        },
        "scope": "adapter capabilities, not weights/environment readiness or performance",
    }


def require_native_mode(model, mode):
    if mode not in MODES:
        raise ValueError("Unknown training mode")
    if mode not in NATIVE_LEARNERS.get(model, {}):
        raise ValueError("Native learner not connected: " + model + "/" + mode)
    return NATIVE_LEARNERS[model][mode]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, inventory())


if __name__ == "__main__":
    main()
