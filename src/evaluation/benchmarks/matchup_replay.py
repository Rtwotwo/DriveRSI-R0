"""Optional exact BEV evidence for a fixed-policy matchup; no GS asset inference."""
from pathlib import Path

from driversi.rendering.bev.replay import export_replay
from driversi.sim.persistence.recording import RolloutRecord, write_rollout


class MatchupReplay:
    def __init__(self, world):
        self.snapshot = world.snapshot()
        self.record = RolloutRecord.start(world)

    def append(self, world, actor_ids, actions, transition):
        self.record.append(world, dict(zip(actor_ids, map(int, actions))), transition)

    def export(self, scene, output, episode):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        scene.save(output / "scene.npz")
        roles = episode["roles"]
        write_rollout(scene, output / "rollout.npz", record=self.record, snapshot=self.snapshot,
                      controlled_ids=tuple(row["actor_id"] for row in roles),
                      policy="frozen-learner-opponent-matchup-v1",
                      controller_configuration={"roles": roles},
                      protocol={"kind": "independent-matchup-replay", "episode": episode,
                                "gs_assets_verified": False,
                                "content_scope": "compiled_scene_entities_not_full_source_log",
                                "ppo_training_rewards_in_rollout": False,
                                "photoreal_training": False})
        export_replay(scene, output / "rollout.npz", output / "viewer.html")
        return {"directory": str(output.resolve()), "viewer": str((output / "viewer.html").resolve()),
                "gs_assets_verified": False, "mode": "exact_state_action_BEV",
                "source_scene": episode["scene"], "steps": episode["steps"]}
