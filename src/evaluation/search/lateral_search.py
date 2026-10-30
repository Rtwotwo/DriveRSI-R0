"""Budgeted reactive lateral scenario expansion; not teacher correction or PFSP.

Every attempt changes only one declared controller route. Other actors re-decide
from current state. Failed attempts and charged simulation work are retained.
Only complete, bounded maneuvers can be selected; no official driving score is
inferred and all exports remain unqualified for training.
"""
import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from driversi.agents.traffic_control.following import FollowingConfig, RouteFollower
from driversi.agents.traffic_control.routing import resolve_routes
from driversi.contracts.context import SceneContext
from driversi.contracts.driving_task import bind_driving_tasks
from driversi.contracts.road import RoadGraph
from driversi.contracts.scene import Scene
from driversi.evaluation.benchmarks.lateral import blend_lane_change, export_lateral_pair, run_lateral_pair
from driversi.sim.backends.pufferdrive import PufferDriveWorld
from driversi.training.sampling.environment import validate_development_split


class SearchBudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class SearchBudget:
    max_attempts: int = 12
    max_joint_steps: int = 2000
    max_wall_seconds: float = 300.0

    def __post_init__(self):
        import math
        if (type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 128
                or type(self.max_joint_steps) is not int or self.max_joint_steps < 1
                or isinstance(self.max_wall_seconds, bool)
                or not math.isfinite(self.max_wall_seconds) or self.max_wall_seconds <= 0):
            raise ValueError("Invalid bounded search budget")


class StepAccount:
    def __init__(self, budget):
        self.budget, self.steps, self.started = budget, 0, time.monotonic()

    def check(self):
        if time.monotonic() - self.started >= self.budget.max_wall_seconds:
            raise SearchBudgetExceeded("wall_time")

    def consume(self):
        self.check()
        if self.steps >= self.budget.max_joint_steps:
            raise SearchBudgetExceeded("joint_steps")
        # Charge attempted native work, including a step that subsequently fails.
        self.steps += 1


def run(scene_path, graph_path, candidates, output, *, budget=SearchBudget(),
        config=FollowingConfig(), context_path=None):
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("A nonempty candidate list is required")
    required = {"challenger", "responder", "length_m", "intervention_tick"}
    if any(not isinstance(row, dict) or set(row) != required for row in candidates):
        raise ValueError("Candidates require explicit actor IDs, length and intervention tick")
    source, graph = Scene.load(Path(scene_path)), RoadGraph.load(Path(graph_path))
    validate_development_split(source.metadata["split"])
    routes, routing = resolve_routes(source, graph)
    scene = bind_driving_tasks(source, routes)
    context = (SceneContext.load(Path(context_path)).with_driving_tasks(source, scene)
               if context_path else None)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    account = StepAccount(budget)
    result = {"schema_version": 1, "kind": "bounded-lateral-search-v1", "status": "running",
              "purpose": "scenario_expansion_not_same_failure_repair", "trainable": False,
              "budget": asdict(budget), "attempts": [], "selected": None,
              "requested_candidates": len(candidates), "charged_joint_steps": 0,
              "wall_limit_semantics": "checked_between_steps_and_attempts_not_preemptive",
              "selection": "first_complete_bounded_maneuver_in_declared_candidate_order"}

    def publish():
        result["charged_joint_steps"] = account.steps
        result["wall_seconds"] = time.monotonic() - account.started
        pending = output / "search.pending"
        pending.write_text(json.dumps(result, indent=2, allow_nan=False))
        pending.replace(output / "search.json")

    publish()
    for index, specification in enumerate(candidates):
        if index >= budget.max_attempts:
            result["stop_reason"] = "attempt_limit"
            break
        before = account.steps
        row = {"id": f"candidate-{index:03d}", "parameters": specification, "status": "running"}
        result["attempts"].append(row)
        publish()
        try:
            account.check()
            actor, responder = specification["challenger"], specification["responder"]
            tick = specification["intervention_tick"]
            if (actor not in routes or responder not in routes or actor == responder
                    or type(tick) is not int or not 0 <= tick < len(scene.times)-2):
                raise ValueError("Invalid controlled pair or intervention tick")
            ctrl = RouteFollower(routes, config)
            with PufferDriveWorld(scene) as preview:
                for _ in range(tick):
                    if preview.episode_done:
                        raise ValueError("World ended before candidate intervention")
                    commands = ctrl.commands(preview.observe())
                    account.consume()
                    preview.step({c["actor"]: c["action"] for c in commands})
                xy = preview.state()[scene.actor_ids.index(actor), :2]
            candidate = blend_lane_change(routes[actor], routes[responder], xy,
                                          length_m=specification["length_m"])
            report, records, snapshot = run_lateral_pair(
                scene, routes, candidate, challenger=actor, responder=responder,
                intervention_tick=tick, config=config, step_guard=account.consume)
            report = export_lateral_pair(scene, graph, routes, candidate, report, records, snapshot,
                                         output / row["id"], source_scene_path=scene_path,
                                         route_report=routing, scene_context=context, config=config)
            row.update(status="completed", report=row["id"]+"/report.json",
                       maneuver_validation_pass=report["maneuver_validation_pass"],
                       interaction_evidence=report["interaction_evidence"],
                       bounds_failures=report["challenge"]["bounds_failures"],
                       maneuver_completed=report["challenge"]["maneuver_completed"])
            if result["selected"] is None and report["maneuver_validation_pass"]:
                result["selected"] = row["id"]
        except SearchBudgetExceeded as error:
            row.update(status="budget_exhausted", reason=str(error))
            result["stop_reason"] = str(error)
        except Exception as error:
            row.update(status="rejected", reason={"type": type(error).__name__, "message": str(error)})
        row["charged_joint_steps"] = account.steps - before
        publish()
        if row["status"] == "budget_exhausted":
            break
    result["status"] = "partial" if len(result["attempts"]) < len(candidates) or any(
        r["status"] == "budget_exhausted" for r in result["attempts"]) else "completed"
    result.setdefault("stop_reason", "candidate_list_exhausted")
    publish()
    write_search_viewer(output)
    return result


def write_search_viewer(output):
    """Inspection index includes failures; it never implies teacher qualification."""
    from html import escape
    output = Path(output)
    result = json.loads((output / "search.json").read_text())
    rows = []
    for row in result["attempts"]:
        spec = row["parameters"]
        name = escape(row["id"])
        if row.get("report"):
            name = f'<a href="{escape(row["id"])}/viewer.html">{name}</a>'
        details = row.get("reason", row.get("bounds_failures", []))
        if row.get("status") == "completed" and not row.get("maneuver_completed"):
            details = list(details) + ["maneuver_incomplete"]
        cells = [name, escape(str(spec["challenger"])), escape(str(spec["responder"])),
                 escape(str(spec["length_m"])), escape(str(spec["intervention_tick"])),
                 escape(row["status"]), str(bool(row.get("interaction_evidence"))),
                 str(bool(row.get("maneuver_validation_pass"))),
                 str(row.get("charged_joint_steps", 0)), escape(json.dumps(details, ensure_ascii=False))]
        rows.append("<tr>" + "".join("<td>"+c+"</td>" for c in cells) + "</tr>")
    body = """<!doctype html><html lang="zh"><meta charset="utf-8">
<title>DriveRSI 交互候选检查</title><style>
body{background:#101a24;color:#e9eff4;font:15px system-ui;margin:32px}a{color:#83caff}
article{background:#1c2b39;padding:20px;border-radius:12px;margin:20px 0}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:left;padding:10px;border-bottom:1px solid #405260}
pre{white-space:pre-wrap}strong{color:#f4ce7a}</style><h1>多车交互候选检查</h1>
<article><strong>规则控制器的场景扩展；不是学习型对手或合格教师数据。</strong>
<p>配对分支保持共同初态、任务和普通交通控制器；只改变挑战车辆的跟踪路线。
对象仍按共同世界响应。点击候选可检查两条实际执行轨迹，失败记录一并保留。</p>
<p>交互响应与完成机动分开判定。未通过、未知路权和渲染支持不足不能用总分抵消。</p></article>"""
    summary = {k: result[k] for k in ("status", "budget", "charged_joint_steps", "wall_seconds", "selected")}
    body += "<article><h2>本次预算与结果</h2><pre>"+escape(json.dumps(summary, ensure_ascii=False, indent=2))+"</pre>"
    body += '<a href="search.json">原始搜索记录</a></article><table><tr>'
    body += ''.join('<th>'+h+'</th>' for h in ['候选','挑战车','响应车','长度 m','干预 tick','状态','响应证据','完整验收','计费联合步','失败原因'])
    body += '</tr>'+''.join(rows)+'</table></html>'
    pending = output / 'index.pending'
    pending.write_text(body)
    pending.replace(output / 'index.html')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--road-graph", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--scene-context", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=12)
    parser.add_argument("--max-joint-steps", type=int, default=2000)
    parser.add_argument("--max-wall-seconds", type=float, default=300)
    args = parser.parse_args(argv)
    result = run(args.scene, args.road_graph, json.loads(args.candidates.read_text()), args.output,
                 budget=SearchBudget(args.max_attempts, args.max_joint_steps, args.max_wall_seconds),
                 context_path=args.scene_context)
    print(json.dumps(result, allow_nan=False))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
