"""Declarative model roles; compatibility never implies a supplied or qualified adapter."""
import copy
import math

from driversi.contracts.learning import ACTION_SPACE, OBSERVATION_VERSION

_ROLES = {"action_policy": "action_proposal", "trajectory_policy": "trajectory_proposal",
          "dynamics_predictor": "prediction_only", "observation_generator": "observation_only"}
_FIELDS = {"schema_version", "role", "input_schema", "output_schema", "input_clock", "output_clock",
           "output_dt_seconds", "horizon_seconds", "state_scope", "likelihood", "authority"}
STRUCTURED_POLICY = dict(schema_version=1, role="action_policy", input_schema=OBSERVATION_VERSION,
    output_schema=ACTION_SPACE, input_clock="current_observation", output_clock="next_control_tick",
    output_dt_seconds=.1, horizon_seconds=.1, state_scope="stateless",
    likelihood="categorical_action", authority="action_proposal")
VISUAL_POLICY = {**STRUCTURED_POLICY, "input_schema": "actor-owned-heterogeneous-rgb-v1",
                "input_clock": "history_to_observation", "state_scope": "explicit_actor",
                "likelihood": "none"}
RUNTIMES = ("structured_ppo", "visual_sampling")


def validate_capability(value):
    if not isinstance(value, dict) or set(value) != _FIELDS:
        raise ValueError("Model capability fields must match schema version 1")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("Unknown model capability schema")
    if (not isinstance(value["role"], str) or value["role"] not in _ROLES
            or value["authority"] != _ROLES[value["role"]]):
        raise ValueError("Model role cannot acquire action or authoritative-world privileges implicitly")
    for name in ("input_schema", "output_schema"):
        if not isinstance(value[name], str) or not value[name].strip():
            raise ValueError("Nonempty model input/output schemas required")
    if (value["input_clock"] not in ("current_observation", "history_to_observation", "history_and_planned_future")
            or value["output_clock"] not in ("next_control_tick", "future_sequence")
            or value["state_scope"] not in ("stateless", "explicit_actor", "explicit_world")
            or value["likelihood"] not in ("none", "categorical_action", "trajectory_density")):
        raise ValueError("Unsupported model clock, state or likelihood semantics")
    for name in ("output_dt_seconds", "horizon_seconds"):
        x = value[name]
        if type(x) not in (int, float) or not math.isfinite(x) or x <= 0:
            raise ValueError("Model output clock must use finite positive seconds")
    if value["horizon_seconds"] < value["output_dt_seconds"]:
        raise ValueError("Model horizon is shorter than one output interval")
    if value["output_clock"] == "next_control_tick" and value["horizon_seconds"] != value["output_dt_seconds"]:
        raise ValueError("A next-tick action spans exactly one control interval")
    if value["role"] in ("dynamics_predictor", "observation_generator") and value["likelihood"] != "none":
        raise ValueError("Prediction likelihood is not a policy action likelihood")
    return copy.deepcopy(value)


def compatibility(value, runtime):
    value = validate_capability(value)
    if runtime not in RUNTIMES:
        raise ValueError("Unknown supported runtime")
    required = STRUCTURED_POLICY if runtime == "structured_ppo" else VISUAL_POLICY
    differences = [name for name in required if value[name] != required[name]]
    return {"runtime": runtime, "compatible": not differences, "mismatched_fields": differences,
            "adapter_provided_by_declaration": False, "driving_quality_qualified": False}


def require_capability(value, runtime):
    report = compatibility(value, runtime)
    if not report["compatible"]:
        raise ValueError(f"Model capability incompatible with {runtime}: {', '.join(report['mismatched_fields'])}")
    return report


def inspect_capability(value):
    return {"capability": validate_capability(value),
            "runtimes": [compatibility(value, name) for name in RUNTIMES],
            "model_loaded": False, "training_or_rendering_qualified": False}
