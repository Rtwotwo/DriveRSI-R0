import copy

import numpy as np
import pytest

from driversi.rendering.gaussian.mtgs_road_surface import (
    MeasuredRoadPolicy,
    MeasuredRoadSurface,
    TraversalConsensusRoadSurface,
    validate_traversal_consensus_evidence,
)
from driversi.rendering.gaussian.worldengine_geometry import UnsupportedTerrain


def cloud(height=1.0, tilt=0.0):
    x, y = np.meshgrid(np.linspace(-2, 2, 41), np.linspace(-2, 2, 41))
    return np.column_stack((x.ravel(), y.ravel(), (height + .02*x - .03*y + tilt*x).ravel()))


def surface(height, tilt=0.0):
    return MeasuredRoadSurface(cloud(height, tilt), {"test_traversal": str(height)})


def bad_surface():
    x, y = np.meshgrid(np.linspace(-2, 2, 13), np.linspace(-2, 2, 13))
    points = np.column_stack((x.ravel(), y.ravel(), (1 + .02*x - .03*y).ravel()))
    indices = np.flatnonzero(np.linalg.norm(points[:, :2], axis=1) < 1.7)[::7]
    points[indices, 2] -= .4
    return MeasuredRoadSurface(points, {"test_traversal": "contaminated_source"})


def test_bad_source_uses_two_independent_agreeing_measurements_without_relaxation():
    road = TraversalConsensusRoadSurface(bad_surface(), [surface(1.005), surface(1.01)])
    sample = road.sample([0, 0])
    assert sample["height_m"] == pytest.approx(1.0075)
    rows = sample["independent_surface_validation"]["traversals"]
    assert rows[0]["reason"] == "ambiguous_measured_road_layers"
    assert rows[0]["rejection"]["excluded_fraction"] > .05
    assert rows[0]["rejection"]["excluded_points_xyz"]
    assert all(row["fit"]["policy"] == MeasuredRoadPolicy().__dict__ for row in rows[1:])
    validate_traversal_consensus_evidence(sample)
    translation, evidence = road.lift([0, 0, 1.4], [.2, -.1, 0])
    assert translation[2] == pytest.approx(.007)
    assert evidence["source_clearance_m"] == pytest.approx(.3925)


def test_one_independent_support_is_not_enough_even_when_source_valid():
    road = TraversalConsensusRoadSurface(surface(1.0), [surface(1.005), bad_surface()])
    with pytest.raises(UnsupportedTerrain, match="two supported independent") as error:
        road.sample([0, 0])
    assert len(error.value.evidence["independent_surface_validation"]["traversals"]) == 3


def test_coherent_source_secondary_layer_cannot_be_overridden_by_references():
    x, y = np.meshgrid(np.linspace(-1.5, 1.5, 6), np.linspace(-1.5, 1.5, 6))
    secondary = np.column_stack((x.ravel(), y.ravel(), (.6 + .02*x - .03*y).ravel()))
    x, y = np.meshgrid(np.linspace(-2, 2, 13), np.linspace(-2, 2, 13))
    primary = np.column_stack((x.ravel(), y.ravel(), (1 + .02*x - .03*y).ravel()))
    source = MeasuredRoadSurface(np.concatenate((primary, secondary)), {"test": "two_road_layers"})
    road = TraversalConsensusRoadSurface(source, [surface(1.005), surface(1.01)])
    with pytest.raises(UnsupportedTerrain, match="secondary layer remains ambiguous"):
        road.sample([0, 0])


def test_large_or_balanced_ambiguous_source_cannot_skip_bounded_guard():
    source = MeasuredRoadSurface(np.concatenate((cloud(), cloud(.6))), {"test": "two_dense_layers"})
    road = TraversalConsensusRoadSurface(source, [surface(1.005), surface(1.01)])
    with pytest.raises(UnsupportedTerrain):
        road.sample([0, 0])


def test_rejected_source_primary_height_still_constrains_consensus():
    road = TraversalConsensusRoadSurface(bad_surface(), [surface(2.), surface(2.005)])
    with pytest.raises(UnsupportedTerrain, match="planes disagree"):
        road.sample([0, 0])


def test_ambiguous_reference_secondary_layer_cannot_be_ignored():
    x, y = np.meshgrid(np.linspace(-1.5, 1.5, 6), np.linspace(-1.5, 1.5, 6))
    secondary = np.column_stack((x.ravel(), y.ravel(), (.6 + .02*x - .03*y).ravel()))
    x, y = np.meshgrid(np.linspace(-2, 2, 13), np.linspace(-2, 2, 13))
    primary = np.column_stack((x.ravel(), y.ravel(), (1 + .02*x - .03*y).ravel()))
    ambiguous = MeasuredRoadSurface(np.concatenate((primary, secondary)), {"test": "two_layers_reference"})
    road = TraversalConsensusRoadSurface(surface(1.), [surface(1.005), surface(1.01), ambiguous])
    with pytest.raises(UnsupportedTerrain):
        road.sample([0, 0])


def test_exact_hull_proof_can_exclude_unreachable_secondary_support():
    from driversi.rendering.gaussian.mtgs_road_surface import _residual_layer_check

    points = cloud(.5)
    points[:, 0] += 4
    result = _residual_layer_check(tuple(map(tuple, points)), (0., 0.))
    assert result["status"] == "no_supported_secondary"
    assert result["proof"] == "query_outside_all_residual_points_convex_hull"


def test_valid_disagreeing_measurement_cannot_be_outvoted_or_source_ignored():
    road = TraversalConsensusRoadSurface(surface(1.0), [surface(1.005), surface(1.01), surface(1.2)])
    with pytest.raises(UnsupportedTerrain, match="planes disagree"):
        road.sample([0, 0])
    road = TraversalConsensusRoadSurface(surface(1.2), [surface(1.005), surface(1.01)])
    with pytest.raises(UnsupportedTerrain, match="planes disagree"):
        road.sample([0, 0])


def test_slope_disagreement_even_with_equal_center_is_rejected():
    road = TraversalConsensusRoadSurface(surface(1.0), [surface(1.005), surface(1.01, .06)])
    with pytest.raises(UnsupportedTerrain, match="planes disagree"):
        road.sample([0, 0])


def test_duplicate_clouds_and_changed_policy_rejected():
    fit = surface(1.)
    with pytest.raises(ValueError, match="Distinct"):
        TraversalConsensusRoadSurface(fit, [surface(1.005), fit])
    with pytest.raises(ValueError, match="two independent"):
        TraversalConsensusRoadSurface(fit, [surface(1.005)])
    changed = MeasuredRoadSurface(cloud(1.01), {"test": "changed"}, MeasuredRoadPolicy(max_rmse_m=.04))
    with pytest.raises(ValueError, match="frozen"):
        TraversalConsensusRoadSurface(fit, [surface(1.005), changed])


def test_admission_rechecks_identity_limits_and_consensus_value():
    road = TraversalConsensusRoadSurface(surface(1.), [surface(1.005), surface(1.01)])
    original = road.sample([0, 0])
    for mutate in (
        lambda s: s.update(height_m=2.),
        lambda s: s["independent_surface_validation"]["traversals"][1].update(
            cloud_array_sha256=s["independent_surface_validation"]["traversals"][0]["cloud_array_sha256"]),
        lambda s: s["independent_surface_validation"]["traversals"][1]["fit"].update(rmse_m=.031),
        lambda s: s["independent_surface_validation"]["traversals"][1]["fit"]["provenance"]["geometric_layer"].update(excluded_fraction=.051),
        lambda s: s["independent_surface_validation"]["traversals"][1]["fit"]["spatial_holdouts"][0].update(max_residual_m=.081),
    ):
        tampered = copy.deepcopy(original)
        mutate(tampered)
        with pytest.raises(ValueError):
            validate_traversal_consensus_evidence(tampered)
    original["height_m"] = -100
    assert road.sample([0, 0])["height_m"] == pytest.approx(1.005)


def test_invalid_queries_and_extrapolation_rejected():
    road = TraversalConsensusRoadSurface(surface(1.), [surface(1.005), surface(1.01)])
    for query in ([float("nan"), 0], [0, 0, 0]):
        with pytest.raises(ValueError):
            road.sample(query)
    with pytest.raises(UnsupportedTerrain):
        road.sample([5., 0])


def test_original_eight_centimetre_layer_cannot_be_missed_by_six_centimetre_proposals():
    from driversi.rendering.gaussian.mtgs_road_surface import _residual_layer_check

    xy = np.array([(a, b) for a in [-.8, -.3, .3, .8]
                   for b in [-.8, -.3, .3, .8]], float)
    xy[1] = xy[0]
    secondary = np.column_stack((xy, .6 + .02 * xy[:, 0] - .03 * xy[:, 1]))
    secondary[0, 2] += .07
    secondary[1, 2] -= .07
    original = MeasuredRoadSurface(secondary, {"test": "eight_cm_surface"}).sample([0, 0])
    assert original["support_count"] == 16
    result = _residual_layer_check(tuple(map(tuple, secondary)), (0., 0.))
    assert result["status"] == "ambiguous"
    x, y = np.meshgrid(np.linspace(-2, 2, 13), np.linspace(-2, 2, 13))
    primary = np.column_stack((x.ravel(), y.ravel(), (1 + .02*x - .03*y).ravel()))
    source = MeasuredRoadSurface(np.concatenate((primary, secondary)), {"test": "layered_source"})
    with pytest.raises(UnsupportedTerrain, match="secondary layer remains ambiguous"):
        TraversalConsensusRoadSurface(source, [surface(1.005), surface(1.01)]).sample([0, 0])


def test_bounded_hypothesis_failure_is_unresolved_not_clear():
    from driversi.rendering.gaussian.mtgs_road_surface import _residual_layer_check

    xy = np.array([(a, b) for a in [-.8, -.3, .3, .8]
                   for b in [-.8, -.3, .3, .8]], float)
    points = np.column_stack((xy, np.arange(len(xy), dtype=float)))
    result = _residual_layer_check(tuple(map(tuple, points)), (0., 0.))
    assert result["status"] == "unresolved"


def tiny_secondary_surface():
    xy = np.array([(a, b) for a in [-.8, -.3, .3, .8]
                   for b in [-.8, -.3, .3, .8]], float)
    secondary = np.column_stack((xy, .6 + .02 * xy[:, 0] - .03 * xy[:, 1]))
    return MeasuredRoadSurface(np.concatenate((cloud(), secondary)), {"test": "small_secondary"})


def test_below_five_percent_secondary_plane_rejected_by_pooled_and_consensus():
    from driversi.rendering.gaussian.mtgs_road_surface import IndependentlyValidatedRoadSurface

    ambiguous = tiny_secondary_surface()
    with pytest.raises(UnsupportedTerrain, match="ambiguous_measured_road_secondary_layer") as error:
        IndependentlyValidatedRoadSurface(ambiguous, surface(1.005)).sample([0, 0])
    assert error.value.evidence["excluded_fraction"] < .05
    assert len(error.value.evidence["excluded_points_xyz"]) == 16
    assert error.value.evidence["secondary_layer_check"]["status"] == "ambiguous"
    with pytest.raises(UnsupportedTerrain, match="secondary layer remains ambiguous"):
        TraversalConsensusRoadSurface(ambiguous, [surface(1.005), surface(1.01)]).sample([0, 0])
    with pytest.raises(UnsupportedTerrain, match="secondary layer remains ambiguous"):
        TraversalConsensusRoadSurface(surface(1.005), [surface(1.01), surface(1.015), ambiguous]).sample([0, 0])


def test_supported_layer_residual_proof_is_recomputed_not_trusted():
    original = TraversalConsensusRoadSurface(surface(1.), [surface(1.005), surface(1.01)]).sample([0, 0])
    altered = copy.deepcopy(original)
    row = altered["independent_surface_validation"]["traversals"][0]
    layer = row["fit"]["provenance"]["geometric_layer"]
    secondary = tiny_secondary_surface().points[-16:]
    layer["excluded_points_xyz"] = secondary.tolist()
    layer["excluded_point_ids"] = list(range(5000, 5016))
    layer["total_support_count"] += 16
    layer["excluded_fraction"] = 16 / layer["total_support_count"]
    # The previously valid empty-residual proof is deliberately retained.
    with pytest.raises(ValueError, match="secondary layer remains ambiguous"):
        validate_traversal_consensus_evidence(altered)
    old = copy.deepcopy(original)
    del old["independent_surface_validation"]["terrain_proof_version"]
    with pytest.raises(ValueError, match="Unsupported traversal"):
        validate_traversal_consensus_evidence(old)


def test_pooled_validator_rejects_old_or_forged_supported_layer_proofs():
    from driversi.rendering.gaussian.mtgs_road_surface import (
        IndependentlyValidatedRoadSurface,
        validate_independent_surface_evidence,
    )

    original = IndependentlyValidatedRoadSurface(surface(1.), surface(1.005)).sample([0, 0])
    validate_independent_surface_evidence(original)
    tampered = copy.deepcopy(original)
    del tampered["independent_surface_validation"]["terrain_proof_version"]
    with pytest.raises(ValueError, match="Current independent terrain"):
        validate_independent_surface_evidence(tampered)
    tampered = copy.deepcopy(original)
    layer = tampered["independent_surface_validation"]["fit"]["provenance"]["geometric_layer"]
    layer["excluded_point_ids"] = [1]
    layer["excluded_points_xyz"] = [[0., 0., .6]]
    layer["total_support_count"] += 1
    layer["excluded_fraction"] = 1 / layer["total_support_count"]
    with pytest.raises(ValueError, match="Observed secondary-layer check differs"):
        validate_independent_surface_evidence(tampered)


def test_less_than_minimum_secondary_support_keeps_original_geometric_limits():
    from driversi.rendering.gaussian.mtgs_road_surface import IndependentlyValidatedRoadSurface

    xy = np.array([(a, b) for a in [-.8, -.3, .3, .8]
                   for b in [-.8, -.3, .3, .8]], float)[:15]
    residual = np.column_stack((xy, .6 + .02 * xy[:, 0] - .03 * xy[:, 1]))
    source = MeasuredRoadSurface(np.concatenate((cloud(), residual)), {"test": "few_returns"})
    sample = IndependentlyValidatedRoadSurface(source, surface(1.005)).sample([0, 0])
    proof = sample["provenance"]["geometric_layer"]["secondary_layer_check"]
    assert proof["residual_point_count"] == 15
    assert proof["status"] == "no_supported_secondary"
    assert sample["policy"] == MeasuredRoadPolicy().__dict__
