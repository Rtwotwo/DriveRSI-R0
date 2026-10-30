import copy

from driversi.rendering.gaussian.worldengine_geometry import UnsupportedTerrain, terrain_failure_evidence


def test_search_budget_is_not_a_confirmed_second_road():
    error = UnsupportedTerrain("ambiguous_measured_road_secondary_layer")
    error.evidence = {"secondary_layer_check": {"status": "unresolved", "reason": "residual_plane_budget_exceeded"}}
    before = copy.deepcopy(error.evidence)
    result = terrain_failure_evidence(error)
    assert result["status"] == "unsupported"
    assert result["diagnosis"] == "secondary_surface_not_resolved"
    assert result["confirmed_secondary_fits"] == 0
    assert result["reason"] == str(error)
    result["evidence"].clear()
    assert error.evidence == before


def test_actual_secondary_witness_is_preserved_through_nested_proofs():
    error = UnsupportedTerrain("ambiguous_measured_road_secondary_layer")
    witness = {"height_m": 3.47, "support_count": 24}
    error.evidence = {"layer_separation": {"excluded_points_xyz": [[1., 2., 3.]], "secondary_layer_check": {"status": "ambiguous", "supported_secondary": witness}}}
    result = terrain_failure_evidence(error)
    assert result["diagnosis"] == "confirmed_competing_surface_fit"
    assert result["confirmed_secondary_fits"] == 1
    assert result["evidence"] == error.evidence
    assert result["status"] == "unsupported"


def test_unmeasured_height_is_never_certified_or_called_a_second_surface():
    result = terrain_failure_evidence(UnsupportedTerrain("Incomplete finite height-map interpolation support"))
    assert result["status"] == "unsupported"
    assert result["diagnosis"] == "surface_support_not_established"
    assert result["evidence"] == {}
