from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from inspect import getsource
from math import inf, nan, nextafter
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import path_planner.v2.hopper_authority as authority
import path_planner.v2.hopper_api as hopper_api
import path_planner.v2.hopper_route_validation as route_validation
import path_planner.v2.oracles.hopper as hopper_oracle
import path_planner.v2.providers.hopper as hopper_provider
import path_planner.v2.profiles as profiles
from path_planner.v2.contracts import (
    AcceleratorPolicyV2,
    ObjectiveProfileV2,
    PlanningFailureV2,
    PlanningRequestV2,
    PlanningSuccessV2,
    PoseStateV2,
    ResourceBudgetV2,
)
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.serialization import canonical_json_bytes
from path_planner.v2.terrain import (
    FineGridGeometryV2,
    FineSafetyAnchorV2,
    TerrainProvenanceV2,
    TerrainSnapshotV2,
)


PARAMETER_SET_ID = (
    "hopper_generic_internal_computational_simulation_proxy_midterm_g2g3/v1"
)
BASE_PROFILE_ID = (
    "hopper-generic-internal-computational-simulation-proxy-midterm/v1"
)
CAPABILITY_REVISION = (
    "simulation_proxy_generic_internal_lunar_ballistic/v3"
)
SUPPORT_PLANE_MODEL_ID = "hopper_horizontal_same_support_full_envelope_50mm/v1"
SUPPORT_HEIGHT_TOLERANCE_M = 0.05
STOP_CONDITION_ID = (
    "hopper_generic_internal_same_height_capture_le_2p5mps_"
    "stop_computational_simulation_proxy/v1"
)
STOP_EVALUATOR_ID = (
    "hopper_generic_internal_stop_evaluator_exact_binary64/v1"
)
ENERGY_MODEL_ID = (
    "hopper_generic_internal_launch_speed_squared_normalized_2p5mps_"
    "relative_energy_computational_simulation_proxy/v1"
)
ENERGY_EVALUATOR_ID = (
    "hopper_generic_internal_relative_energy_evaluator_exact_binary64/v1"
)


def _generic_record():
    return authority.HOPPER_GENERIC_INTERNAL_SIMULATION_PROXY_IMPLEMENTATION_V1


def _flat_anchor() -> FineSafetyAnchorV2:
    shape = (60, 60)
    snapshot = TerrainSnapshotV2(
        geometry=FineGridGeometryV2(width=shape[1], height=shape[0]),
        elevation_m=np.zeros(shape, dtype="<f8"),
        slope_deg=np.zeros(shape, dtype="<f8"),
        traversable_mask=np.ones(shape, dtype=bool),
        hard_obstacle_mask=np.zeros(shape, dtype=bool),
        observed_mask=np.ones(shape, dtype=bool),
        confidence=np.ones(shape, dtype="<f8"),
        provenance=TerrainProvenanceV2(
            source_kind="hopper-authority-test-fixture/v1",
            source_id="generic-internal-flat-terrain",
            source_hash="a" * 64,
            physical_obstacle_cells_written=False,
        ),
    )
    return FineSafetyAnchorV2(snapshot)


def _deadline() -> PlanningDeadlineV2:
    return PlanningDeadlineV2(0.0, 1_000.0, lambda: 0.0)


def _provider() -> hopper_provider.HopperPrimitiveProviderV2:
    return hopper_provider.HopperPrimitiveProviderV2(
        authority.HopperProviderAuthorityV2(
            hopper_profile=(
                authority.hopper_generic_internal_simulation_proxy_midterm_v1()
            ),
            parameter_set_id=PARAMETER_SET_ID,
            authority_schema_version="hopper-provider-authority/v1",
        )
    )


def _request(
    anchor: FineSafetyAnchorV2,
    goal: PoseStateV2,
    *,
    max_expanded_states: int,
) -> PlanningRequestV2:
    profile = authority.hopper_generic_internal_simulation_proxy_midterm_v1()
    return PlanningRequestV2(
        request_id="hopper-generic-internal-authority-test",
        platform_profile_id=profile.profile.profile_id,
        start_state=PoseStateV2(15.25, 15.25, 0.0),
        goal_state=goal,
        terrain_snapshot=anchor.snapshot,
        objective_profile=ObjectiveProfileV2(
            distance_weight=1.0,
            risk_weight=0.0,
            energy_weight=0.0,
            time_weight=0.0,
        ),
        resource_budget=ResourceBudgetV2(
            max_expanded_states=max_expanded_states,
            max_route_states=10,
            max_memory_bytes=0,
        ),
        timeout_s=1_000.0,
        accelerator_policy=AcceleratorPolicyV2.DISABLED,
        determinism_seed=17,
    )


def test_generic_internal_record_has_frozen_values_and_never_grants_formal_evidence():
    record = _generic_record()

    assert type(record) is authority.HopperGenericInternalSimulationProxyImplementationRecordV2
    assert record.parameter_set_id == PARAMETER_SET_ID
    assert record.base_profile_id == BASE_PROFILE_ID
    assert record.capability_revision == CAPABILITY_REVISION
    assert record.body_envelope_radius_m.hex() == "0x1.8000000000000p-2"
    assert record.launch_reference_height_m.hex() == "0x1.8000000000000p-1"
    assert record.arc_clearance_margin_m.hex() == "0x1.0000000000000p-3"
    assert record.landing_footprint_radius_m.hex() == "0x1.4000000000000p-1"
    assert record.support_plane_model_id == SUPPORT_PLANE_MODEL_ID
    assert record.support_height_tolerance_m.hex() == SUPPORT_HEIGHT_TOLERANCE_M.hex()
    assert record.relief_preservation_required is True
    assert record.schema_version == (
        "hopper-generic-internal-computational-simulation-proxy-"
        "implementation-record/v2"
    )
    assert (
        record.launch_reference_height_m
        >= record.body_envelope_radius_m + record.arc_clearance_margin_m
    )
    assert record.stop_condition == STOP_CONDITION_ID
    assert record.stop_evaluator_id == STOP_EVALUATOR_ID
    assert record.energy_model == ENERGY_MODEL_ID
    assert record.energy_evaluator_id == ENERGY_EVALUATOR_ID
    assert record.evidence_class == (
        "project_internal_generic_computational_simulation_proxy/v1"
    )
    assert record.simulation_proxy is True
    assert record.physical_capability_claimed is False
    assert record.hardware_certification_claimed is False
    assert record.formal_evidence_eligible is False


def test_generic_internal_evaluators_enforce_exact_speed_domain_and_frozen_words():
    record = _generic_record()

    for speed in (1.5, 2.0, 2.5):
        assert record.stop_evaluator(speed) is True
    assert record.stop_evaluator(3.0) is False
    assert record.stop_evaluator(nextafter(2.5, inf)) is False

    assert tuple(
        record.energy_evaluator(speed).hex()
        for speed in (1.5, 2.0, 2.5, 3.0)
    ) == (
        "0x1.70a3d70a3d70ap-2",
        "0x1.47ae147ae147cp-1",
        "0x1.0000000000000p+0",
        "0x1.70a3d70a3d70ap+0",
    )

    class FloatSubclass(float):
        pass

    for invalid in (True, 0, FloatSubclass(2.5)):
        with pytest.raises(TypeError):
            record.stop_evaluator(invalid)
        with pytest.raises(TypeError):
            record.energy_evaluator(invalid)
    for invalid in (0.0, -1.0, nan, inf):
        with pytest.raises(ValueError):
            record.stop_evaluator(invalid)
        with pytest.raises(ValueError):
            record.energy_evaluator(invalid)


def test_registry_is_sorted_duplicate_free_and_preserves_gate5b_tokens():
    registry = authority.HOPPER_PARAMETER_SET_REGISTRY_V2
    ids = tuple(record.parameter_set_id for record in registry)

    assert type(registry) is tuple
    assert ids == tuple(sorted(ids))
    assert len(ids) == len(set(ids)) == 2
    assert registry[0] is authority.HOPPER_GATE5B_ALGORITHM_FIXTURE_V1
    assert authority._hopper_parameter_set_lineage_token_v2(registry[0]) == (
        "hopper_gate5b_algorithm_fixture/v1",
        "hopper-lunar-ballistic-gate5b-fixture/v1",
        "0x1.0000000000000p-2",
        "0x1.0000000000000p-1",
        "0x1.999999999999ap-4",
        "0x1.3333333333333p-2",
        "hopper_same_height_nominal_recenter_capture_le_3mps_"
        "stop_simulation_proxy/v1",
        "hopper_launch_speed_squared_relative_energy/v1",
        "test_fixture",
        True,
        False,
        "hopper-parameter-set-record/v1",
    )
    assert authority.HOPPER_GATE5B_ALGORITHM_FIXTURE_V1.formal_evidence_eligible is False
    assert (
        authority._lookup_hopper_parameter_set_v2(PARAMETER_SET_ID)
        is _generic_record()
    )
    assert authority._lookup_hopper_parameter_set_v2(
        "hopper_candidate_only_not_supported/v1"
    ) == (
        "hopper-parameter-set-absent/v1",
        "hopper_candidate_only_not_supported/v1",
    )


def test_generic_profile_is_opt_in_and_default_capability_revision_is_unchanged():
    assert profiles.HOPPER_LUNAR_BALLISTIC_CAPABILITY_REVISION_V2 == (
        "simulation_proxy_lunar_ballistic/v1"
    )
    assert (
        profiles.HOPPER_GENERIC_INTERNAL_SIMULATION_PROXY_CAPABILITY_REVISION_V2
        == CAPABILITY_REVISION
    )

    fixture = authority.hopper_gate5b_algorithm_fixture_v1()
    generic = authority.hopper_generic_internal_simulation_proxy_midterm_v1()

    assert fixture.profile.capability_revision == (
        "simulation_proxy_lunar_ballistic/v1"
    )
    assert generic.profile.capability_revision == CAPABILITY_REVISION
    assert generic.profile.profile_id == BASE_PROFILE_ID
    assert generic.launch_speeds_mps == (1.5, 2.0, 2.5, 3.0)
    assert generic.launch_elevations_rad == fixture.launch_elevations_rad
    assert generic.azimuth_direction_count == 16
    assert generic.max_landing_slope_deg == 15.0
    assert generic.landing_probability_threshold == 0.99
    assert generic.profile.max_traversable_slope_deg == 30.0
    assert generic.body_envelope_radius_m == 0.375
    assert generic.launch_reference_height_m == 0.75
    assert generic.arc_clearance_margin_m == 0.125
    assert generic.landing_footprint_radius_m == 0.625
    assert profiles.HOPPER_SUPPORT_PLANE_MODEL_ID_V2 == SUPPORT_PLANE_MODEL_ID
    assert (
        profiles.HOPPER_SUPPORT_HEIGHT_TOLERANCE_M_V2.hex()
        == SUPPORT_HEIGHT_TOLERANCE_M.hex()
    )
    assert generic.stop_condition == STOP_CONDITION_ID
    assert generic.energy_model == ENERGY_MODEL_ID


def test_old_generic_v2_capability_cannot_carry_full_envelope_support_behavior():
    generic = authority.hopper_generic_internal_simulation_proxy_midterm_v1()
    old_platform = replace(
        generic.profile,
        capability_revision="simulation_proxy_generic_internal_lunar_ballistic/v2",
    )

    with pytest.raises(ValueError, match="capability_revision"):
        replace(generic, profile=old_platform)


@pytest.mark.parametrize(
    ("field", "direction"),
    (
        ("body_envelope_radius_m", inf),
        ("launch_reference_height_m", -inf),
        ("arc_clearance_margin_m", inf),
        ("landing_footprint_radius_m", -inf),
    ),
)
def test_generic_record_rejects_one_ulp_numeric_drift(field: str, direction: float):
    record = _generic_record()
    drifted = nextafter(getattr(record, field), direction)

    with pytest.raises(ValueError):
        replace(record, **{field: drifted})


@pytest.mark.parametrize(
    ("field", "mutated"),
    (
        ("support_plane_model_id", "hopper_horizontal_same_support_flat/v1"),
        ("support_height_tolerance_m", nextafter(0.05, inf)),
        ("relief_preservation_required", False),
    ),
)
def test_generic_record_rejects_support_contract_identity_drift(
    field: str,
    mutated: object,
):
    with pytest.raises(ValueError):
        replace(_generic_record(), **{field: mutated})


def test_jump_l2_accepts_2p5_and_rejects_3p0_only_at_stop_validation():
    anchor = _flat_anchor()
    profile = authority.hopper_generic_internal_simulation_proxy_midterm_v1()
    results = []
    for speed_index in (2, 3):
        candidate = hopper_oracle.HopperJumpCandidateV2(
            start_state=PoseStateV2(15.25, 15.25, 0.0),
            support_height_m=0.0,
            hopper_profile=profile,
            parameter_set_id=PARAMETER_SET_ID,
            speed_index=speed_index,
            elevation_index=1,
            azimuth_index=0,
        )
        results.append(
            hopper_oracle.validate_hopper_jump_l2(
                candidate,
                anchor,
                _deadline(),
            )
        )

    assert results[0].reason_code == "hopper_jump_l2_valid"
    assert results[0].stage == "stop_validation"
    assert results[0].selected_landing_mass is not None
    assert results[0].selected_landing_mass >= 0.99
    assert results[1].reason_code == "hopper_stop_condition_failed"
    assert results[1].stage == "stop_validation"


def test_provider_enumerates_all_192_actions_and_rejects_48_speed_3_actions():
    anchor = _flat_anchor()
    request = _request(
        anchor,
        PoseStateV2(15.123, 15.456, 0.0),
        max_expanded_states=1,
    )
    result = _provider().plan(request, anchor, _deadline())

    assert type(result) is PlanningFailureV2
    assert result.reason_code == "hopper_expansion_budget_exhausted"
    assert result.search_telemetry.expanded_states == 1
    assert result.search_telemetry.generated_primitives == 192
    assert result.search_telemetry.rejected_l2 == 48


def test_generic_provider_route_l2_and_api_seal_agree_on_success():
    anchor = _flat_anchor()
    request = _request(
        anchor,
        PoseStateV2(14.047186939188279, 15.25, 0.0),
        max_expanded_states=2,
    )
    provider = _provider()

    direct = provider.plan(request, anchor, _deadline())
    dispatched = hopper_api.dispatch_hopper_provider_v2(
        request,
        provider.profile,
        provider,
        anchor,
        _deadline(),
    )

    assert type(direct) is PlanningSuccessV2
    assert type(dispatched) is PlanningSuccessV2
    assert direct.route == dispatched.route
    assert direct.cost_breakdown == dispatched.cost_breakdown
    assert direct.validation_evidence == dispatched.validation_evidence
    replay = route_validation.validate_hopper_route_l2(
        direct.route,
        request,
        anchor,
        provider.hopper_authority,
        _deadline(),
    )
    assert replay.passed is True
    assert replay.reason_code == "hopper_route_l2_valid"
    assert replay.cost_breakdown == direct.cost_breakdown


def test_gate5b_fixture_identity_values_and_speed_3_behavior_are_unchanged():
    record = authority.HOPPER_GATE5B_ALGORITHM_FIXTURE_V1
    profile = authority.hopper_gate5b_algorithm_fixture_v1()
    candidate = hopper_oracle.HopperJumpCandidateV2(
        start_state=PoseStateV2(15.25, 15.25, 0.0),
        support_height_m=0.0,
        hopper_profile=profile,
        parameter_set_id=record.parameter_set_id,
        speed_index=3,
        elevation_index=1,
        azimuth_index=0,
    )

    result = hopper_oracle.validate_hopper_jump_l2(
        candidate,
        _flat_anchor(),
        _deadline(),
    )

    assert record.body_envelope_radius_m.hex() == "0x1.0000000000000p-2"
    assert record.launch_reference_height_m.hex() == "0x1.0000000000000p-1"
    assert record.arc_clearance_margin_m.hex() == "0x1.999999999999ap-4"
    assert record.landing_footprint_radius_m.hex() == "0x1.3333333333333p-2"
    assert record.stop_evaluator(3.0) is True
    assert record.energy_evaluator(3.0) == 1.0
    assert record.evidence_class == "test_fixture"
    assert record.formal_evidence_eligible is False
    assert profile.profile.capability_revision == (
        "simulation_proxy_lunar_ballistic/v1"
    )
    assert result.reason_code == "hopper_jump_l2_valid"


def test_generic_lineage_binds_evaluator_ids_and_source_digests_without_objects():
    record = _generic_record()
    in_memory = authority._hopper_parameter_set_in_memory_token_v2(record)
    lineage = authority._hopper_parameter_set_lineage_token_v2(record)
    encoded = canonical_json_bytes(lineage)

    assert record.stop_evaluator in in_memory
    assert record.energy_evaluator in in_memory
    assert record.stop_evaluator not in lineage
    assert record.energy_evaluator not in lineage
    assert record.stop_evaluator_id in lineage
    assert record.energy_evaluator_id in lineage
    assert record.stop_evaluator_source_sha256 in lineage
    assert record.energy_evaluator_source_sha256 in lineage
    assert record.support_plane_model_id in lineage
    assert record.support_height_tolerance_m.hex() in lineage
    assert record.relief_preservation_required in lineage
    assert record.stop_evaluator_source_sha256 == sha256(
        getsource(record.stop_evaluator).encode("utf-8")
    ).hexdigest()
    assert record.energy_evaluator_source_sha256 == sha256(
        getsource(record.energy_evaluator).encode("utf-8")
    ).hexdigest()
    assert b"<function" not in encoded
    assert b" at 0x" not in encoded
    assert b"repr(" not in encoded


def test_registry_lineage_is_identical_across_python_hash_seeds():
    source_root = Path(__file__).resolve().parents[1] / "src"
    code = (
        "from hashlib import sha256;"
        "from path_planner.v2.hopper_authority import "
        "HOPPER_PARAMETER_SET_REGISTRY_LINEAGE_TOKEN_V2 as token;"
        "from path_planner.v2.serialization import canonical_json_bytes;"
        "print(sha256(canonical_json_bytes(token)).hexdigest())"
    )
    digests = []
    for seed in ("7", "8675309"):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = seed
        environment["PYTHONPATH"] = str(source_root)
        digests.append(
            subprocess.check_output(
                [sys.executable, "-c", code],
                cwd=source_root.parent,
                env=environment,
                text=True,
            ).strip()
        )

    assert digests[0] == digests[1]
    assert len(digests[0]) == 64


def test_evaluator_and_registry_drift_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    record = _generic_record()
    original_evaluator = record.energy_evaluator
    object.__setattr__(record, "energy_evaluator", lambda speed: speed)
    try:
        with pytest.raises(
            ValueError,
            match="hopper_authority_contract_mismatch",
        ):
            authority._hopper_parameter_set_in_memory_token_v2(record)
        with pytest.raises(
            ValueError,
            match="hopper_authority_contract_mismatch",
        ):
            authority._lookup_hopper_parameter_set_v2(PARAMETER_SET_ID)
    finally:
        object.__setattr__(record, "energy_evaluator", original_evaluator)

    monkeypatch.setattr(
        authority,
        "HOPPER_PARAMETER_SET_REGISTRY_V2",
        tuple(reversed(authority.HOPPER_PARAMETER_SET_REGISTRY_V2)),
    )
    with pytest.raises(
        ValueError,
        match="hopper_authority_contract_mismatch",
    ):
        authority._lookup_hopper_parameter_set_v2(PARAMETER_SET_ID)


def test_generic_profile_one_ulp_drift_is_authority_mismatch():
    anchor = _flat_anchor()
    provider = _provider()
    object.__setattr__(
        provider.hopper_authority.hopper_profile,
        "body_envelope_radius_m",
        nextafter(0.375, inf),
    )
    request = _request(
        anchor,
        PoseStateV2(15.123, 15.456, 0.0),
        max_expanded_states=1,
    )

    result = provider.plan(request, anchor, _deadline())

    assert type(result) is PlanningFailureV2
    assert result.reason_code == "hopper_authority_contract_mismatch"


@pytest.mark.parametrize(
    ("field", "mutated"),
    (
        ("gravity_mps2", nextafter(1.62, inf)),
        ("launch_speeds_mps", (1.5, 2.0, 2.5)),
        ("landing_probability_threshold", nextafter(0.99, -inf)),
    ),
)
def test_shared_profile_record_match_rejects_action_or_safety_mutation(
    field: str,
    mutated: object,
):
    profile = authority.hopper_generic_internal_simulation_proxy_midterm_v1()
    object.__setattr__(profile, field, mutated)

    assert (
        authority._hopper_profile_matches_parameter_set_record_v2(
            profile,
            _generic_record(),
        )
        is False
    )


def test_in_place_evaluator_code_mutation_fails_identity_seal():
    record = _generic_record()
    original_code = record.energy_evaluator.__code__

    def altered_operation_order(speed_mps: float) -> float:
        return (speed_mps * speed_mps) / 6.25

    record.energy_evaluator.__code__ = altered_operation_order.__code__
    try:
        with pytest.raises(
            ValueError,
            match="hopper_authority_contract_mismatch",
        ):
            authority._hopper_parameter_set_in_memory_token_v2(record)
    finally:
        record.energy_evaluator.__code__ = original_code


def test_direct_provider_rejects_jump_l2_binding_drift(
    monkeypatch: pytest.MonkeyPatch,
):
    anchor = _flat_anchor()
    request = _request(
        anchor,
        PoseStateV2(15.123, 15.456, 0.0),
        max_expanded_states=1,
    )
    monkeypatch.setattr(
        hopper_provider,
        "validate_hopper_jump_l2",
        lambda *args: None,
    )

    result = _provider().plan(request, anchor, _deadline())

    assert type(result) is PlanningFailureV2
    assert result.reason_code == "hopper_authority_contract_mismatch"


def test_direct_provider_rejects_route_l2_binding_drift(
    monkeypatch: pytest.MonkeyPatch,
):
    anchor = _flat_anchor()
    request = _request(
        anchor,
        PoseStateV2(14.047186939188279, 15.25, 0.0),
        max_expanded_states=2,
    )
    trusted_route_l2 = route_validation.validate_hopper_route_l2

    def equivalent_wrapper(*args, **kwargs):
        return trusted_route_l2(*args, **kwargs)

    monkeypatch.setattr(
        route_validation,
        "validate_hopper_route_l2",
        equivalent_wrapper,
    )

    result = _provider().plan(request, anchor, _deadline())

    assert type(result) is PlanningFailureV2
    assert result.reason_code == "hopper_authority_contract_mismatch"


@pytest.mark.parametrize(
    ("field", "mutated"),
    (
        ("gravity_mps2", nextafter(1.62, inf)),
        ("launch_speeds_mps", (1.5, 2.0, 2.5)),
        ("landing_probability_threshold", nextafter(0.99, -inf)),
    ),
)
@pytest.mark.parametrize("consumer", ("provider", "api"))
def test_post_construction_profile_mutation_is_structured_authority_mismatch(
    field: str,
    mutated: object,
    consumer: str,
):
    anchor = _flat_anchor()
    provider = _provider()
    object.__setattr__(provider.hopper_authority.hopper_profile, field, mutated)
    request = _request(
        anchor,
        PoseStateV2(15.123, 15.456, 0.0),
        max_expanded_states=1,
    )

    if consumer == "provider":
        result = provider.plan(request, anchor, _deadline())
    else:
        result = hopper_api.dispatch_hopper_provider_v2(
            request,
            provider.profile,
            provider,
            anchor,
            _deadline(),
        )

    assert type(result) is PlanningFailureV2
    assert result.reason_code == "hopper_authority_contract_mismatch"


def test_direct_route_l2_rejects_its_jump_l2_binding_drift(
    monkeypatch: pytest.MonkeyPatch,
):
    anchor = _flat_anchor()
    request = _request(
        anchor,
        PoseStateV2(14.047186939188279, 15.25, 0.0),
        max_expanded_states=2,
    )
    provider = _provider()
    direct = provider.plan(request, anchor, _deadline())
    assert type(direct) is PlanningSuccessV2
    monkeypatch.setattr(
        route_validation,
        "validate_hopper_jump_l2",
        lambda *args: None,
    )

    replay = route_validation.validate_hopper_route_l2(
        direct.route,
        request,
        anchor,
        provider.hopper_authority,
        _deadline(),
    )

    assert replay.passed is False
    assert replay.reason_code == "hopper_authority_contract_mismatch"


def test_api_seal_rejects_provider_plan_function_drift(
    monkeypatch: pytest.MonkeyPatch,
):
    anchor = _flat_anchor()
    request = _request(
        anchor,
        PoseStateV2(14.047186939188279, 15.25, 0.0),
        max_expanded_states=2,
    )
    provider = _provider()
    trusted_plan = hopper_provider.HopperPrimitiveProviderV2.plan

    def altered_plan(self, planning_request, safety_anchor, deadline):
        return trusted_plan(self, planning_request, safety_anchor, deadline)

    monkeypatch.setattr(
        hopper_provider.HopperPrimitiveProviderV2,
        "plan",
        altered_plan,
    )

    result = hopper_api.dispatch_hopper_provider_v2(
        request,
        provider.profile,
        provider,
        anchor,
        _deadline(),
    )

    assert type(result) is PlanningFailureV2
    assert result.reason_code == "hopper_authority_contract_mismatch"


@pytest.mark.parametrize("consumer", ("provider", "jump_l2", "route_l2"))
def test_each_consumer_rejects_shared_profile_match_helper_drift(
    monkeypatch: pytest.MonkeyPatch,
    consumer: str,
):
    anchor = _flat_anchor()
    provider = _provider()
    if consumer == "provider":
        monkeypatch.setattr(
            hopper_provider,
            "_hopper_profile_matches_parameter_set_record_v2",
            lambda *args: True,
        )
        result = provider.plan(
            _request(
                anchor,
                PoseStateV2(15.123, 15.456, 0.0),
                max_expanded_states=1,
            ),
            anchor,
            _deadline(),
        )
        assert type(result) is PlanningFailureV2
        assert result.reason_code == "hopper_authority_contract_mismatch"
        return

    if consumer == "jump_l2":
        monkeypatch.setattr(
            hopper_oracle,
            "_hopper_profile_matches_parameter_set_record_v2",
            lambda *args: True,
        )
        result = hopper_oracle.validate_hopper_jump_l2(
            hopper_oracle.HopperJumpCandidateV2(
                start_state=PoseStateV2(15.25, 15.25, 0.0),
                support_height_m=0.0,
                hopper_profile=(
                    authority.hopper_generic_internal_simulation_proxy_midterm_v1()
                ),
                parameter_set_id=PARAMETER_SET_ID,
                speed_index=2,
                elevation_index=1,
                azimuth_index=0,
            ),
            anchor,
            _deadline(),
        )
        assert result.reason_code == "hopper_authority_contract_mismatch"
        return

    request = _request(
        anchor,
        PoseStateV2(14.047186939188279, 15.25, 0.0),
        max_expanded_states=2,
    )
    direct = provider.plan(request, anchor, _deadline())
    assert type(direct) is PlanningSuccessV2
    monkeypatch.setattr(
        route_validation,
        "_hopper_profile_matches_parameter_set_record_v2",
        lambda *args: True,
    )
    result = route_validation.validate_hopper_route_l2(
        direct.route,
        request,
        anchor,
        provider.hopper_authority,
        _deadline(),
    )
    assert result.passed is False
    assert result.reason_code == "hopper_authority_contract_mismatch"


def test_api_seal_rejects_parameter_token_helper_drift(
    monkeypatch: pytest.MonkeyPatch,
):
    anchor = _flat_anchor()
    request = _request(
        anchor,
        PoseStateV2(14.047186939188279, 15.25, 0.0),
        max_expanded_states=2,
    )
    provider = _provider()
    monkeypatch.setattr(
        hopper_api,
        "_hopper_parameter_set_in_memory_token_v2",
        lambda *args: ("forged-token",),
    )

    result = hopper_api.dispatch_hopper_provider_v2(
        request,
        provider.profile,
        provider,
        anchor,
        _deadline(),
    )

    assert type(result) is PlanningFailureV2
    assert result.reason_code == "hopper_authority_contract_mismatch"


@pytest.mark.parametrize("tamper_target", ("jump_l2", "route_l2"))
def test_api_seal_rejects_jump_or_route_l2_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
    tamper_target: str,
):
    anchor = _flat_anchor()
    request = _request(
        anchor,
        PoseStateV2(14.047186939188279, 15.25, 0.0),
        max_expanded_states=2,
    )
    provider = _provider()
    if tamper_target == "jump_l2":
        monkeypatch.setattr(
            hopper_provider,
            "validate_hopper_jump_l2",
            lambda *args: None,
        )
    else:
        monkeypatch.setattr(
            route_validation,
            "validate_hopper_route_l2",
            lambda *args: None,
        )

    result = hopper_api.dispatch_hopper_provider_v2(
        request,
        provider.profile,
        provider,
        anchor,
        _deadline(),
    )

    assert type(result) is PlanningFailureV2
    assert result.reason_code == "hopper_authority_contract_mismatch"
