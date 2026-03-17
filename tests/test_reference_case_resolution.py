from __future__ import annotations

import pathlib
import unittest
from typing import Any

from scripts.authority import (
    AuthoritySelectionError,
    allowed_phase_gate_case_roles,
    case_meta_schema,
    load_authority_bundle,
    resolve_phase_gate_case,
    resolve_reference_case,
    resolve_reference_case_by_frozen_id,
    stage_plan_schema,
    validate_case_meta,
    validate_frozen_ladder,
    validate_stage_plan,
)


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def sample_case_meta_payload(
    bundle: Any,
    *,
    case_role: str = "R1-core",
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = resolve_reference_case(bundle, case_role=case_role)
    payload = {
        "schema_version": "1.0.0",
        "case_id": resolved.frozen_id,
        "case_role": resolved.case_role,
        "ladder_position": resolved.ladder_position,
        "phase_gates": list(resolved.phase_gates),
        "baseline": "Baseline A",
        "runtime_base": "OpenFOAM 12",
        "reviewed_source_tuple_id": "SRC_OPENFOAM12_REFERENCE",
        "requested_vof_solver_mode": "vof_transient_preconditioned",
        "resolved_vof_solver_exec": "incompressibleVoF",
        "resolved_pressure_backend": "native_cpu",
        "openfoam_bashrc_used": "/opt/openfoam12/etc/bashrc",
        "available_commands": {
            "foamRun": "/opt/openfoam12/bin/foamRun",
            "checkMesh": "/opt/openfoam12/bin/checkMesh",
            "setFields": "/opt/openfoam12/bin/setFields",
        },
        "mesh_full_360": 1,
        "mesh_resolution_scale": 2.0,
        "hydraulic_domain_mode": "internal_only",
        "near_field_radius_d": 10.0,
        "near_field_length_d": 20.0,
        "steady_end_time_iter": 10,
        "steady_write_interval_iter": 5,
        "steady_turbulence_model": "kOmegaSST",
        "vof_turbulence_model": "laminar",
        "delta_t_s": 1e-8,
        "write_interval_s": 1e-8,
        "end_time_s": 2e-8,
        "max_co": 0.05,
        "max_alpha_co": 0.01,
        "resolved_direct_slot_numerics": {
            "slot_count": 6,
            "nominal_slot_width_mm": 0.64,
        },
        "startup_fill_extension_d": 0.0,
        "air_core_seed_radius_d_requested": 1.0,
        "air_core_seed_radius_m_resolved": 0.0005,
        "air_core_seed_cap_applied": False,
        "fill_radius_m_resolved": 0.00075,
        "fill_z_start_m": -0.001,
        "fill_z_stop_m": 0.001,
        "DeltaP_Pa": 6894760.0,
        "DeltaP_effective_Pa": 6894760.0,
        "check_valve_loss_applied": False,
        "provenance": {
            "probe_payload": "baseline_a/probe.json",
            "host_env": "baseline_a/host_env.json",
            "manifest_refs": "baseline_a/manifest_refs.json",
        },
    }
    if overrides:
        payload.update(overrides)
    return payload


def sample_stage_plan_payload(
    bundle: Any,
    *,
    case_role: str = "R1-core",
    phase_gate: str = "Phase 5",
    conditional_reason: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = resolve_reference_case(bundle, case_role=case_role)
    unconditional_case_roles = allowed_phase_gate_case_roles(bundle, phase_gate=phase_gate)
    all_allowed_case_roles = allowed_phase_gate_case_roles(
        bundle,
        phase_gate=phase_gate,
        include_conditional=True,
    )
    conditional_selection = case_role not in unconditional_case_roles
    available_case_roles = (
        list(all_allowed_case_roles) if conditional_selection else list(unconditional_case_roles)
    )
    payload = {
        "schema_version": "1.0.0",
        "case_id": resolved.frozen_id,
        "case_role": resolved.case_role,
        "phase_gate": phase_gate,
        "baseline": "Baseline A",
        "runtime_base": "OpenFOAM 12",
        "reviewed_source_tuple_id": "SRC_OPENFOAM12_REFERENCE",
        "provenance": {
            "probe_payload": "baseline_a/probe.json",
            "host_env": "baseline_a/host_env.json",
            "manifest_refs": "baseline_a/manifest_refs.json",
        },
        "phase_gate_selection": {
            "selected_case_role": resolved.case_role,
            "available_case_roles": available_case_roles,
            "ordered_ladder": list(bundle.ladder.ordered_case_ids),
            "conditional_selection": conditional_selection,
        },
        "stages": [
            {
                "name": "transient_run",
                "cmd": "foamRun -solver incompressibleVoF",
            }
        ],
    }
    if conditional_reason is not None:
        payload["phase_gate_selection"]["conditional_reason"] = conditional_reason
    if overrides:
        payload.update(overrides)
    return payload


class ReferenceCaseResolutionTests(unittest.TestCase):
    def test_each_frozen_role_resolves_to_the_authority_case_id(self) -> None:
        bundle = load_authority_bundle(repo_root())
        expected = {
            "R2": "phase0_r2_dambreak_reference_v1",
            "R1-core": "phase0_r1_core_57_28_1000_internal_generic_v1",
            "R1": "phase0_r1_57_28_1000_internal_v1",
            "R0": "phase0_r0_57_28_1000_full360_v1",
        }

        for case_role, frozen_id in expected.items():
            with self.subTest(case_role=case_role):
                resolved = resolve_reference_case(bundle, case_role=case_role)
                self.assertEqual(resolved.case_role, case_role)
                self.assertEqual(resolved.frozen_id, frozen_id)
                self.assertEqual(
                    resolve_reference_case_by_frozen_id(bundle, frozen_id=frozen_id).case_role,
                    case_role,
                )

    def test_unknown_case_role_and_frozen_id_fail_fast(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with self.assertRaisesRegex(AuthoritySelectionError, "unknown case role 'R9'"):
            resolve_reference_case(bundle, case_role="R9")

        with self.assertRaisesRegex(
            AuthoritySelectionError,
            "unknown frozen case id 'phase0_r9_unknown_v1'",
        ):
            resolve_reference_case_by_frozen_id(bundle, frozen_id="phase0_r9_unknown_v1")

    def test_reordered_validation_ladder_fails_fast(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with self.assertRaisesRegex(
            AuthoritySelectionError,
            "validation ladder must remain R2 -> R1-core -> R1 -> R0",
        ):
            validate_frozen_ladder(bundle, ("R2", "R1", "R1-core", "R0"))

    def test_phase_gate_role_availability_matches_frozen_mapping(self) -> None:
        bundle = load_authority_bundle(repo_root())

        self.assertEqual(
            allowed_phase_gate_case_roles(bundle, phase_gate="Phase 0"),
            ("R2", "R1-core", "R1", "R0"),
        )
        self.assertEqual(
            allowed_phase_gate_case_roles(bundle, phase_gate="Phase 2"),
            ("R2", "R1-core"),
        )
        self.assertEqual(
            allowed_phase_gate_case_roles(bundle, phase_gate="Phase 2", include_conditional=True),
            ("R2", "R1-core", "R1"),
        )
        self.assertEqual(
            allowed_phase_gate_case_roles(bundle, phase_gate="Phase 5"),
            ("R2", "R1-core"),
        )
        self.assertEqual(
            allowed_phase_gate_case_roles(bundle, phase_gate="Phase 6"),
            ("R1",),
        )
        self.assertEqual(
            allowed_phase_gate_case_roles(bundle, phase_gate="Phase 8"),
            ("R1", "R0", "R1-core"),
        )

    def test_out_of_scope_phase_gate_case_selection_fails_fast(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with self.assertRaisesRegex(
            AuthoritySelectionError,
            "phase gate 'Phase 5' does not allow case role 'R0'",
        ):
            resolve_phase_gate_case(bundle, phase_gate="Phase 5", case_role="R0")

    def test_phase_gate_conditional_selection_requires_explicit_opt_in(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with self.assertRaisesRegex(
            AuthoritySelectionError,
            "phase gate 'Phase 2' allows case role 'R1' only conditionally",
        ):
            resolve_phase_gate_case(bundle, phase_gate="Phase 2", case_role="R1")

        resolved = resolve_phase_gate_case(
            bundle,
            phase_gate="Phase 2",
            case_role="R1",
            allow_conditional=True,
        )
        self.assertEqual(resolved.case_role, "R1")

    def test_case_meta_schema_and_validation_use_authority_owned_roles(self) -> None:
        bundle = load_authority_bundle(repo_root())
        schema = case_meta_schema(bundle)

        self.assertEqual(schema["canonical_name"], "case_meta.json")
        self.assertEqual(schema["type"], "object")
        self.assertEqual(
            schema["properties"]["case_role"]["enum"],
            ["R2", "R1-core", "R1", "R0"],
        )
        self.assertIn("case_id", schema["required"])
        self.assertIn("baseline", schema["required"])
        self.assertIn("provenance", schema["required"])
        self.assertIn("resolved_direct_slot_numerics", schema["required"])
        self.assertIn("openfoam_bashrc_used", schema["required"])
        self.assertEqual(schema["properties"]["phase_gates"]["minItems"], 1)
        self.assertEqual(schema["properties"]["mesh_full_360"]["enum"], [0, 1])
        self.assertEqual(schema["properties"]["provenance"]["required"], ["probe_payload", "host_env", "manifest_refs"])
        case_variants = schema["allOf"][0]["oneOf"]
        self.assertEqual(len(case_variants), 4)
        r1_core_variant = next(
            variant
            for variant in case_variants
            if variant["properties"]["case_role"]["const"] == "R1-core"
        )
        self.assertEqual(
            r1_core_variant["properties"]["case_id"]["const"],
            "phase0_r1_core_57_28_1000_internal_generic_v1",
        )
        self.assertEqual(r1_core_variant["properties"]["ladder_position"]["const"], 2)
        self.assertEqual(r1_core_variant["properties"]["phase_gates"]["minItems"], 4)

        validate_case_meta(bundle, sample_case_meta_payload(bundle))

    def test_case_meta_validation_rejects_mismatched_role_and_case_id(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with self.assertRaisesRegex(
            AuthoritySelectionError,
            "case_meta.json case_role 'R1' must resolve to case_id 'phase0_r1_57_28_1000_internal_v1'",
        ):
            validate_case_meta(
                bundle,
                sample_case_meta_payload(
                    bundle,
                    case_role="R1",
                    overrides={"case_id": "phase0_r1_core_57_28_1000_internal_generic_v1"},
                ),
            )

    def test_case_meta_validation_rejects_non_canonical_types(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with self.assertRaisesRegex(
            AuthoritySelectionError,
            "case_meta.json ladder_position must be an integer",
        ):
            validate_case_meta(
                bundle,
                sample_case_meta_payload(bundle, overrides={"ladder_position": "2"}),
            )

    def test_case_meta_validation_allows_phase_gate_reordering_with_same_members(self) -> None:
        bundle = load_authority_bundle(repo_root())

        validate_case_meta(
            bundle,
            sample_case_meta_payload(
                bundle,
                overrides={"phase_gates": ["Phase 8", "Phase 5", "Phase 2", "Phase 0"]},
            ),
        )

        validate_case_meta(
            bundle,
            sample_case_meta_payload(bundle, case_role="R1"),
        )

        with self.assertRaisesRegex(
            AuthoritySelectionError,
            "case_meta.json phase_gates must be a list of phase-gate names",
        ):
            validate_case_meta(
                bundle,
                sample_case_meta_payload(bundle, overrides={"phase_gates": {"Phase 0": True}}),
            )

    def test_case_meta_validation_requires_provenance_references(self) -> None:
        bundle = load_authority_bundle(repo_root())
        payload = sample_case_meta_payload(bundle)
        del payload["provenance"]["manifest_refs"]

        with self.assertRaisesRegex(
            AuthoritySelectionError,
            "case_meta.json provenance is missing required fields: manifest_refs",
        ):
            validate_case_meta(bundle, payload)

    def test_stage_plan_schema_and_validation_enforce_phase_gate_selection(self) -> None:
        bundle = load_authority_bundle(repo_root())
        schema = stage_plan_schema(bundle)

        self.assertEqual(schema["canonical_name"], "stage_plan.json")
        self.assertEqual(schema["type"], "object")
        self.assertIn("baseline", schema["required"])
        self.assertIn("provenance", schema["required"])
        self.assertEqual(
            schema["properties"]["phase_gate_selection"]["properties"]["ordered_ladder"]["const"],
            ["R2", "R1-core", "R1", "R0"],
        )
        conditional_reason_schema = schema["properties"]["phase_gate_selection"]["properties"][
            "conditional_reason"
        ]
        self.assertEqual(conditional_reason_schema["minLength"], 1)
        self.assertEqual(
            schema["properties"]["stages"]["items"]["properties"]["name"]["minLength"],
            1,
        )
        self.assertEqual(
            schema["properties"]["stages"]["items"]["properties"]["name"]["pattern"],
            r".*\S.*",
        )
        self.assertEqual(
            schema["properties"]["stages"]["items"]["properties"]["cmd"]["minLength"],
            1,
        )
        self.assertEqual(
            schema["properties"]["stages"]["items"]["properties"]["cmd"]["pattern"],
            r".*\S.*",
        )
        self.assertEqual(schema["properties"]["stages"]["minItems"], 1)
        self.assertIn("allOf", schema["properties"]["phase_gate_selection"])
        self.assertEqual(
            schema["properties"]["phase_gate_selection"]["properties"]["conditional_reason"][
                "pattern"
            ],
            r".*\S.*",
        )
        self.assertEqual(
            len(schema["properties"]["phase_gate_selection"]["allOf"]),
            2,
        )
        self.assertEqual(
            schema["properties"]["phase_gate_selection"]["allOf"][1],
            {
                "if": {
                    "properties": {"conditional_selection": {"const": False}},
                    "required": ["conditional_selection"],
                },
                "then": {"not": {"required": ["conditional_reason"]}},
            },
        )
        stage_plan_variants = schema["allOf"][0]["oneOf"]
        self.assertEqual(len(stage_plan_variants), 14)
        phase5_r1_core_variant = next(
            variant
            for variant in stage_plan_variants
            if variant["properties"]["phase_gate"]["const"] == "Phase 5"
            and variant["properties"]["case_role"]["const"] == "R1-core"
        )
        self.assertEqual(
            phase5_r1_core_variant["properties"]["case_id"]["const"],
            "phase0_r1_core_57_28_1000_internal_generic_v1",
        )
        self.assertEqual(
            phase5_r1_core_variant["properties"]["phase_gate_selection"]["properties"][
                "selected_case_role"
            ]["const"],
            "R1-core",
        )
        self.assertEqual(
            phase5_r1_core_variant["properties"]["phase_gate_selection"]["properties"][
                "conditional_selection"
            ]["const"],
            False,
        )
        phase2_r1_variant = next(
            variant
            for variant in stage_plan_variants
            if variant["properties"]["phase_gate"]["const"] == "Phase 2"
            and variant["properties"]["case_role"]["const"] == "R1"
        )
        self.assertEqual(
            phase2_r1_variant["properties"]["case_id"]["const"],
            "phase0_r1_57_28_1000_internal_v1",
        )
        self.assertEqual(
            phase2_r1_variant["properties"]["phase_gate_selection"]["properties"][
                "selected_case_role"
            ]["const"],
            "R1",
        )
        self.assertEqual(
            phase2_r1_variant["properties"]["phase_gate_selection"]["properties"][
                "conditional_selection"
            ]["const"],
            True,
        )

        validate_stage_plan(bundle, sample_stage_plan_payload(bundle))

    def test_stage_plan_validation_rejects_phase_local_ladder_rewrites(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with self.assertRaisesRegex(
            AuthoritySelectionError,
            "stage_plan.json ordered_ladder must remain R2 -> R1-core -> R1 -> R0",
        ):
            payload = sample_stage_plan_payload(bundle)
            payload["phase_gate_selection"]["ordered_ladder"] = ["R2", "R1", "R1-core", "R0"]
            validate_stage_plan(
                bundle,
                payload,
            )

    def test_stage_plan_validation_rejects_non_canonical_types(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with self.assertRaisesRegex(
            AuthoritySelectionError,
            "stage_plan.json available_case_roles must be a list of case roles",
        ):
            payload = sample_stage_plan_payload(bundle)
            payload["phase_gate_selection"]["available_case_roles"] = {"R2": True, "R1-core": True}
            validate_stage_plan(
                bundle,
                payload,
            )

        with self.assertRaisesRegex(
            AuthoritySelectionError,
            "stage_plan.json ordered_ladder must be a list of case roles",
        ):
            payload = sample_stage_plan_payload(bundle)
            payload["phase_gate_selection"]["ordered_ladder"] = {"R2": True, "R1-core": True}
            validate_stage_plan(
                bundle,
                payload,
            )

        with self.assertRaisesRegex(
            AuthoritySelectionError,
            "stage_plan.json each stage must define string name and cmd values",
        ):
            payload = sample_stage_plan_payload(bundle)
            payload["stages"] = [{"name": 1, "cmd": 2}]
            validate_stage_plan(
                bundle,
                payload,
            )

        with self.assertRaisesRegex(
            AuthoritySelectionError,
            "stage_plan.json stage cwd must be a string when provided",
        ):
            payload = sample_stage_plan_payload(bundle)
            payload["stages"] = [{"name": "transient_run", "cmd": "foamRun", "cwd": 1}]
            validate_stage_plan(
                bundle,
                payload,
            )

    def test_stage_plan_validation_allows_available_role_reordering_with_same_members(self) -> None:
        bundle = load_authority_bundle(repo_root())

        validate_stage_plan(
            bundle,
            sample_stage_plan_payload(
                bundle,
                overrides={
                    "phase_gate_selection": {
                        "selected_case_role": "R1-core",
                        "available_case_roles": ["R1-core", "R2"],
                        "ordered_ladder": ["R2", "R1-core", "R1", "R0"],
                        "conditional_selection": False,
                    }
                },
            ),
        )

    def test_stage_plan_validation_allows_documented_conditional_phase2_selection(self) -> None:
        bundle = load_authority_bundle(repo_root())

        validate_stage_plan(
            bundle,
            sample_stage_plan_payload(
                bundle,
                case_role="R1",
                phase_gate="Phase 2",
                conditional_reason="patch-manifest coverage under test",
            ),
        )

    def test_stage_plan_validation_requires_explicit_opt_in_for_conditional_phase2_selection(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with self.assertRaisesRegex(
            AuthoritySelectionError,
            "stage_plan.json conditional_selection is only valid for authority-conditional case roles",
        ):
            payload = sample_stage_plan_payload(bundle)
            payload["phase_gate_selection"]["conditional_selection"] = True
            payload["phase_gate_selection"]["conditional_reason"] = "not actually conditional"
            validate_stage_plan(
                bundle,
                payload,
            )

        with self.assertRaisesRegex(
            AuthoritySelectionError,
            "phase gate 'Phase 2' allows case role 'R1' only conditionally",
        ):
            payload = sample_stage_plan_payload(
                bundle,
                case_role="R1",
                phase_gate="Phase 2",
                conditional_reason="patch-manifest coverage under test",
            )
            payload["phase_gate_selection"]["available_case_roles"] = ["R2", "R1-core"]
            payload["phase_gate_selection"]["conditional_selection"] = False
            del payload["phase_gate_selection"]["conditional_reason"]
            validate_stage_plan(
                bundle,
                payload,
            )

    def test_stage_plan_validation_requires_conditional_selection_field(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with self.assertRaisesRegex(
            AuthoritySelectionError,
            "stage_plan.json phase_gate_selection is missing required fields: conditional_selection",
        ):
            payload = sample_stage_plan_payload(bundle)
            del payload["phase_gate_selection"]["conditional_selection"]
            validate_stage_plan(
                bundle,
                payload,
            )


if __name__ == "__main__":
    unittest.main()
