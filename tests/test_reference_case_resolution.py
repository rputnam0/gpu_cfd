from __future__ import annotations

import pathlib
import unittest

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
        self.assertEqual(schema["properties"]["phase_gates"]["minItems"], 1)

        validate_case_meta(
            bundle,
            {
                "schema_version": "1.0.0",
                "case_id": "phase0_r1_core_57_28_1000_internal_generic_v1",
                "case_role": "R1-core",
                "ladder_position": 2,
                "phase_gates": ["Phase 0", "Phase 2", "Phase 5", "Phase 8"],
            },
        )

    def test_case_meta_validation_rejects_mismatched_role_and_case_id(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with self.assertRaisesRegex(
            AuthoritySelectionError,
            "case_meta.json case_role 'R1' must resolve to case_id 'phase0_r1_57_28_1000_internal_v1'",
        ):
            validate_case_meta(
                bundle,
                {
                    "schema_version": "1.0.0",
                    "case_id": "phase0_r1_core_57_28_1000_internal_generic_v1",
                    "case_role": "R1",
                    "ladder_position": 3,
                    "phase_gates": ["Phase 0", "Phase 2", "Phase 6", "Phase 7", "Phase 8"],
                },
            )

    def test_case_meta_validation_rejects_non_canonical_types(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with self.assertRaisesRegex(
            AuthoritySelectionError,
            "case_meta.json ladder_position must be an integer",
        ):
            validate_case_meta(
                bundle,
                {
                    "schema_version": "1.0.0",
                    "case_id": "phase0_r1_core_57_28_1000_internal_generic_v1",
                    "case_role": "R1-core",
                    "ladder_position": "2",
                    "phase_gates": ["Phase 0", "Phase 2", "Phase 5", "Phase 8"],
                },
            )

    def test_case_meta_validation_allows_phase_gate_reordering_with_same_members(self) -> None:
        bundle = load_authority_bundle(repo_root())

        validate_case_meta(
            bundle,
            {
                "schema_version": "1.0.0",
                "case_id": "phase0_r1_core_57_28_1000_internal_generic_v1",
                "case_role": "R1-core",
                "ladder_position": 2,
                "phase_gates": ["Phase 8", "Phase 5", "Phase 2", "Phase 0"],
            },
        )

        validate_case_meta(
            bundle,
            {
                "schema_version": "1.0.0",
                "case_id": "phase0_r1_57_28_1000_internal_v1",
                "case_role": "R1",
                "ladder_position": 3,
                "phase_gates": ["Phase 0", "Phase 2", "Phase 6", "Phase 7", "Phase 8"],
            },
        )

        with self.assertRaisesRegex(
            AuthoritySelectionError,
            "case_meta.json phase_gates must be a list of phase-gate names",
        ):
            validate_case_meta(
                bundle,
                {
                    "schema_version": "1.0.0",
                    "case_id": "phase0_r1_core_57_28_1000_internal_generic_v1",
                    "case_role": "R1-core",
                    "ladder_position": 2,
                    "phase_gates": {"Phase 0": True},
                },
            )

    def test_stage_plan_schema_and_validation_enforce_phase_gate_selection(self) -> None:
        bundle = load_authority_bundle(repo_root())
        schema = stage_plan_schema(bundle)

        self.assertEqual(schema["canonical_name"], "stage_plan.json")
        self.assertEqual(schema["type"], "object")
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
            schema["properties"]["stages"]["items"]["properties"]["cmd"]["minLength"],
            1,
        )
        self.assertEqual(schema["properties"]["stages"]["minItems"], 1)
        self.assertIn("allOf", schema["properties"]["phase_gate_selection"])

        validate_stage_plan(
            bundle,
            {
                "schema_version": "1.0.0",
                "case_id": "phase0_r1_core_57_28_1000_internal_generic_v1",
                "case_role": "R1-core",
                "phase_gate": "Phase 5",
                "phase_gate_selection": {
                    "selected_case_role": "R1-core",
                    "available_case_roles": ["R2", "R1-core"],
                    "ordered_ladder": ["R2", "R1-core", "R1", "R0"],
                    "conditional_selection": False,
                },
                "stages": [
                    {
                        "name": "transient_run",
                        "cmd": "foamRun -solver incompressibleVoF",
                    }
                ],
            },
        )

    def test_stage_plan_validation_rejects_phase_local_ladder_rewrites(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with self.assertRaisesRegex(
            AuthoritySelectionError,
            "stage_plan.json ordered_ladder must remain R2 -> R1-core -> R1 -> R0",
        ):
            validate_stage_plan(
                bundle,
                {
                    "schema_version": "1.0.0",
                    "case_id": "phase0_r1_core_57_28_1000_internal_generic_v1",
                    "case_role": "R1-core",
                    "phase_gate": "Phase 5",
                    "phase_gate_selection": {
                        "selected_case_role": "R1-core",
                        "available_case_roles": ["R2", "R1-core"],
                        "ordered_ladder": ["R2", "R1", "R1-core", "R0"],
                        "conditional_selection": False,
                    },
                    "stages": [{"name": "transient_run", "cmd": "foamRun"}],
                },
            )

    def test_stage_plan_validation_rejects_non_canonical_types(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with self.assertRaisesRegex(
            AuthoritySelectionError,
            "stage_plan.json available_case_roles must be a list of case roles",
        ):
            validate_stage_plan(
                bundle,
                {
                    "schema_version": "1.0.0",
                    "case_id": "phase0_r1_core_57_28_1000_internal_generic_v1",
                    "case_role": "R1-core",
                    "phase_gate": "Phase 5",
                    "phase_gate_selection": {
                        "selected_case_role": "R1-core",
                        "available_case_roles": {"R2": True, "R1-core": True},
                        "ordered_ladder": ["R2", "R1-core", "R1", "R0"],
                        "conditional_selection": False,
                    },
                    "stages": [{"name": "transient_run", "cmd": "foamRun"}],
                },
            )

        with self.assertRaisesRegex(
            AuthoritySelectionError,
            "stage_plan.json ordered_ladder must be a list of case roles",
        ):
            validate_stage_plan(
                bundle,
                {
                    "schema_version": "1.0.0",
                    "case_id": "phase0_r1_core_57_28_1000_internal_generic_v1",
                    "case_role": "R1-core",
                    "phase_gate": "Phase 5",
                    "phase_gate_selection": {
                        "selected_case_role": "R1-core",
                        "available_case_roles": ["R2", "R1-core"],
                        "ordered_ladder": {"R2": True, "R1-core": True},
                        "conditional_selection": False,
                    },
                    "stages": [{"name": "transient_run", "cmd": "foamRun"}],
                },
            )

        with self.assertRaisesRegex(
            AuthoritySelectionError,
            "stage_plan.json each stage must define string name and cmd values",
        ):
            validate_stage_plan(
                bundle,
                {
                    "schema_version": "1.0.0",
                    "case_id": "phase0_r1_core_57_28_1000_internal_generic_v1",
                    "case_role": "R1-core",
                    "phase_gate": "Phase 5",
                    "phase_gate_selection": {
                        "selected_case_role": "R1-core",
                        "available_case_roles": ["R2", "R1-core"],
                        "ordered_ladder": ["R2", "R1-core", "R1", "R0"],
                        "conditional_selection": False,
                    },
                    "stages": [{"name": 1, "cmd": 2}],
                },
            )

        with self.assertRaisesRegex(
            AuthoritySelectionError,
            "stage_plan.json stage cwd must be a string when provided",
        ):
            validate_stage_plan(
                bundle,
                {
                    "schema_version": "1.0.0",
                    "case_id": "phase0_r1_core_57_28_1000_internal_generic_v1",
                    "case_role": "R1-core",
                    "phase_gate": "Phase 5",
                    "phase_gate_selection": {
                        "selected_case_role": "R1-core",
                        "available_case_roles": ["R1-core", "R2"],
                        "ordered_ladder": ["R2", "R1-core", "R1", "R0"],
                        "conditional_selection": False,
                    },
                    "stages": [{"name": "transient_run", "cmd": "foamRun", "cwd": 1}],
                },
            )

    def test_stage_plan_validation_allows_available_role_reordering_with_same_members(self) -> None:
        bundle = load_authority_bundle(repo_root())

        validate_stage_plan(
            bundle,
            {
                "schema_version": "1.0.0",
                "case_id": "phase0_r1_core_57_28_1000_internal_generic_v1",
                "case_role": "R1-core",
                "phase_gate": "Phase 5",
                "phase_gate_selection": {
                    "selected_case_role": "R1-core",
                    "available_case_roles": ["R1-core", "R2"],
                    "ordered_ladder": ["R2", "R1-core", "R1", "R0"],
                    "conditional_selection": False,
                },
                "stages": [{"name": "transient_run", "cmd": "foamRun"}],
            },
        )

    def test_stage_plan_validation_allows_documented_conditional_phase2_selection(self) -> None:
        bundle = load_authority_bundle(repo_root())

        validate_stage_plan(
            bundle,
            {
                "schema_version": "1.0.0",
                "case_id": "phase0_r1_57_28_1000_internal_v1",
                "case_role": "R1",
                "phase_gate": "Phase 2",
                "phase_gate_selection": {
                    "selected_case_role": "R1",
                    "available_case_roles": ["R2", "R1-core", "R1"],
                    "ordered_ladder": ["R2", "R1-core", "R1", "R0"],
                    "conditional_selection": True,
                    "conditional_reason": "patch-manifest coverage under test",
                },
                "stages": [{"name": "transient_run", "cmd": "foamRun"}],
            },
        )

    def test_stage_plan_validation_requires_explicit_opt_in_for_conditional_phase2_selection(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with self.assertRaisesRegex(
            AuthoritySelectionError,
            "stage_plan.json conditional_selection is only valid for authority-conditional case roles",
        ):
            validate_stage_plan(
                bundle,
                {
                    "schema_version": "1.0.0",
                    "case_id": "phase0_r1_core_57_28_1000_internal_generic_v1",
                    "case_role": "R1-core",
                    "phase_gate": "Phase 5",
                    "phase_gate_selection": {
                        "selected_case_role": "R1-core",
                        "available_case_roles": ["R2", "R1-core"],
                        "ordered_ladder": ["R2", "R1-core", "R1", "R0"],
                        "conditional_selection": True,
                        "conditional_reason": "not actually conditional",
                    },
                    "stages": [{"name": "transient_run", "cmd": "foamRun"}],
                },
            )

        with self.assertRaisesRegex(
            AuthoritySelectionError,
            "phase gate 'Phase 2' allows case role 'R1' only conditionally",
        ):
            validate_stage_plan(
                bundle,
                {
                    "schema_version": "1.0.0",
                    "case_id": "phase0_r1_57_28_1000_internal_v1",
                    "case_role": "R1",
                    "phase_gate": "Phase 2",
                    "phase_gate_selection": {
                        "selected_case_role": "R1",
                        "available_case_roles": ["R2", "R1-core"],
                        "ordered_ladder": ["R2", "R1-core", "R1", "R0"],
                        "conditional_selection": False,
                    },
                    "stages": [{"name": "transient_run", "cmd": "foamRun"}],
                },
            )

    def test_stage_plan_validation_requires_conditional_selection_field(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with self.assertRaisesRegex(
            AuthoritySelectionError,
            "stage_plan.json phase_gate_selection is missing required fields: conditional_selection",
        ):
            validate_stage_plan(
                bundle,
                {
                    "schema_version": "1.0.0",
                    "case_id": "phase0_r1_core_57_28_1000_internal_generic_v1",
                    "case_role": "R1-core",
                    "phase_gate": "Phase 5",
                    "phase_gate_selection": {
                        "selected_case_role": "R1-core",
                        "available_case_roles": ["R2", "R1-core"],
                        "ordered_ladder": ["R2", "R1-core", "R1", "R0"],
                    },
                    "stages": [{"name": "transient_run", "cmd": "foamRun"}],
                },
            )


if __name__ == "__main__":
    unittest.main()
