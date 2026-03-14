from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

from scripts.authority import (
    AuthorityConflictError,
    AuthoritySchemaError,
    load_authority_bundle,
    main,
)


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


class AuthorityBundleTests(unittest.TestCase):
    def test_loads_complete_authority_bundle(self) -> None:
        bundle = load_authority_bundle(repo_root())

        self.assertEqual(bundle.pins.primary_toolkit_lane, "CUDA 12.9.1")
        self.assertEqual(bundle.ladder.ordered_case_ids, ("R2", "R1-core", "R1", "R0"))
        self.assertIn("R1-core", bundle.cases.by_case_id)
        self.assertEqual(bundle.support.global_policy.default_fallback_policy, "failFast")
        self.assertIn("P8_R1_NATIVE_GRAPH_BASELINE", bundle.acceptance.tuples_by_id)
        self.assertIn("pressure_solve_native", bundle.graph.stages_by_id)
        self.assertIn("Pressure bridge", bundle.semantic_source_map.entries_by_surface)

    def test_missing_required_file_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            self._copy_tree(repo_root(), temp_root)
            (temp_root / "docs" / "authority" / "support_matrix.json").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "support_matrix.json"):
                load_authority_bundle(temp_root)

    def test_unknown_schema_version_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            self._copy_tree(repo_root(), temp_root)
            json_path = temp_root / "docs" / "authority" / "reference_case_contract.json"
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            payload["schema_version"] = "9.9.9"
            json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(
                AuthoritySchemaError,
                "reference_case_contract.json.*9.9.9",
            ):
                load_authority_bundle(temp_root)

    def test_missing_schema_version_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            self._copy_tree(repo_root(), temp_root)
            json_path = temp_root / "docs" / "authority" / "reference_case_contract.json"
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            payload.pop("schema_version", None)
            json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(
                AuthoritySchemaError,
                "reference_case_contract.json is missing schema_version",
            ):
                load_authority_bundle(temp_root)

    def test_mismatched_authority_markdown_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            self._copy_tree(repo_root(), temp_root)
            json_path = temp_root / "docs" / "authority" / "support_matrix.json"
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            payload["authority_markdown"] = "reference_case_contract.md"
            json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(
                AuthoritySchemaError,
                "support_matrix.json.*support_matrix.md",
            ):
                load_authority_bundle(temp_root)

    def test_conflicting_authority_value_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            self._copy_tree(repo_root(), temp_root)
            json_path = temp_root / "docs" / "authority" / "reference_case_contract.json"
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            payload["locked_defaults"]["r1_core_required_case_id"] = "R1"
            json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(
                AuthorityConflictError,
                "r1_core_required_case_id.*R1-core",
            ):
                load_authority_bundle(temp_root)

    def test_duplicate_authority_ids_fail_fast(self) -> None:
        duplicate_cases = {
            "path": pathlib.Path("docs/authority/reference_case_contract.json"),
            "array_key": "frozen_cases",
            "duplicate_index": 0,
            "label": "duplicate case_id",
        }
        duplicate_tuples = {
            "path": pathlib.Path("docs/authority/acceptance_manifest.json"),
            "array_key": "accepted_tuples",
            "duplicate_index": 0,
            "label": "duplicate tuple_id",
        }
        duplicate_stages = {
            "path": pathlib.Path("docs/authority/graph_capture_support_matrix.json"),
            "array_key": "stages",
            "duplicate_index": 0,
            "label": "duplicate stage_id",
        }

        for scenario in (duplicate_cases, duplicate_tuples, duplicate_stages):
            with self.subTest(label=scenario["label"]):
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_root = pathlib.Path(temp_dir)
                    self._copy_tree(repo_root(), temp_root)
                    json_path = temp_root / scenario["path"]
                    payload = json.loads(json_path.read_text(encoding="utf-8"))
                    payload[scenario["array_key"]].append(
                        dict(payload[scenario["array_key"]][scenario["duplicate_index"]])
                    )
                    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

                    with self.assertRaisesRegex(AuthorityConflictError, scenario["label"]):
                        load_authority_bundle(temp_root)

    def test_acceptance_manifest_companion_reference_drift_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            self._copy_tree(repo_root(), temp_root)
            json_path = temp_root / "docs" / "authority" / "acceptance_manifest.json"
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            payload["reference_case_contract"] = "support_matrix.json"
            json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(
                AuthoritySchemaError,
                "acceptance_manifest.json.*reference_case_contract.json",
            ):
                load_authority_bundle(temp_root)

    def test_phase_gate_mapping_unknown_case_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            self._copy_tree(repo_root(), temp_root)
            json_path = temp_root / "docs" / "authority" / "reference_case_contract.json"
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            payload["phase_gate_mapping"]["Phase 6"]["accepted_case"] = "R9"
            json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(
                AuthorityConflictError,
                "phase_gate_mapping references unknown case ids: R9",
            ):
                load_authority_bundle(temp_root)

    def test_accepted_tuple_unknown_execution_mode_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            self._copy_tree(repo_root(), temp_root)
            json_path = temp_root / "docs" / "authority" / "acceptance_manifest.json"
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            payload["accepted_tuples"][0]["execution_mode"] = "graphFixd"
            json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(
                AuthorityConflictError,
                "acceptance_manifest.json references unknown execution modes: graphFixd",
            ):
                load_authority_bundle(temp_root)

    def test_duplicate_markdown_authority_key_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            self._copy_tree(repo_root(), temp_root)
            markdown_path = temp_root / "docs" / "authority" / "master_pin_manifest.md"
            text = markdown_path.read_text(encoding="utf-8")
            duplicate_row = (
                "| Primary toolkit lane | CUDA 99.0 | Synthetic duplicate for test. |\n"
            )
            marker = "| GPU target | `NVARCH=120`, native `sm_120` plus PTX | PTX/JIT validation remains mandatory. |\n"
            markdown_path.write_text(
                text.replace(marker, marker + duplicate_row),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                AuthorityConflictError,
                "duplicate markdown key 'Primary toolkit lane' in section: Frozen Defaults",
            ):
                load_authority_bundle(temp_root)

    def test_duplicate_semantic_source_surface_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            self._copy_tree(repo_root(), temp_root)
            markdown_path = temp_root / "docs" / "authority" / "semantic_source_map.md"
            text = markdown_path.read_text(encoding="utf-8")
            duplicate_row = (
                "| Pressure bridge | duplicate.cc | `PressureMatrixCache` | Synthetic duplicate for test. |\n"
            )
            marker = (
                "| Pressure bridge | linear-solver boundary and matrix staging | `PressureMatrixCache`, "
                "`packDeviceStaging(...)`, `DeviceDirect`, existing `foamExternalSolvers` AmgX bridge | "
                "`PinnedHost` is correctness-only. |\n"
            )
            markdown_path.write_text(
                text.replace(marker, marker + duplicate_row),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                AuthorityConflictError,
                "duplicate contract surface 'Pressure bridge' in semantic_source_map.md",
            ):
                load_authority_bundle(temp_root)

    def test_downstream_callers_resolve_every_authority_category_from_one_api(self) -> None:
        bundle = load_authority_bundle(repo_root())

        self.assertEqual(bundle.reference_case("R0").frozen_id, "phase0_r0_57_28_1000_full360_v1")
        self.assertEqual(bundle.graph.stage("nozzle_bc_update").fallback_mode, "async_no_graph")
        self.assertEqual(
            bundle.acceptance.tuples_by_id["P5_R1CORE_AMGX_ASYNC_BASELINE"].required_stage_ids[-2:],
            ("pressure_solve_amgx", "pressure_post"),
        )
        self.assertEqual(bundle.semantic_source_map.owner_for("Alpha transport"), "alphaPredictor.C")
        self.assertEqual(bundle.semantic_source_map.owner_for("Pressure bridge"), "PressureMatrixCache")
        self.assertEqual(
            bundle.semantic_source_map.owner_for("Pressure corrector"),
            "pressureCorrector.C",
        )

    def test_cli_validation_path_reports_text_summary_by_default(self) -> None:
        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = main(["--root", str(repo_root())])

        self.assertEqual(exit_code, 0)
        self.assertIn("Authority bundle load report", stdout.getvalue())
        self.assertIn("reference_case_contract.json", stdout.getvalue())
        self.assertNotIn('"loaded_json"', stdout.getvalue())

    def test_cli_validation_path_reports_json_when_requested(self) -> None:
        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = main(["--root", str(repo_root()), "--json"])

        self.assertEqual(exit_code, 0)
        self.assertIn('"loaded_json"', stdout.getvalue())
        self.assertIn("reference_case_contract.json", stdout.getvalue())

    def _copy_tree(self, source: pathlib.Path, destination: pathlib.Path) -> None:
        for path in source.rglob("*"):
            relative = path.relative_to(source)
            target = destination / relative
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())


if __name__ == "__main__":
    unittest.main()
