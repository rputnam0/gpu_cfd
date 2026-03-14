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

    def test_downstream_callers_resolve_every_authority_category_from_one_api(self) -> None:
        bundle = load_authority_bundle(repo_root())

        self.assertEqual(bundle.reference_case("R0").frozen_id, "phase0_r0_57_28_1000_full360_v1")
        self.assertEqual(bundle.graph.stage("nozzle_bc_update").fallback_mode, "async_no_graph")
        self.assertEqual(
            bundle.acceptance.tuples_by_id["P5_R1CORE_AMGX_ASYNC_BASELINE"].required_stage_ids[-2:],
            ("pressure_solve_amgx", "pressure_post"),
        )
        self.assertEqual(bundle.semantic_source_map.owner_for("Pressure bridge"), "PressureMatrixCache")

    def test_cli_validation_path_reports_success(self) -> None:
        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = main(["--root", str(repo_root())])

        self.assertEqual(exit_code, 0)
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
