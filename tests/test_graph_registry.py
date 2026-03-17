from __future__ import annotations

from dataclasses import replace
import json
import pathlib
import tempfile
import unittest

from scripts.authority import (
    GraphRegistryValidationError,
    build_graph_stage_registry,
    load_authority_bundle,
    load_graph_stage_registry,
    validate_acceptance_tuple_stage_requirements,
    validate_tuple_stage_requirements,
)
from scripts.authority.bundle import GraphCaptureMatrix, GraphStage


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


class GraphStageRegistryTests(unittest.TestCase):
    def test_loader_exposes_canonical_stage_and_run_mode_lookups(self) -> None:
        registry = load_graph_stage_registry(repo_root())

        self.assertEqual(
            registry.required_orchestration_ranges,
            ("solver/timeStep", "solver/steady_state", "solver/pimpleOuter"),
        )
        self.assertEqual(registry.stage("pressure_solve_amgx").capture_policy, "graph-external until individually validated")
        self.assertTrue(registry.run_mode("graph_fixed").production_accepted)
        self.assertFalse(registry.run_mode("sync_debug").production_accepted)
        self.assertEqual(registry.resolve_fallback_mode("pressure_solve_native"), "async_no_graph")

    def test_acceptance_tuple_stage_requirements_validate_against_canonical_registry(self) -> None:
        bundle = load_authority_bundle(repo_root())
        registry = build_graph_stage_registry(bundle)

        report = validate_acceptance_tuple_stage_requirements(bundle, registry=registry)

        self.assertEqual(report.validated_tuple_count, len(bundle.acceptance.tuples_by_id))
        self.assertIn("P8_R1_NATIVE_GRAPH_BASELINE", report.tuple_stage_ids)
        self.assertEqual(
            report.tuple_stage_ids["P8_R1_NATIVE_GRAPH_BASELINE"][-3:],
            ("pressure_solve_native", "pressure_post", "nozzle_bc_update"),
        )
        self.assertIn("nozzle_bc_update", report.stage_ids_in_use)

    def test_unknown_stage_ids_fail_validation(self) -> None:
        bundle = load_authority_bundle(repo_root())
        registry = build_graph_stage_registry(bundle)

        with self.assertRaisesRegex(
            GraphRegistryValidationError,
            "LOCAL_ALIAS references unknown canonical stage ids: pressureSolveNative",
        ):
            validate_tuple_stage_requirements(
                registry,
                {"LOCAL_ALIAS": ("pressureSolveNative",)},
            )

    def test_report_is_json_serializable_for_downstream_consumers(self) -> None:
        bundle = load_authority_bundle(repo_root())
        registry = build_graph_stage_registry(bundle)
        validation = validate_acceptance_tuple_stage_requirements(bundle, registry=registry)

        report = registry.emit_report(
            validation,
            authority_revisions=bundle.authority_revisions,
            accepted_tuples=bundle.acceptance.tuples_by_id,
        )

        payload = report.as_dict()
        self.assertEqual(payload["schema_version"], "1.0.0")
        self.assertEqual(payload["run_modes"]["async_no_graph"]["production_accepted"], True)
        self.assertEqual(
            payload["accepted_tuples"]["P8_R1_NATIVE_GRAPH_BASELINE"]["required_stage_ids"][-3:],
            ["pressure_solve_native", "pressure_post", "nozzle_bc_update"],
        )
        self.assertEqual(
            payload["accepted_tuples"]["P8_R1_NATIVE_GRAPH_BASELINE"]["execution_mode"],
            "graph_fixed",
        )
        self.assertIn("graph_capture_support_matrix", payload["authority_revisions"])
        json.dumps(payload)

    def test_unknown_requested_run_mode_fails_fast(self) -> None:
        registry = load_graph_stage_registry(repo_root())

        with self.assertRaisesRegex(
            GraphRegistryValidationError,
            "unknown run mode 'graphFixd'",
        ):
            registry.run_mode("graphFixd")

    def test_duplicate_run_modes_fail_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            self._copy_tree(repo_root(), temp_root)
            json_path = temp_root / "docs" / "authority" / "graph_capture_support_matrix.json"
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            payload["run_modes"].append(dict(payload["run_modes"][0]))
            json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(
                GraphRegistryValidationError,
                "duplicate run mode 'sync_debug'",
            ):
                load_graph_stage_registry(temp_root)

    def test_non_boolean_production_accepted_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            self._copy_tree(repo_root(), temp_root)
            json_path = temp_root / "docs" / "authority" / "graph_capture_support_matrix.json"
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            payload["run_modes"][0]["production_accepted"] = "false"
            json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(
                GraphRegistryValidationError,
                "run mode 'sync_debug' production_accepted must be a boolean",
            ):
                load_graph_stage_registry(temp_root)

    def test_unknown_stage_fallback_mode_fails_fast_during_registry_build(self) -> None:
        bundle = load_authority_bundle(repo_root())
        invalid_stage = GraphStage(
            stage_id="warmup",
            fallback_mode="graphFixd",
            raw={**bundle.graph.stages_by_id["warmup"].raw, "fallback_mode": "graphFixd"},
        )
        invalid_graph = GraphCaptureMatrix(
            run_modes=bundle.graph.run_modes,
            stages_by_id={**bundle.graph.stages_by_id, "warmup": invalid_stage},
            required_orchestration_ranges=bundle.graph.required_orchestration_ranges,
            raw={
                **bundle.graph.raw,
                "stages": [
                    {**stage, "fallback_mode": "graphFixd"}
                    if stage["stage_id"] == "warmup"
                    else dict(stage)
                    for stage in bundle.graph.raw["stages"]
                ],
            },
        )
        invalid_bundle = replace(bundle, graph=invalid_graph)

        with self.assertRaisesRegex(
            GraphRegistryValidationError,
            "stages reference unknown fallback run modes: warmup -> graphFixd",
        ):
            build_graph_stage_registry(invalid_bundle)

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
