from __future__ import annotations

import pathlib
import unittest

from scripts.authority import load_authority_bundle
from scripts.authority.acceptance import (
    AcceptanceClassResult,
    AcceptanceEvaluationContext,
    AcceptanceWaiver,
    evaluate_acceptance,
    resolve_accepted_tuple,
)


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def passing_hard_gates() -> dict[str, int | bool]:
    return {
        "unexpected_htod_bytes": 0,
        "unexpected_dtoh_bytes": 0,
        "cpu_um_faults": 0,
        "gpu_um_faults": 0,
        "cudaDeviceSynchronize_calls": 0,
        "post_warmup_alloc_calls": 0,
        "mandatory_nvtx_ranges_present": True,
        "cpu_boundary_fallback_events": 0,
        "host_patch_execution_events": 0,
        "pinned_host_pressure_stage_events": 0,
        "host_setFields_startup_events": 0,
        "unsafe_functionObject_commit_events": 0,
    }


def passing_soft_gates() -> dict[str, float]:
    return {
        "graph_launches_per_step": 2,
        "top_kernel_time_regression_pct": 5.0,
    }


class MatchingWaiverHook:
    def resolve_soft_gate_waiver(
        self,
        *,
        manifest_revision: str,
        tuple_id: str,
        failed_soft_gate_ids: tuple[str, ...],
    ) -> AcceptanceWaiver | None:
        if failed_soft_gate_ids != ("graph_launches_per_step",):
            return None
        return AcceptanceWaiver(
            manifest_revision=manifest_revision,
            tuple_id=tuple_id,
            reason="Locked exception for benchmark capture noise",
        )


class MismatchedWaiverHook:
    def resolve_soft_gate_waiver(
        self,
        *,
        manifest_revision: str,
        tuple_id: str,
        failed_soft_gate_ids: tuple[str, ...],
    ) -> AcceptanceWaiver | None:
        return AcceptanceWaiver(
            manifest_revision="wrong-revision",
            tuple_id="P8_R0_NATIVE_GRAPH_BASELINE",
            reason="Does not bind to this tuple or revision",
        )


class AcceptanceEvaluatorTests(unittest.TestCase):
    def test_resolves_required_tuple_and_emits_deterministic_pass_verdict(self) -> None:
        bundle = load_authority_bundle(repo_root())

        accepted_tuple = resolve_accepted_tuple(bundle, "P5_R2_TRANSPORT_NATIVE_ASYNC_BASELINE")
        verdict = evaluate_acceptance(
            bundle,
            tuple_id=accepted_tuple.tuple_id,
            hard_gate_observations=passing_hard_gates(),
            soft_gate_observations=passing_soft_gates(),
            class_results={
                "TC_R2_TRANSPORT": AcceptanceClassResult(
                    class_id="TC_R2_TRANSPORT",
                    passed=True,
                    details="Transport slice stayed within the frozen tolerance envelope.",
                )
            },
        )

        self.assertEqual(accepted_tuple.case_id, "R2")
        self.assertEqual(verdict.disposition, "pass")
        self.assertEqual(verdict.reason, "All hard gates, soft gates, and active threshold/parity classes passed.")
        self.assertEqual(verdict.thresholds_used["tolerance_class"]["class_id"], "TC_R2_TRANSPORT")
        self.assertEqual(
            verdict.required_orchestration_ranges,
            ("solver/timeStep", "solver/steady_state", "solver/pimpleOuter"),
        )
        self.assertEqual(
            verdict.required_stage_ids,
            (
                "pre_solve",
                "outer_iter_body",
                "alpha_pre",
                "alpha_subcycle_body",
                "mixture_update",
                "momentum_predictor",
                "pressure_assembly",
                "pressure_solve_native",
                "pressure_post",
            ),
        )

    def test_unknown_tuple_id_returns_non_admitted_verdict(self) -> None:
        bundle = load_authority_bundle(repo_root())

        verdict = evaluate_acceptance(
            bundle,
            tuple_id="NOT_A_REAL_TUPLE",
            hard_gate_observations=passing_hard_gates(),
            soft_gate_observations=passing_soft_gates(),
            class_results={},
        )

        self.assertEqual(verdict.disposition, "non_admitted")
        self.assertIn("NOT_A_REAL_TUPLE", verdict.reason)
        self.assertFalse(verdict.release_eligible)
        self.assertFalse(verdict.baseline_lock_eligible)

    def test_hard_gate_failure_forces_fail_even_when_soft_gates_pass(self) -> None:
        bundle = load_authority_bundle(repo_root())
        hard_gates = passing_hard_gates()
        hard_gates["unexpected_htod_bytes"] = 1

        verdict = evaluate_acceptance(
            bundle,
            tuple_id="P5_R2_TRANSPORT_NATIVE_ASYNC_BASELINE",
            hard_gate_observations=hard_gates,
            soft_gate_observations=passing_soft_gates(),
            class_results={
                "TC_R2_TRANSPORT": AcceptanceClassResult(
                    class_id="TC_R2_TRANSPORT",
                    passed=True,
                )
            },
        )

        self.assertEqual(verdict.disposition, "fail")
        self.assertIn("Hard gate failures", verdict.reason)
        self.assertEqual(verdict.gate_results["soft"]["graph_launches_per_step"]["passed"], True)

    def test_soft_gate_waiver_must_match_manifest_revision_and_tuple_binding(self) -> None:
        bundle = load_authority_bundle(repo_root())
        soft_gates = passing_soft_gates()
        soft_gates["graph_launches_per_step"] = 5
        class_results = {
            "TC_R1CORE_GENERIC": AcceptanceClassResult(
                class_id="TC_R1CORE_GENERIC",
                passed=True,
            ),
            "RP_STRICT": AcceptanceClassResult(
                class_id="RP_STRICT",
                passed=True,
            ),
        }

        mismatched = evaluate_acceptance(
            bundle,
            tuple_id="P5_R1CORE_NATIVE_ASYNC_BASELINE",
            hard_gate_observations=passing_hard_gates(),
            soft_gate_observations=soft_gates,
            class_results=class_results,
            waiver_hook=MismatchedWaiverHook(),
        )
        matched = evaluate_acceptance(
            bundle,
            tuple_id="P5_R1CORE_NATIVE_ASYNC_BASELINE",
            hard_gate_observations=passing_hard_gates(),
            soft_gate_observations=soft_gates,
            class_results=class_results,
            waiver_hook=MatchingWaiverHook(),
        )

        self.assertEqual(mismatched.disposition, "soft_fail")
        self.assertIsNone(mismatched.waiver)
        self.assertEqual(matched.disposition, "pass")
        self.assertIsNotNone(matched.waiver)
        assert matched.waiver is not None
        self.assertEqual(
            matched.waiver["manifest_revision"],
            bundle.authority_revisions["acceptance_manifest"]["sha256"],
        )

    def test_scoped_hard_gates_are_skipped_when_context_marks_them_out_of_scope(self) -> None:
        bundle = load_authority_bundle(repo_root())
        hard_gates = passing_hard_gates()
        hard_gates["pinned_host_pressure_stage_events"] = 3
        hard_gates["host_setFields_startup_events"] = 2

        verdict = evaluate_acceptance(
            bundle,
            tuple_id="P5_R2_TRANSPORT_NATIVE_ASYNC_BASELINE",
            hard_gate_observations=hard_gates,
            soft_gate_observations=passing_soft_gates(),
            class_results={
                "TC_R2_TRANSPORT": AcceptanceClassResult(
                    class_id="TC_R2_TRANSPORT",
                    passed=True,
                )
            },
            evaluation_context=AcceptanceEvaluationContext(
                is_production_acceptance_run=False,
                uses_accepted_startup_path=False,
            ),
        )

        self.assertEqual(verdict.disposition, "pass")
        self.assertEqual(
            verdict.gate_results["hard"]["pinned_host_pressure_stage_events"]["applicable"],
            False,
        )
        self.assertEqual(
            verdict.gate_results["hard"]["host_setFields_startup_events"]["applicable"],
            False,
        )

    def test_amgx_tuple_requires_device_direct_pressure_bridge_for_admission(self) -> None:
        bundle = load_authority_bundle(repo_root())

        verdict = evaluate_acceptance(
            bundle,
            tuple_id="P5_R1CORE_AMGX_ASYNC_BASELINE",
            hard_gate_observations=passing_hard_gates(),
            soft_gate_observations=passing_soft_gates(),
            class_results={
                "TC_R1CORE_GENERIC": AcceptanceClassResult(
                    class_id="TC_R1CORE_GENERIC",
                    passed=True,
                ),
                "RP_STRICT": AcceptanceClassResult(
                    class_id="RP_STRICT",
                    passed=True,
                ),
                "BP_AMGX_R1CORE": AcceptanceClassResult(
                    class_id="BP_AMGX_R1CORE",
                    passed=True,
                ),
            },
            evaluation_context=AcceptanceEvaluationContext(
                pressure_bridge_mode="PinnedHost",
            ),
        )

        self.assertEqual(verdict.disposition, "non_admitted")
        self.assertIn("DeviceDirect", verdict.reason)
        self.assertFalse(verdict.admitted)

    def test_mismatched_class_result_id_cannot_satisfy_required_class(self) -> None:
        bundle = load_authority_bundle(repo_root())

        verdict = evaluate_acceptance(
            bundle,
            tuple_id="P5_R2_TRANSPORT_NATIVE_ASYNC_BASELINE",
            hard_gate_observations=passing_hard_gates(),
            soft_gate_observations=passing_soft_gates(),
            class_results={
                "TC_R2_TRANSPORT": AcceptanceClassResult(
                    class_id="TC_R2_SURFACE",
                    passed=True,
                )
            },
        )

        self.assertEqual(verdict.disposition, "fail")
        self.assertIn("Threshold/parity class failures", verdict.reason)
        self.assertIn(
            "does not match required class",
            verdict.class_results["TC_R2_TRANSPORT"]["details"],
        )


if __name__ == "__main__":
    unittest.main()
