"""Acceptance-manifest evaluator scaffold for formal tuple verdicts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .bundle import (
    AcceptedTuple,
    AuthorityBundle,
    AuthorityConflictError,
)


VERDICT_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class AcceptanceClassResult:
    class_id: str
    passed: bool
    details: str | None = None
    observed: dict[str, Any] | None = None


@dataclass(frozen=True)
class AcceptanceWaiver:
    manifest_revision: str
    tuple_id: str
    reason: str
    approved_by: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class AcceptanceEvaluationContext:
    is_production_acceptance_run: bool = False
    uses_accepted_startup_path: bool = True
    in_timed_steady_state_window: bool = True
    in_steady_state_inner_ranges: bool = True
    pressure_bridge_mode: str | None = None


class AcceptanceWaiverHook(Protocol):
    def resolve_soft_gate_waiver(
        self,
        *,
        manifest_revision: str,
        tuple_id: str,
        failed_soft_gate_ids: tuple[str, ...],
    ) -> AcceptanceWaiver | None: ...


@dataclass(frozen=True)
class AcceptedTupleResolution:
    tuple_id: str
    phase_gate: str
    case_id: str
    case_variant: str
    backend: str
    execution_mode: str
    kernel_family_mode: str
    admission: str
    production_eligible: bool
    tolerance_class: str
    parity_class_ids: tuple[str, ...]
    required_orchestration_ranges: tuple[str, ...]
    required_stage_ids: tuple[str, ...]
    production_defaults: dict[str, Any]
    thresholds_used: dict[str, Any]
    manifest_revision: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class AcceptanceVerdict:
    schema_version: str
    manifest_revision: str
    tuple_id: str
    admitted: bool
    admission: str | None
    disposition: str
    reason: str
    production_eligible: bool
    release_eligible: bool
    baseline_lock_eligible: bool
    required_orchestration_ranges: tuple[str, ...]
    required_stage_ids: tuple[str, ...]
    production_defaults: dict[str, Any]
    gate_results: dict[str, dict[str, dict[str, Any]]]
    thresholds_used: dict[str, Any]
    class_results: dict[str, dict[str, Any]]
    waiver: dict[str, Any] | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_accepted_tuple(bundle: AuthorityBundle, tuple_id: str) -> AcceptedTupleResolution:
    """Resolve one accepted tuple ID into the authority-owned evaluation contract."""

    accepted_tuple = bundle.acceptance.tuples_by_id.get(tuple_id)
    if accepted_tuple is None:
        raise ValueError(f"unknown accepted tuple id: {tuple_id}")

    threshold_classes = bundle.acceptance.raw["threshold_classes"]
    tolerance_class = str(accepted_tuple.raw["tolerance_class"])
    field_qoi_thresholds = threshold_classes["field_qoi"]
    if tolerance_class not in field_qoi_thresholds:
        raise AuthorityConflictError(
            f"acceptance_manifest.json references unknown tolerance class {tolerance_class!r}"
        )

    parity_thresholds = threshold_classes["parity_replay"]
    parity_class_ids = _active_parity_class_ids(accepted_tuple)
    unknown_parity = sorted(class_id for class_id in parity_class_ids if class_id not in parity_thresholds)
    if unknown_parity:
        raise AuthorityConflictError(
            "acceptance_manifest.json references unknown parity classes: "
            + ", ".join(unknown_parity)
        )

    thresholds_used = {
        "tolerance_class": {
            "class_id": tolerance_class,
            "category": "field_qoi",
            "thresholds": dict(field_qoi_thresholds[tolerance_class]),
        },
        "parity_classes": [
            {
                "class_id": class_id,
                "category": "parity_replay",
                "thresholds": dict(parity_thresholds[class_id]),
            }
            for class_id in parity_class_ids
        ],
    }
    return AcceptedTupleResolution(
        tuple_id=accepted_tuple.tuple_id,
        phase_gate=str(accepted_tuple.raw["phase_gate"]),
        case_id=accepted_tuple.case_id,
        case_variant=str(accepted_tuple.raw["case_variant"]),
        backend=accepted_tuple.backend,
        execution_mode=accepted_tuple.execution_mode,
        kernel_family_mode=str(accepted_tuple.raw["kernel_family_mode"]),
        admission=str(accepted_tuple.raw["admission"]),
        production_eligible=bool(accepted_tuple.raw["production_eligible"]),
        tolerance_class=tolerance_class,
        parity_class_ids=parity_class_ids,
        required_orchestration_ranges=tuple(
            str(item)
            for item in bundle.acceptance.raw["nvtx_contract_defaults"]["required_orchestration_ranges"]
        ),
        required_stage_ids=accepted_tuple.required_stage_ids,
        production_defaults=dict(bundle.acceptance.raw["production_defaults"]),
        thresholds_used=thresholds_used,
        manifest_revision=bundle.authority_revisions["acceptance_manifest"]["sha256"],
        raw=dict(accepted_tuple.raw),
    )


def evaluate_acceptance(
    bundle: AuthorityBundle,
    *,
    tuple_id: str,
    hard_gate_observations: dict[str, Any],
    soft_gate_observations: dict[str, Any],
    class_results: dict[str, AcceptanceClassResult],
    evaluation_context: AcceptanceEvaluationContext | None = None,
    waiver_hook: AcceptanceWaiverHook | None = None,
) -> AcceptanceVerdict:
    """Evaluate one tuple deterministically against hard gates, soft gates, and class results."""

    manifest_revision = bundle.authority_revisions["acceptance_manifest"]["sha256"]
    context = evaluation_context or AcceptanceEvaluationContext()
    try:
        resolved = resolve_accepted_tuple(bundle, tuple_id)
    except ValueError:
        return AcceptanceVerdict(
            schema_version=VERDICT_SCHEMA_VERSION,
            manifest_revision=manifest_revision,
            tuple_id=tuple_id,
            admitted=False,
            admission=None,
            disposition="non_admitted",
            reason=f"Tuple ID {tuple_id!r} is not admitted by acceptance_manifest.json.",
            production_eligible=False,
            release_eligible=False,
            baseline_lock_eligible=False,
            required_orchestration_ranges=(),
            required_stage_ids=(),
            production_defaults=dict(bundle.acceptance.raw["production_defaults"]),
            gate_results={"hard": {}, "soft": {}},
            thresholds_used={},
            class_results={},
            waiver=None,
        )

    required_pressure_bridge_mode = resolved.raw.get("required_pressure_bridge_mode")
    if required_pressure_bridge_mode and context.pressure_bridge_mode != required_pressure_bridge_mode:
        return AcceptanceVerdict(
            schema_version=VERDICT_SCHEMA_VERSION,
            manifest_revision=resolved.manifest_revision,
            tuple_id=resolved.tuple_id,
            admitted=False,
            admission=resolved.admission,
            disposition="non_admitted",
            reason=(
                "Tuple requires pressure bridge mode "
                f"{required_pressure_bridge_mode!r}, observed "
                f"{context.pressure_bridge_mode!r}; run remains diagnostic-only outside the "
                "accepted tuple matrix."
            ),
            production_eligible=resolved.production_eligible,
            release_eligible=False,
            baseline_lock_eligible=False,
            required_orchestration_ranges=resolved.required_orchestration_ranges,
            required_stage_ids=resolved.required_stage_ids,
            production_defaults=resolved.production_defaults,
            gate_results={"hard": {}, "soft": {}},
            thresholds_used=resolved.thresholds_used,
            class_results={},
            waiver=None,
        )

    hard_results = _evaluate_gate_family(
        bundle.acceptance.raw["hard_gates"],
        hard_gate_observations,
        context,
    )
    soft_results = _evaluate_gate_family(
        bundle.acceptance.raw["soft_gates"],
        soft_gate_observations,
        context,
    )
    class_evaluation = _evaluate_active_classes(resolved, class_results)

    failed_hard = tuple(key for key, result in hard_results.items() if not bool(result["passed"]))
    failed_soft = tuple(key for key, result in soft_results.items() if not bool(result["passed"]))
    failed_classes = tuple(
        key for key, result in class_evaluation.items() if not bool(result["passed"])
    )

    waiver = _resolve_matching_waiver(
        waiver_hook,
        manifest_revision=resolved.manifest_revision,
        tuple_id=resolved.tuple_id,
        failed_soft_gate_ids=failed_soft,
    )
    soft_failures_waived = bool(failed_soft) and waiver is not None

    if failed_hard:
        disposition = "fail"
        reason = "Hard gate failures: " + ", ".join(failed_hard) + "."
    elif failed_classes:
        disposition = "fail"
        reason = "Threshold/parity class failures: " + ", ".join(failed_classes) + "."
    elif failed_soft and not soft_failures_waived:
        disposition = "soft_fail"
        reason = (
            "Soft gate failures require an exact waiver bound to the manifest revision and tuple ID: "
            + ", ".join(failed_soft)
            + "."
        )
    elif soft_failures_waived:
        disposition = "pass"
        reason = "Soft gate failures were covered by an exact waiver for this manifest revision and tuple."
    else:
        disposition = "pass"
        reason = "All hard gates, soft gates, and active threshold/parity classes passed."

    release_eligible = disposition == "pass" and resolved.production_eligible
    baseline_lock_eligible = disposition == "pass" and resolved.production_eligible
    return AcceptanceVerdict(
        schema_version=VERDICT_SCHEMA_VERSION,
        manifest_revision=resolved.manifest_revision,
        tuple_id=resolved.tuple_id,
        admitted=True,
        admission=resolved.admission,
        disposition=disposition,
        reason=reason,
        production_eligible=resolved.production_eligible,
        release_eligible=release_eligible,
        baseline_lock_eligible=baseline_lock_eligible,
        required_orchestration_ranges=resolved.required_orchestration_ranges,
        required_stage_ids=resolved.required_stage_ids,
        production_defaults=resolved.production_defaults,
        gate_results={
            "hard": hard_results,
            "soft": soft_results,
        },
        thresholds_used=resolved.thresholds_used,
        class_results=class_evaluation,
        waiver=asdict(waiver) if waiver is not None else None,
    )


def _active_parity_class_ids(accepted_tuple: AcceptedTuple) -> tuple[str, ...]:
    class_ids = []
    for key in (
        "restart_reload_parity_class",
        "execution_parity_class",
        "backend_parity_class",
        "kernel_parity_class",
    ):
        value = accepted_tuple.raw.get(key)
        if value:
            class_ids.append(str(value))
    return tuple(class_ids)


def _evaluate_gate_family(
    definitions: dict[str, dict[str, Any]],
    observations: dict[str, Any],
    context: AcceptanceEvaluationContext,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for gate_id, definition in definitions.items():
        applicable = _gate_scope_applies(str(definition.get("scope") or ""), context)
        observed = observations.get(gate_id)
        missing = gate_id not in observations
        if not applicable:
            passed = True
            missing = False
        elif missing:
            passed = False
        else:
            passed = _compare(observed, definition["operator"], definition["value"])
        results[gate_id] = {
            "observed": observed,
            "operator": definition["operator"],
            "expected": definition["value"],
            "scope": definition.get("scope"),
            "applicable": applicable,
            "skipped": not applicable,
            "passed": passed,
            "missing": missing,
        }
    return results


def _evaluate_active_classes(
    resolved: AcceptedTupleResolution,
    provided_results: dict[str, AcceptanceClassResult],
) -> dict[str, dict[str, Any]]:
    active_class_ids = (resolved.tolerance_class, *resolved.parity_class_ids)
    results: dict[str, dict[str, Any]] = {}
    for class_id in active_class_ids:
        provided = provided_results.get(class_id)
        if provided is None:
            results[class_id] = {
                "class_id": class_id,
                "passed": False,
                "details": "Missing required class evaluation result.",
                "observed": None,
            }
            continue
        if provided.class_id != class_id:
            results[class_id] = {
                "class_id": class_id,
                "passed": False,
                "details": (
                    f"Supplied class result {provided.class_id!r} does not match required class "
                    f"{class_id!r}."
                ),
                "observed": provided.observed,
            }
            continue
        results[class_id] = {
            "class_id": class_id,
            "passed": bool(provided.passed),
            "details": provided.details,
            "observed": provided.observed,
        }
    return results


def _resolve_matching_waiver(
    waiver_hook: AcceptanceWaiverHook | None,
    *,
    manifest_revision: str,
    tuple_id: str,
    failed_soft_gate_ids: tuple[str, ...],
) -> AcceptanceWaiver | None:
    if waiver_hook is None or not failed_soft_gate_ids:
        return None
    waiver = waiver_hook.resolve_soft_gate_waiver(
        manifest_revision=manifest_revision,
        tuple_id=tuple_id,
        failed_soft_gate_ids=failed_soft_gate_ids,
    )
    if waiver is None:
        return None
    if waiver.manifest_revision != manifest_revision:
        return None
    if waiver.tuple_id != tuple_id:
        return None
    return waiver


def _compare(observed: Any, operator: str, expected: Any) -> bool:
    if operator == "==":
        return observed == expected
    if operator == "<=":
        return observed <= expected
    raise AuthorityConflictError(f"unsupported acceptance gate operator {operator!r}")


def _gate_scope_applies(
    scope: str,
    context: AcceptanceEvaluationContext,
) -> bool:
    if not scope:
        return True
    if scope == "production_acceptance_runs":
        return context.is_production_acceptance_run
    if scope == "accepted_startup_path":
        return context.uses_accepted_startup_path
    if scope == "timed_steady_state_windows":
        return context.in_timed_steady_state_window
    if scope == "steady_state_inner_ranges":
        return context.in_steady_state_inner_ranges
    raise AuthorityConflictError(f"unsupported acceptance gate scope {scope!r}")
