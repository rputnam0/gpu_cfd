"""Canonical graph stage registry and tuple-stage validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .bundle import AcceptedTuple, AuthorityBundle, load_authority_bundle


STAGE_REGISTRY_SCHEMA_VERSION = "1.0.0"


class GraphRegistryValidationError(ValueError):
    """Raised when graph stage or run-mode resolution fails validation."""


@dataclass(frozen=True)
class GraphRunMode:
    run_mode: str
    production_accepted: bool
    description: str


@dataclass(frozen=True)
class CanonicalGraphStage:
    stage_id: str
    intended_phase: str
    capture_policy: str
    loop_owner: str
    fallback_mode: str
    notes: str


@dataclass(frozen=True)
class TupleStageValidationReport:
    validated_tuple_count: int
    tuple_stage_ids: dict[str, tuple[str, ...]]
    stage_ids_in_use: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "validated_tuple_count": self.validated_tuple_count,
            "tuple_stage_ids": {
                tuple_id: list(stage_ids)
                for tuple_id, stage_ids in self.tuple_stage_ids.items()
            },
            "stage_ids_in_use": list(self.stage_ids_in_use),
        }


@dataclass(frozen=True)
class StageRegistryReport:
    schema_version: str
    run_modes: dict[str, GraphRunMode]
    stages: dict[str, CanonicalGraphStage]
    required_orchestration_ranges: tuple[str, ...]
    global_capture_rules: dict[str, Any]
    accepted_tuples: dict[str, dict[str, Any]]
    authority_revisions: dict[str, dict[str, str]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_modes": {
                run_mode: {
                    "run_mode": policy.run_mode,
                    "production_accepted": policy.production_accepted,
                    "description": policy.description,
                }
                for run_mode, policy in self.run_modes.items()
            },
            "stages": {
                stage_id: {
                    "stage_id": stage.stage_id,
                    "intended_phase": stage.intended_phase,
                    "capture_policy": stage.capture_policy,
                    "loop_owner": stage.loop_owner,
                    "fallback_mode": stage.fallback_mode,
                    "notes": stage.notes,
                }
                for stage_id, stage in self.stages.items()
            },
            "required_orchestration_ranges": list(self.required_orchestration_ranges),
            "global_capture_rules": dict(self.global_capture_rules),
            "accepted_tuples": {
                tuple_id: dict(payload)
                for tuple_id, payload in self.accepted_tuples.items()
            },
            "authority_revisions": dict(self.authority_revisions),
        }


@dataclass(frozen=True)
class GraphStageRegistry:
    run_modes_by_id: dict[str, GraphRunMode]
    stages_by_id: dict[str, CanonicalGraphStage]
    required_orchestration_ranges: tuple[str, ...]
    global_capture_rules: dict[str, Any]

    def stage(self, stage_id: str) -> CanonicalGraphStage:
        try:
            return self.stages_by_id[stage_id]
        except KeyError as exc:
            raise GraphRegistryValidationError(
                f"unknown canonical stage id {stage_id!r}"
            ) from exc

    def run_mode(self, run_mode: str) -> GraphRunMode:
        try:
            return self.run_modes_by_id[run_mode]
        except KeyError as exc:
            raise GraphRegistryValidationError(f"unknown run mode {run_mode!r}") from exc

    def resolve_fallback_mode(self, stage_id: str) -> str:
        fallback_mode = self.stage(stage_id).fallback_mode
        self.run_mode(fallback_mode)
        return fallback_mode

    def emit_report(
        self,
        validation: TupleStageValidationReport,
        *,
        authority_revisions: dict[str, dict[str, str]] | None = None,
        accepted_tuples: Mapping[str, AcceptedTuple] | None = None,
    ) -> StageRegistryReport:
        accepted_payload = {
            tuple_id: {
                "required_stage_ids": list(stage_ids),
            }
            for tuple_id, stage_ids in validation.tuple_stage_ids.items()
        }
        if accepted_tuples is not None:
            for tuple_id, tuple_entry in accepted_tuples.items():
                if tuple_id not in accepted_payload:
                    continue
                accepted_payload[tuple_id].update(
                    {
                        "case_id": tuple_entry.case_id,
                        "backend": tuple_entry.backend,
                        "execution_mode": tuple_entry.execution_mode,
                    }
                )
        return StageRegistryReport(
            schema_version=STAGE_REGISTRY_SCHEMA_VERSION,
            run_modes=dict(self.run_modes_by_id),
            stages=dict(self.stages_by_id),
            required_orchestration_ranges=self.required_orchestration_ranges,
            global_capture_rules=dict(self.global_capture_rules),
            accepted_tuples=accepted_payload,
            authority_revisions=authority_revisions or {},
        )


def load_graph_stage_registry(root: Path | str | None = None) -> GraphStageRegistry:
    return build_graph_stage_registry(load_authority_bundle(root))


def build_graph_stage_registry(bundle: AuthorityBundle) -> GraphStageRegistry:
    run_modes: dict[str, GraphRunMode] = {}
    for item in bundle.graph.raw["run_modes"]:
        run_mode_id = str(item["run_mode"])
        if run_mode_id in run_modes:
            raise GraphRegistryValidationError(f"duplicate run mode {run_mode_id!r}")
        production_accepted = item["production_accepted"]
        if not isinstance(production_accepted, bool):
            raise GraphRegistryValidationError(
                f"run mode {run_mode_id!r} production_accepted must be a boolean"
            )
        run_modes[run_mode_id] = GraphRunMode(
            run_mode=run_mode_id,
            production_accepted=production_accepted,
            description=str(item["description"]),
        )
    stages = {
        stage_id: CanonicalGraphStage(
            stage_id=stage_id,
            intended_phase=str(stage.raw["intended_phase"]),
            capture_policy=str(stage.raw["capture_policy"]),
            loop_owner=str(stage.raw["loop_owner"]),
            fallback_mode=str(stage.raw["fallback_mode"]),
            notes=str(stage.raw["notes"]),
        )
        for stage_id, stage in bundle.graph.stages_by_id.items()
    }
    invalid_stage_fallbacks = sorted(
        {
            stage.stage_id: stage.fallback_mode
            for stage in stages.values()
            if stage.fallback_mode not in run_modes
        }.items()
    )
    if invalid_stage_fallbacks:
        formatted = ", ".join(
            f"{stage_id} -> {fallback_mode}"
            for stage_id, fallback_mode in invalid_stage_fallbacks
        )
        raise GraphRegistryValidationError(
            "stages reference unknown fallback run modes: " + formatted
        )
    registry = GraphStageRegistry(
        run_modes_by_id=run_modes,
        stages_by_id=stages,
        required_orchestration_ranges=tuple(bundle.graph.required_orchestration_ranges),
        global_capture_rules=dict(bundle.graph.raw["global_capture_rules"]),
    )
    validate_acceptance_tuple_stage_requirements(
        bundle,
        registry=registry,
        expected_orchestration_ranges=registry.required_orchestration_ranges,
    )
    return registry


def validate_acceptance_tuple_stage_requirements(
    bundle: AuthorityBundle,
    *,
    registry: GraphStageRegistry | None = None,
    expected_orchestration_ranges: tuple[str, ...] | None = None,
) -> TupleStageValidationReport:
    resolved_registry = registry
    if resolved_registry is None:
        resolved_registry = build_graph_stage_registry(bundle)
    accepted_execution_modes = tuple(
        bundle.acceptance.raw["coverage_rules"]["accepted_execution_modes"]
    )
    backend_restrictions = bundle.acceptance.raw["coverage_rules"]["backend_restrictions"]
    amgx_admitted_case_ids = set(backend_restrictions["amgx_admitted_case_ids"])
    amgx_admitted_execution_modes = set(backend_restrictions["amgx_admitted_execution_modes"])
    tuple_stage_ids: dict[str, tuple[str, ...]] = {}
    for tuple_id, accepted_tuple in bundle.acceptance.tuples_by_id.items():
        _validate_acceptance_tuple_contract(
            accepted_tuple,
            registry=resolved_registry,
            accepted_execution_modes=accepted_execution_modes,
            amgx_admitted_case_ids=amgx_admitted_case_ids,
            amgx_admitted_execution_modes=amgx_admitted_execution_modes,
        )
        tuple_stage_ids[tuple_id] = accepted_tuple.required_stage_ids
    report = validate_tuple_stage_requirements(
        resolved_registry,
        tuple_stage_ids,
    )
    acceptance_orchestration_ranges = tuple(
        bundle.acceptance.raw["nvtx_contract_defaults"]["required_orchestration_ranges"]
    )
    if acceptance_orchestration_ranges != resolved_registry.required_orchestration_ranges:
        raise GraphRegistryValidationError(
            "acceptance NVTX orchestration ranges do not match the canonical graph registry"
        )
    if (
        expected_orchestration_ranges is not None
        and expected_orchestration_ranges != resolved_registry.required_orchestration_ranges
    ):
        raise GraphRegistryValidationError(
            "expected orchestration ranges do not match the canonical graph registry"
        )
    return report


def _validate_acceptance_tuple_contract(
    accepted_tuple: AcceptedTuple,
    *,
    registry: GraphStageRegistry,
    accepted_execution_modes: tuple[str, ...],
    amgx_admitted_case_ids: set[str],
    amgx_admitted_execution_modes: set[str],
) -> None:
    tuple_id = accepted_tuple.tuple_id
    execution_mode = accepted_tuple.execution_mode
    if execution_mode not in accepted_execution_modes:
        raise GraphRegistryValidationError(
            f"{tuple_id} uses non-accepted execution mode {execution_mode!r}"
        )
    if not registry.run_mode(execution_mode).production_accepted:
        raise GraphRegistryValidationError(
            f"{tuple_id} uses non-production run mode {execution_mode!r}"
        )

    stage_ids = accepted_tuple.required_stage_ids
    if "write_stage" in stage_ids:
        raise GraphRegistryValidationError(
            f"{tuple_id} may not require 'write_stage' in the current acceptance manifest"
        )

    expected_pressure_stage_by_backend = {
        "native": "pressure_solve_native",
        "amgx": "pressure_solve_amgx",
    }
    try:
        expected_pressure_stage = expected_pressure_stage_by_backend[accepted_tuple.backend]
    except KeyError as exc:
        raise GraphRegistryValidationError(
            f"{tuple_id} uses unknown backend {accepted_tuple.backend!r}"
        ) from exc
    unexpected_pressure_stages = sorted(
        set(expected_pressure_stage_by_backend.values()) - {expected_pressure_stage}
    )
    if expected_pressure_stage not in stage_ids:
        raise GraphRegistryValidationError(
            f"{tuple_id} backend {accepted_tuple.backend!r} requires stage {expected_pressure_stage!r}"
        )
    present_unexpected_stages = [
        stage_id for stage_id in unexpected_pressure_stages if stage_id in stage_ids
    ]
    if present_unexpected_stages:
        raise GraphRegistryValidationError(
            f"{tuple_id} backend {accepted_tuple.backend!r} may not require stages {', '.join(repr(stage_id) for stage_id in present_unexpected_stages)}"
        )

    if accepted_tuple.backend == "amgx":
        if accepted_tuple.case_id not in amgx_admitted_case_ids:
            raise GraphRegistryValidationError(
                f"{tuple_id} backend 'amgx' is not admitted for case {accepted_tuple.case_id!r}"
            )
        if execution_mode not in amgx_admitted_execution_modes:
            raise GraphRegistryValidationError(
                f"{tuple_id} backend 'amgx' is not admitted for execution mode {execution_mode!r}"
            )


def validate_tuple_stage_requirements(
    registry: GraphStageRegistry,
    tuple_stage_ids: Mapping[str, Iterable[str]],
) -> TupleStageValidationReport:
    validated: dict[str, tuple[str, ...]] = {}
    stage_ids_in_use: set[str] = set()
    for tuple_id, stage_ids in tuple_stage_ids.items():
        normalized_stage_ids = tuple(str(stage_id) for stage_id in stage_ids)
        unknown_stage_ids = [
            stage_id for stage_id in normalized_stage_ids if stage_id not in registry.stages_by_id
        ]
        if unknown_stage_ids:
            raise GraphRegistryValidationError(
                f"{tuple_id} references unknown canonical stage ids: {', '.join(sorted(set(unknown_stage_ids)))}"
            )
        validated[tuple_id] = normalized_stage_ids
        stage_ids_in_use.update(normalized_stage_ids)
    return TupleStageValidationReport(
        validated_tuple_count=len(validated),
        tuple_stage_ids=validated,
        stage_ids_in_use=tuple(sorted(stage_ids_in_use)),
    )
