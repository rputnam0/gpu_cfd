#!/usr/bin/env python3
"""Authority bundle loader and validator for the gpu_cfd roadmap docs."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
from dataclasses import dataclass
from typing import Any


SUPPORTED_SCHEMA_VERSION = "1.0.0"


class AuthorityLoadError(RuntimeError):
    """Base class for authority loading failures."""


class AuthoritySchemaError(AuthorityLoadError):
    """Raised when a JSON authority artifact has an unsupported schema."""


class AuthorityConflictError(AuthorityLoadError):
    """Raised when authority artifacts disagree on a frozen value."""


@dataclass(frozen=True)
class JsonAuthorityArtifact:
    filename: str
    schema_version: str
    authority_markdown: str


@dataclass(frozen=True)
class PinManifest:
    reviewed_source_tuple_id: str
    runtime_base: str
    primary_toolkit_lane: str
    experimental_toolkit_lane: str
    driver_floor: str
    gpu_target: str
    workstation_target: str
    instrumentation: str
    nsight_systems: str
    nsight_compute: str
    compute_sanitizer: str
    source_components: dict[str, "PinSourceComponent"]
    required_revalidation: tuple[str, ...]


@dataclass(frozen=True)
class PinSourceComponent:
    component: str
    upstream_object: str
    frozen_ref_kind: str
    frozen_ref: str
    resolved_commit: str


@dataclass(frozen=True)
class ReferenceCase:
    case_id: str
    frozen_id: str
    purpose: str
    frozen_default_contract: dict[str, Any]


@dataclass(frozen=True)
class CaseContract:
    by_case_id: dict[str, ReferenceCase]
    phase_gate_mapping: dict[str, Any]
    locked_defaults: dict[str, Any]


@dataclass(frozen=True)
class ValidationLadder:
    ordered_case_ids: tuple[str, ...]


@dataclass(frozen=True)
class SupportGlobalPolicy:
    default_fallback_policy: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class SupportMatrix:
    global_policy: SupportGlobalPolicy
    raw: dict[str, Any]


@dataclass(frozen=True)
class AcceptedTuple:
    tuple_id: str
    case_id: str
    backend: str
    execution_mode: str
    required_stage_ids: tuple[str, ...]
    raw: dict[str, Any]


@dataclass(frozen=True)
class AcceptanceManifest:
    tuples_by_id: dict[str, AcceptedTuple]
    raw: dict[str, Any]


@dataclass(frozen=True)
class GraphStage:
    stage_id: str
    fallback_mode: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class GraphCaptureMatrix:
    run_modes: tuple[str, ...]
    stages_by_id: dict[str, GraphStage]
    required_orchestration_ranges: tuple[str, ...]
    raw: dict[str, Any]

    def stage(self, stage_id: str) -> GraphStage:
        return self.stages_by_id[stage_id]


@dataclass(frozen=True)
class SemanticSourceEntry:
    contract_surface: str
    semantic_reference: str
    local_target_family: str
    notes: str


@dataclass(frozen=True)
class SemanticSourceMap:
    entries_by_surface: dict[str, SemanticSourceEntry]

    def owner_for(self, contract_surface: str) -> str:
        owner = self.entries_by_surface[contract_surface].local_target_family
        backticked_targets = re.findall(r"`([^`]+)`", owner)
        if backticked_targets:
            return backticked_targets[0].strip()
        normalized_owner = owner.strip()
        normalized_owner = re.sub(r"^local\s+", "", normalized_owner)
        normalized_owner = re.split(r",|\s+plus\s+|\s+/\s+", normalized_owner, maxsplit=1)[0]
        normalized_owner = re.sub(r"\s+path$", "", normalized_owner)
        return normalized_owner.strip().strip("`")


@dataclass(frozen=True)
class ContinuityLedger:
    central_package_authorities: tuple[str, ...]
    package_consumption_rules: tuple[str, ...]


@dataclass(frozen=True)
class AuthorityLoadReport:
    root: str
    loaded_markdown: tuple[str, ...]
    loaded_json: tuple[JsonAuthorityArtifact, ...]
    diagnostics: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "loaded_markdown": list(self.loaded_markdown),
            "loaded_json": [
                {
                    "filename": artifact.filename,
                    "schema_version": artifact.schema_version,
                    "authority_markdown": artifact.authority_markdown,
                }
                for artifact in self.loaded_json
            ],
            "diagnostics": list(self.diagnostics),
        }


@dataclass(frozen=True)
class AuthorityBundle:
    root: pathlib.Path
    continuity: ContinuityLedger
    pins: PinManifest
    cases: CaseContract
    ladder: ValidationLadder
    support: SupportMatrix
    acceptance: AcceptanceManifest
    graph: GraphCaptureMatrix
    semantic_source_map: SemanticSourceMap
    authority_revisions: dict[str, dict[str, str]]
    report: AuthorityLoadReport

    def reference_case(self, case_id: str) -> ReferenceCase:
        return self.cases.by_case_id[case_id]


REQUIRED_MARKDOWN_FILES = (
    "docs/authority/continuity_ledger.md",
    "docs/authority/master_pin_manifest.md",
    "docs/authority/reference_case_contract.md",
    "docs/authority/validation_ladder.md",
    "docs/authority/support_matrix.md",
    "docs/authority/acceptance_manifest.md",
    "docs/authority/graph_capture_support_matrix.md",
    "docs/authority/semantic_source_map.md",
)

REQUIRED_JSON_FILES = (
    "docs/authority/reference_case_contract.json",
    "docs/authority/support_matrix.json",
    "docs/authority/acceptance_manifest.json",
    "docs/authority/graph_capture_support_matrix.json",
)

EXPECTED_JSON_AUTHORITY_MARKDOWN = {
    "reference_case_contract.json": "reference_case_contract.md",
    "support_matrix.json": "support_matrix.md",
    "acceptance_manifest.json": "acceptance_manifest.md",
    "graph_capture_support_matrix.json": "graph_capture_support_matrix.md",
}

EXPECTED_JSON_COMPANION_REFERENCES = {
    "acceptance_manifest.json": {
        "reference_case_contract": "reference_case_contract.json",
        "support_matrix": "support_matrix.json",
        "graph_capture_support_matrix": "graph_capture_support_matrix.json",
    }
}


def repo_root(start: pathlib.Path | None = None) -> pathlib.Path:
    if start is not None:
        return start.resolve()
    return pathlib.Path(__file__).resolve().parents[2]


def load_authority_bundle(root: pathlib.Path | str | None = None) -> AuthorityBundle:
    resolved_root = repo_root(pathlib.Path(root) if root is not None else None)

    markdown_text = {
        relative_path: read_required_file(resolved_root / relative_path)
        for relative_path in REQUIRED_MARKDOWN_FILES
    }
    json_payloads = {
        relative_path: load_json_artifact(resolved_root / relative_path)
        for relative_path in REQUIRED_JSON_FILES
    }

    continuity = parse_continuity_ledger(markdown_text["docs/authority/continuity_ledger.md"])
    pins = parse_pin_manifest(markdown_text["docs/authority/master_pin_manifest.md"])
    cases = parse_case_contract(json_payloads["docs/authority/reference_case_contract.json"])
    ladder = parse_validation_ladder(markdown_text["docs/authority/validation_ladder.md"])
    support = parse_support_matrix(json_payloads["docs/authority/support_matrix.json"])
    acceptance = parse_acceptance_manifest(json_payloads["docs/authority/acceptance_manifest.json"])
    graph = parse_graph_capture_matrix(
        json_payloads["docs/authority/graph_capture_support_matrix.json"]
    )
    semantic_source_map = parse_semantic_source_map(
        markdown_text["docs/authority/semantic_source_map.md"]
    )
    authority_revisions = {
        key: {
            "path": relative_path,
            "sha256": hashlib.sha256((resolved_root / relative_path).read_bytes()).hexdigest(),
        }
        for key, relative_path in {
            "master_pin_manifest": "docs/authority/master_pin_manifest.md",
            "acceptance_manifest": "docs/authority/acceptance_manifest.json",
            "support_matrix": "docs/authority/support_matrix.json",
            "graph_capture_support_matrix": "docs/authority/graph_capture_support_matrix.json",
        }.items()
    }

    diagnostics = validate_consistency(
        markdown_text=markdown_text,
        continuity=continuity,
        pins=pins,
        cases=cases,
        ladder=ladder,
        support=support,
        acceptance=acceptance,
        graph=graph,
        semantic_source_map=semantic_source_map,
    )

    loaded_json = tuple(
        JsonAuthorityArtifact(
            filename=pathlib.Path(relative_path).name,
            schema_version=str(payload["schema_version"]),
            authority_markdown=str(payload["authority_markdown"]),
        )
        for relative_path, payload in json_payloads.items()
    )
    report = AuthorityLoadReport(
        root=resolved_root.as_posix(),
        loaded_markdown=tuple(pathlib.Path(path).name for path in markdown_text),
        loaded_json=loaded_json,
        diagnostics=tuple(diagnostics),
    )
    return AuthorityBundle(
        root=resolved_root,
        continuity=continuity,
        pins=pins,
        cases=cases,
        ladder=ladder,
        support=support,
        acceptance=acceptance,
        graph=graph,
        semantic_source_map=semantic_source_map,
        authority_revisions=authority_revisions,
        report=report,
    )


def read_required_file(path: pathlib.Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path.name)
    return path.read_text(encoding="utf-8")


def load_json_artifact(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path.name)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "schema_version" not in payload:
        raise AuthoritySchemaError(f"{path.name} is missing schema_version")
    schema_version = payload["schema_version"]
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise AuthoritySchemaError(
            f"{path.name} has unsupported schema_version {schema_version!r}; "
            f"expected {SUPPORTED_SCHEMA_VERSION!r}"
        )
    if "authority_markdown" not in payload:
        raise AuthoritySchemaError(f"{path.name} is missing authority_markdown")
    expected_markdown = EXPECTED_JSON_AUTHORITY_MARKDOWN.get(path.name)
    if expected_markdown is not None and payload["authority_markdown"] != expected_markdown:
        raise AuthoritySchemaError(
            f"{path.name} must reference authority_markdown {expected_markdown!r}; "
            f"found {payload['authority_markdown']!r}"
        )
    expected_companions = EXPECTED_JSON_COMPANION_REFERENCES.get(path.name, {})
    for field_name, expected_filename in expected_companions.items():
        if payload.get(field_name) != expected_filename:
            raise AuthoritySchemaError(
                f"{path.name} must reference {field_name} {expected_filename!r}; "
                f"found {payload.get(field_name)!r}"
            )
    return payload


def parse_continuity_ledger(text: str) -> ContinuityLedger:
    central_section = markdown_section(text, "4. Central Package Authorities")
    rule_lines = section_bullets(text, "5. Package Consumption Rule")
    central_docs = tuple(
        match.group(1)
        for match in re.finditer(
            r"^\d+\.\s+\*\*`([^`]+)`\*\*",
            central_section,
            re.MULTILINE,
        )
    )
    rules = tuple(
        line.split(".", 1)[1].strip() if re.match(r"\d+\.", line) else line
        for line in rule_lines
    )
    return ContinuityLedger(
        central_package_authorities=central_docs,
        package_consumption_rules=rules,
    )


def parse_pin_manifest(text: str) -> PinManifest:
    frozen_defaults = markdown_table_by_first_column(text, "Frozen Defaults")
    source_rows = markdown_table_by_first_column(text, "Resolved Frozen Source Tuple")
    required_revalidation = tuple(parse_required_revalidation(text))
    return PinManifest(
        reviewed_source_tuple_id=frozen_defaults["Reviewed source tuple ID"]["Frozen value"],
        runtime_base=frozen_defaults["Runtime base"]["Frozen value"],
        primary_toolkit_lane=frozen_defaults["Primary toolkit lane"]["Frozen value"],
        experimental_toolkit_lane=frozen_defaults["Experimental toolkit lane"]["Frozen value"],
        driver_floor=frozen_defaults["Driver floor"]["Frozen value"],
        gpu_target=frozen_defaults["GPU target"]["Frozen value"],
        workstation_target=frozen_defaults["Workstation target"]["Frozen value"],
        instrumentation=frozen_defaults["Instrumentation"]["Frozen value"],
        nsight_systems=frozen_defaults["Nsight Systems"]["Frozen value"],
        nsight_compute=frozen_defaults["Nsight Compute"]["Frozen value"],
        compute_sanitizer=frozen_defaults["Compute Sanitizer"]["Frozen value"],
        source_components={
            component: PinSourceComponent(
                component=component,
                upstream_object=row["Upstream object"],
                frozen_ref_kind=row["Frozen ref kind"],
                frozen_ref=row["Frozen ref / version"],
                resolved_commit=row["Exact resolved commit / snapshot"],
            )
            for component, row in source_rows.items()
        },
        required_revalidation=required_revalidation,
    )


def parse_case_contract(payload: dict[str, Any]) -> CaseContract:
    cases = build_unique_index(
        payload["frozen_cases"],
        id_field="case_id",
        artifact_name="reference_case_contract.json",
        item_builder=lambda item: ReferenceCase(
            case_id=str(item["case_id"]),
            frozen_id=str(item["frozen_id"]),
            purpose=str(item["purpose"]),
            frozen_default_contract=dict(item["frozen_default_contract"]),
        ),
    )
    return CaseContract(
        by_case_id=cases,
        phase_gate_mapping=dict(payload["phase_gate_mapping"]),
        locked_defaults=dict(payload["locked_defaults"]),
    )


def parse_validation_ladder(text: str) -> ValidationLadder:
    lines = section_bullets(text, "Frozen Ladder")
    ordered_case_ids = []
    for line in lines:
        match = re.match(r"\d+\.\s+`([^`]+)`", line)
        if match:
            ordered_case_ids.append(match.group(1))
    return ValidationLadder(ordered_case_ids=tuple(ordered_case_ids))


def parse_support_matrix(payload: dict[str, Any]) -> SupportMatrix:
    return SupportMatrix(
        global_policy=SupportGlobalPolicy(
            default_fallback_policy=str(payload["global_policy"]["default_fallback_policy"]),
            raw=dict(payload["global_policy"]),
        ),
        raw=payload,
    )


def parse_acceptance_manifest(payload: dict[str, Any]) -> AcceptanceManifest:
    tuples_by_id = build_unique_index(
        payload["accepted_tuples"],
        id_field="tuple_id",
        artifact_name="acceptance_manifest.json",
        item_builder=lambda item: AcceptedTuple(
            tuple_id=str(item["tuple_id"]),
            case_id=str(item["case_id"]),
            backend=str(item["backend"]),
            execution_mode=str(item["execution_mode"]),
            required_stage_ids=tuple(str(stage_id) for stage_id in item["required_stage_ids"]),
            raw=dict(item),
        ),
    )
    return AcceptanceManifest(tuples_by_id=tuples_by_id, raw=payload)


def parse_graph_capture_matrix(payload: dict[str, Any]) -> GraphCaptureMatrix:
    stages = build_unique_index(
        payload["stages"],
        id_field="stage_id",
        artifact_name="graph_capture_support_matrix.json",
        item_builder=lambda item: GraphStage(
            stage_id=str(item["stage_id"]),
            fallback_mode=str(item["fallback_mode"]),
            raw=dict(item),
        ),
    )
    run_modes = tuple(str(item["run_mode"]) for item in payload["run_modes"])
    return GraphCaptureMatrix(
        run_modes=run_modes,
        stages_by_id=stages,
        required_orchestration_ranges=tuple(payload["required_orchestration_ranges"]),
        raw=payload,
    )


def parse_semantic_source_map(text: str) -> SemanticSourceMap:
    rows = markdown_table_rows(text, "Frozen Mapping")
    entries: dict[str, SemanticSourceEntry] = {}
    for row in rows:
        surface = row["Contract surface"]
        if surface in entries:
            raise AuthorityConflictError(
                f"duplicate contract surface {surface!r} in semantic_source_map.md"
            )
        entries[surface] = SemanticSourceEntry(
            contract_surface=surface,
            semantic_reference=row["Semantic reference"],
            local_target_family=row["Local implementation target family"],
            notes=row["Notes"],
        )
    return SemanticSourceMap(entries_by_surface=entries)


def validate_consistency(
    *,
    markdown_text: dict[str, str],
    continuity: ContinuityLedger,
    pins: PinManifest,
    cases: CaseContract,
    ladder: ValidationLadder,
    support: SupportMatrix,
    acceptance: AcceptanceManifest,
    graph: GraphCaptureMatrix,
    semantic_source_map: SemanticSourceMap,
) -> list[str]:
    diagnostics = [
        "Loaded all required authority markdown and JSON artifacts.",
        f"Pinned primary toolkit lane: {pins.primary_toolkit_lane}.",
        f"Validation ladder: {' -> '.join(ladder.ordered_case_ids)}.",
    ]
    expected_authorities = {
        "master_pin_manifest.md",
        "reference_case_contract.md",
        "validation_ladder.md",
        "support_matrix.md",
        "acceptance_manifest.md",
        "graph_capture_support_matrix.md",
        "semantic_source_map.md",
    }
    found_authorities = set(continuity.central_package_authorities)
    if found_authorities != expected_authorities:
        raise AuthorityConflictError(
            "central package authorities do not match the expected authority bundle"
        )
    if set(cases.by_case_id) != {"R0", "R1", "R1-core", "R2"}:
        raise AuthorityConflictError(
            "reference_case_contract.json case membership drifted from the frozen authority set"
        )
    if ladder.ordered_case_ids != ("R2", "R1-core", "R1", "R0"):
        raise AuthorityConflictError("validation ladder no longer matches the frozen authority order")
    if set(ladder.ordered_case_ids) != set(cases.by_case_id):
        raise AuthorityConflictError(
            "validation_ladder.md and reference_case_contract.json must define the same frozen case ids"
        )
    if cases.locked_defaults.get("r1_core_required_case_id") != "R1-core":
        raise AuthorityConflictError(
            "locked_defaults.r1_core_required_case_id must remain R1-core"
        )
    unknown_phase_gate_cases = sorted(
        extract_unknown_phase_gate_cases(cases.phase_gate_mapping, set(cases.by_case_id))
    )
    if unknown_phase_gate_cases:
        raise AuthorityConflictError(
            "phase_gate_mapping references unknown case ids: "
            + ", ".join(unknown_phase_gate_cases)
        )
    if support.global_policy.default_fallback_policy != "failFast":
        raise AuthorityConflictError(
            "support_matrix.json default_fallback_policy must remain failFast"
        )
    missing_cases = sorted(
        {
            accepted_tuple.case_id
            for accepted_tuple in acceptance.tuples_by_id.values()
            if accepted_tuple.case_id not in cases.by_case_id
        }
    )
    if missing_cases:
        raise AuthorityConflictError(
            f"acceptance_manifest.json references unknown case ids: {', '.join(missing_cases)}"
        )
    unknown_stage_ids = sorted(
        {
            stage_id
            for accepted_tuple in acceptance.tuples_by_id.values()
            for stage_id in accepted_tuple.required_stage_ids
            if stage_id not in graph.stages_by_id
        }
    )
    if unknown_stage_ids:
        raise AuthorityConflictError(
            "acceptance_manifest.json references unknown stage ids: "
            + ", ".join(unknown_stage_ids)
        )
    unknown_execution_modes = sorted(
        {
            accepted_tuple.execution_mode
            for accepted_tuple in acceptance.tuples_by_id.values()
            if accepted_tuple.execution_mode not in graph.run_modes
        }
    )
    if unknown_execution_modes:
        raise AuthorityConflictError(
            "acceptance_manifest.json references unknown execution modes: "
            + ", ".join(unknown_execution_modes)
        )
    acceptance_orchestration_ranges = tuple(
        acceptance.raw["nvtx_contract_defaults"]["required_orchestration_ranges"]
    )
    if acceptance_orchestration_ranges != graph.required_orchestration_ranges:
        raise AuthorityConflictError(
            "acceptance_manifest.json and graph_capture_support_matrix.json required_orchestration_ranges must match exactly"
        )
    graph_fallbacks = {
        stage.fallback_mode
        for stage in graph.stages_by_id.values()
        if stage.fallback_mode not in graph.run_modes
    }
    if graph_fallbacks:
        raise AuthorityConflictError(
            "graph_capture_support_matrix.json uses stage fallback modes not present in run_modes: "
            + ", ".join(sorted(graph_fallbacks))
        )
    validate_markdown_json_alignment(
        markdown_text=markdown_text,
        cases=cases,
        support=support,
        acceptance=acceptance,
        graph=graph,
    )
    required_surfaces = {
        "Pressure bridge",
        "Pressure corrector",
        "Profiling/instrumentation touch points",
    }
    missing_surfaces = sorted(required_surfaces - set(semantic_source_map.entries_by_surface))
    if missing_surfaces:
        raise AuthorityConflictError(
            "semantic_source_map.md is missing required contract surfaces: "
            + ", ".join(missing_surfaces)
        )
    return diagnostics


def validate_markdown_json_alignment(
    *,
    markdown_text: dict[str, str],
    cases: CaseContract,
    support: SupportMatrix,
    acceptance: AcceptanceManifest,
    graph: GraphCaptureMatrix,
) -> None:
    reference_case_rows = markdown_table_rows(
        markdown_text["docs/authority/reference_case_contract.md"],
        "Frozen Cases",
    )
    markdown_case_rows = {
        strip_backticks(row["Case"]): {
            "frozen_id": strip_backticks(row["Frozen ID"]),
            "purpose": normalize_markdown_text(row["Purpose"]),
        }
        for row in reference_case_rows
    }
    json_case_rows = {
        case_id: {
            "frozen_id": strip_backticks(case.frozen_id),
            "purpose": normalize_markdown_text(case.purpose),
        }
        for case_id, case in cases.by_case_id.items()
    }
    if markdown_case_rows != json_case_rows:
        raise AuthorityConflictError(
            "reference_case_contract.md and reference_case_contract.json must define the same frozen case ids and frozen ids"
        )
    reference_phase_gate_lines = {
        normalize_markdown_text(line)
        for line in section_bullets(
            markdown_text["docs/authority/reference_case_contract.md"],
            "Phase-Gate Mapping",
        )
    }
    if reference_phase_gate_lines != build_reference_phase_gate_lines(cases.phase_gate_mapping):
        raise AuthorityConflictError(
            "reference_case_contract.md and reference_case_contract.json must agree on the phase-gate mapping"
        )
    reference_locked_default_lines = {
        normalize_markdown_text(line)
        for line in section_bullets(
            markdown_text["docs/authority/reference_case_contract.md"],
            "Locked Defaults",
        )
    }
    if reference_locked_default_lines != build_reference_locked_default_lines(cases.locked_defaults):
        raise AuthorityConflictError(
            "reference_case_contract.md and reference_case_contract.json must agree on locked defaults"
        )

    validate_support_matrix_markdown(
        markdown_text["docs/authority/support_matrix.md"],
        support,
    )

    acceptance_rows = markdown_table_rows(
        markdown_text["docs/authority/acceptance_manifest.md"],
        "Accepted Tuple Matrix",
    )
    markdown_acceptance_rows = {
        row["tuple_id"]: row for row in normalize_acceptance_markdown_rows(acceptance_rows)
    }
    json_acceptance_rows = {
        tuple_id: normalize_acceptance_json_row(tuple_data.raw)
        for tuple_id, tuple_data in acceptance.tuples_by_id.items()
    }
    if markdown_acceptance_rows != json_acceptance_rows:
        raise AuthorityConflictError(
            "acceptance_manifest.md and acceptance_manifest.json must define the same tuple rows"
        )

    hard_gate_lines = section_bullets(
        markdown_text["docs/authority/acceptance_manifest.md"],
        "Hard Gates",
    )
    markdown_hard_gate_keys = {
        extract_metric_key_from_backticked_expression(line) for line in hard_gate_lines
    }
    if markdown_hard_gate_keys != set(acceptance.raw["hard_gates"]):
        raise AuthorityConflictError(
            "acceptance_manifest.md hard gates must match acceptance_manifest.json hard gates"
        )
    soft_gate_lines = {
        normalize_markdown_text(line)
        for line in section_bullets(
            markdown_text["docs/authority/acceptance_manifest.md"],
            "Soft Gates",
        )
    }
    if soft_gate_lines != build_acceptance_soft_gate_lines(acceptance.raw["soft_gates"]):
        raise AuthorityConflictError(
            "acceptance_manifest.md soft gates must match acceptance_manifest.json soft gates"
        )

    graph_run_modes = {
        extract_first_backticked_token(line)
        for line in section_bullets(
            markdown_text["docs/authority/graph_capture_support_matrix.md"],
            "Run Modes",
        )
    }
    if graph_run_modes != set(graph.run_modes):
        raise AuthorityConflictError(
            "graph_capture_support_matrix.md and graph_capture_support_matrix.json must define the same run modes"
        )
    graph_global_capture_lines = {
        normalize_markdown_text(line)
        for line in section_bullets(
            markdown_text["docs/authority/graph_capture_support_matrix.md"],
            "Global Capture Rules",
        )
    }
    if graph_global_capture_lines != build_graph_global_capture_lines(graph.raw["global_capture_rules"]):
        raise AuthorityConflictError(
            "graph_capture_support_matrix.md and graph_capture_support_matrix.json must agree on global capture rules"
        )

    graph_stage_rows = markdown_table_rows(
        markdown_text["docs/authority/graph_capture_support_matrix.md"],
        "Canonical Stage IDs",
    )
    markdown_graph_rows = {
        row["stage_id"]: row for row in normalize_graph_stage_markdown_rows(graph_stage_rows)
    }
    json_graph_rows = {
        stage_id: normalize_graph_stage_json_row(stage.raw)
        for stage_id, stage in graph.stages_by_id.items()
    }
    if markdown_graph_rows != json_graph_rows:
        raise AuthorityConflictError(
            "graph_capture_support_matrix.md and graph_capture_support_matrix.json must define the same stage rows"
        )


def validate_support_matrix_markdown(text: str, support: SupportMatrix) -> None:
    global_policy_lines = {
        normalize_markdown_text(line) for line in section_bullets(text, "Global Policy")
    }
    if global_policy_lines != build_support_global_policy_lines(support.raw["global_policy"]):
        raise AuthorityConflictError(
            "support_matrix.md and support_matrix.json must agree on global policy"
        )

    phase5_lines = section_bullets(text, "Phase 5 Generic VOF Envelope")
    expected_phase5 = support.raw["phase5_generic_vof_envelope"]
    if parse_prefixed_backticked_list(phase5_lines, "Scalars:") != set(
        expected_phase5["scalar_boundary_kinds"]
    ):
        raise AuthorityConflictError(
            "support_matrix.md and support_matrix.json must agree on the Phase 5 scalar boundary kinds"
        )
    if parse_prefixed_backticked_list(phase5_lines, "Vectors:") != set(
        expected_phase5["vector_boundary_kinds"]
    ):
        raise AuthorityConflictError(
            "support_matrix.md and support_matrix.json must agree on the Phase 5 vector boundary kinds"
        )
    if parse_prefixed_backticked_list(phase5_lines, "Geometry patch families:") != set(
        expected_phase5["geometry_patch_families"]
    ):
        raise AuthorityConflictError(
            "support_matrix.md and support_matrix.json must agree on the Phase 5 geometry patch families"
        )
    runtime_policy_line = next(
        (line for line in phase5_lines if line.startswith("Schemes/runtime policy:")),
        None,
    )
    expected_runtime_fragments = [
        "Euler ddt",
        "generic BC/scheme subset only",
        "no contact-angle patch fields",
        "no nozzle-specific swirl inlet logic",
        "no pressureInletOutletVelocity",
        "no fixedFluxPressure",
        "no processor/coupled patches",
    ]
    if runtime_policy_line is None or not all(
        normalize_markdown_text(fragment) in normalize_markdown_text(runtime_policy_line)
        for fragment in expected_runtime_fragments
    ):
        raise AuthorityConflictError(
            "support_matrix.md and support_matrix.json must agree on the Phase 5 runtime policy"
        )

    scheme_rows = markdown_table_rows(text, "Exact Audited Scheme Tuple")
    markdown_scheme_rows = {
        normalize_markdown_text(row["Block"]): normalize_markdown_text(row["Exact allowed entry"])
        for row in scheme_rows
    }
    json_scheme_rows = build_support_matrix_scheme_rows(support.raw["exact_audited_scheme_tuple"])
    if markdown_scheme_rows != json_scheme_rows:
        raise AuthorityConflictError(
            "support_matrix.md and support_matrix.json must agree on the exact audited scheme tuple"
        )

    function_rows = markdown_table_rows(text, "FunctionObject Classification")
    markdown_function_rows = build_markdown_function_object_rows(function_rows)
    json_function_rows = build_json_function_object_rows(support.raw["function_object_policy"])
    if markdown_function_rows != json_function_rows:
        raise AuthorityConflictError(
            "support_matrix.md and support_matrix.json must agree on functionObject classifications"
        )

    nozzle_rows = markdown_table_rows(text, "Phase 6 Nozzle-Specific Envelope")
    markdown_nozzle_rows = {
        (
            normalize_markdown_text(row["Patch role"]),
            normalize_markdown_text(row["Field"]),
        ): normalize_allowed_kinds_cell(row["Allowed milestone-1 kinds"])
        for row in nozzle_rows
    }
    json_nozzle_rows = {
        (
            normalize_markdown_text(row["patch_role"]),
            normalize_markdown_text(row["field"]),
        ): normalize_allowed_kind_sequence(row["allowed_kinds"])
        for row in support.raw["phase6_nozzle_specific_envelope"]
    }
    if markdown_nozzle_rows != json_nozzle_rows:
        raise AuthorityConflictError(
            "support_matrix.md and support_matrix.json must agree on the Phase 6 nozzle-specific envelope"
        )

    startup_rows = markdown_table_by_first_column(text, "Canonical Startup-Seed DSL")
    startup_dsl = support.raw["startup_seed_dsl"]
    canonical_owner = normalize_markdown_text(startup_rows["Canonical owner"]["Frozen rule"])
    expected_owner = normalize_markdown_text(
        f"{startup_dsl['canonical_owner']['file']} -> {startup_dsl['canonical_owner']['path']}"
    )
    owner_notes = normalize_markdown_text(startup_rows["Canonical owner"].get("Notes", ""))
    if expected_owner not in canonical_owner or startup_dsl["canonical_owner"]["compatibility_shim"] not in owner_notes:
        raise AuthorityConflictError(
            "support_matrix.md and support_matrix.json must agree on the startup-seed canonical owner"
        )
    if parse_backticked_tokens(startup_rows["Top-level keys"]["Frozen rule"]) != set(
        startup_dsl["top_level_keys"]
    ):
        raise AuthorityConflictError(
            "support_matrix.md and support_matrix.json must agree on startup-seed top-level keys"
        )
    if parse_backticked_tokens(startup_rows["Precedence policy"]["Frozen rule"]) != set(
        startup_dsl["precedence_policy"]["allowed_values"]
    ):
        raise AuthorityConflictError(
            "support_matrix.md and support_matrix.json must agree on startup-seed precedence policy"
        )
    if parse_backticked_tokens(
        startup_rows["Supported region families"]["Frozen rule"]
    ) != {row["region_type"] for row in startup_dsl["supported_region_types"]}:
        raise AuthorityConflictError(
            "support_matrix.md and support_matrix.json must agree on supported startup-seed region families"
        )
    allowed_entry_tokens = parse_backticked_tokens(
        startup_rows["Allowed field-value entries"]["Frozen rule"]
    )
    expected_entry_tokens = {
        f"{row['value_class']} {row['field']} <{row['value_type']}>"
        if row["value_type"] != "vector"
        else f"{row['value_class']} {row['field']} (<x> <y> <z>)"
        for row in startup_dsl["allowed_field_value_entries"]
    }
    if allowed_entry_tokens != expected_entry_tokens:
        raise AuthorityConflictError(
            "support_matrix.md and support_matrix.json must agree on allowed startup-seed field-value entries"
        )

    backend_lines = section_bullets(text, "Backend and Operational Policy")
    expected_backend_fragments = [
        "Native pressure is the required baseline on every accepted case.",
        "AmgX is a supported secondary backend only through the Phase 4 bridge.",
        "Any AmgX production claim requires DeviceDirect",
        "Unsupported cases must fail during startup support scanning before the first timestep.",
    ]
    if not all(
        any(fragment in normalize_markdown_text(line) for line in backend_lines)
        for fragment in map(normalize_markdown_text, expected_backend_fragments)
    ):
        raise AuthorityConflictError(
            "support_matrix.md and support_matrix.json must agree on backend operational policy"
        )


def normalize_acceptance_markdown_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        case_id, case_variant = parse_case_variant_cell(row["Case / variant"])
        admission, required_pressure_bridge_mode = normalize_markdown_admission(row["Admission"])
        normalized_rows.append(
            {
                "tuple_id": strip_backticks(row["Tuple ID"]),
                "phase_gate": normalize_markdown_text(row["Phase gate"]),
                "case_id": case_id,
                "case_variant": case_variant,
                "backend": strip_backticks(row["Backend"]),
                "execution_mode": strip_backticks(row["Execution mode"]),
                "kernel_family_mode": strip_backticks(row["Kernel family mode"]),
                "admission": admission,
                "required_pressure_bridge_mode": required_pressure_bridge_mode,
                "production_eligible": normalize_yes_no(row["Production-eligible"]),
                "tolerance_class": normalize_optional_markdown_value(row["Tolerance class"]),
                "restart_reload_parity_class": normalize_optional_markdown_value(
                    row["Restart / reload parity"]
                ),
                "execution_parity_class": normalize_optional_markdown_value(
                    row["Execution-parity class"]
                ),
                "execution_peer_tuple_id": normalize_optional_markdown_value(
                    row["Execution peer tuple"]
                ),
                "backend_parity_class": normalize_optional_markdown_value(
                    row["Backend-parity class"]
                ),
                "backend_peer_tuple_id": normalize_optional_markdown_value(
                    row["Backend peer tuple"]
                ),
                "kernel_parity_class": normalize_optional_markdown_value(
                    row["Kernel-parity class"]
                ),
                "kernel_peer_tuple_id": normalize_optional_markdown_value(
                    row["Kernel peer tuple"]
                ),
            }
        )
    return normalized_rows


def normalize_acceptance_json_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "tuple_id": row["tuple_id"],
        "phase_gate": normalize_markdown_text(row["phase_gate"]),
        "case_id": row["case_id"],
        "case_variant": normalize_markdown_text(row["case_variant"]),
        "backend": row["backend"],
        "execution_mode": row["execution_mode"],
        "kernel_family_mode": row["kernel_family_mode"],
        "admission": row["admission"],
        "required_pressure_bridge_mode": row.get("required_pressure_bridge_mode"),
        "production_eligible": row["production_eligible"],
        "tolerance_class": row["tolerance_class"],
        "restart_reload_parity_class": row.get("restart_reload_parity_class"),
        "execution_parity_class": row.get("execution_parity_class"),
        "execution_peer_tuple_id": row.get("execution_peer_tuple_id"),
        "backend_parity_class": row.get("backend_parity_class"),
        "backend_peer_tuple_id": row.get("backend_peer_tuple_id"),
        "kernel_parity_class": row.get("kernel_parity_class"),
        "kernel_peer_tuple_id": row.get("kernel_peer_tuple_id"),
    }


def normalize_graph_stage_markdown_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "stage_id": strip_backticks(row["Stage ID"]),
            "intended_phase": normalize_markdown_text(row["Intended phase"]),
            "capture_policy": normalize_markdown_text(row["Capture policy"]),
            "loop_owner": normalize_markdown_text(row["Loop owner"]),
            "fallback_mode": strip_backticks(row["Fallback mode"]),
            "notes": normalize_markdown_text(row["Notes"]),
        }
        for row in rows
    ]


def normalize_graph_stage_json_row(row: dict[str, Any]) -> dict[str, str]:
    return {
        "stage_id": row["stage_id"],
        "intended_phase": normalize_markdown_text(row["intended_phase"]),
        "capture_policy": normalize_markdown_text(row["capture_policy"]),
        "loop_owner": normalize_markdown_text(row["loop_owner"]),
        "fallback_mode": row["fallback_mode"],
        "notes": normalize_markdown_text(row["notes"]),
    }


def build_support_matrix_scheme_rows(raw: dict[str, Any]) -> dict[str, str]:
    rows: dict[str, str] = {}
    for block_name in ("ddtSchemes", "gradSchemes", "interpolationSchemes", "divSchemes"):
        for key, value in raw[block_name].items():
            row_key = f"{block_name}.{key}" if key != "default" else f"{block_name}.default"
            rows[row_key] = normalize_markdown_text(str(value))
    return rows


def build_markdown_function_object_rows(rows: list[dict[str, str]]) -> dict[str, tuple[str, str]]:
    normalized: dict[str, tuple[str, str]] = {}
    for row in rows:
        classes = [
            token.strip()
            for token in normalize_markdown_text(row["FunctionObject class"]).split(",")
        ]
        classification = normalize_markdown_text(row["Classification"])
        rule = normalize_markdown_text(row["Production-mode rule"])
        for class_name in classes:
            if class_name == "any unlisted class":
                normalized[class_name] = (classification, rule)
                continue
            normalized[class_name] = (classification, rule)
    return normalized


def build_json_function_object_rows(raw: dict[str, Any]) -> dict[str, tuple[str, str]]:
    normalized = {
        row["class_name"]: (
            normalize_markdown_text(row["classification"]),
            normalize_markdown_text(row["production_mode_rule"]),
        )
        for row in raw["classes"]
    }
    normalized["any unlisted class"] = (
        normalize_markdown_text(raw["default_classification"]),
        normalize_markdown_text("reject before the first timestep"),
    )
    return normalized


def build_reference_phase_gate_lines(phase_gate_mapping: dict[str, Any]) -> set[str]:
    phase0 = phase_gate_mapping["Phase 0"]
    phase2 = phase_gate_mapping["Phase 2"]
    phase5 = phase_gate_mapping["Phase 5"]
    phase8 = phase_gate_mapping["Phase 8"]
    baselines = " and ".join(phase0["baselines"])
    return {
        normalize_markdown_text(
            f"Phase 0 freezes `{' -> '.join(phase0['ordered_case_ladder'])}` on {baselines}."
        ),
        normalize_markdown_text(
            f"Phase 2 uses `{phase2['default_cases'][0]}` plus `{phase2['default_cases'][1]}` by default. "
            f"`{phase2['conditional_cases'][0]}` is used only when nozzle-specific topology or patch-manifest coverage is intentionally under test."
        ),
        normalize_markdown_text(
            f"Phase 5 hard gates use two generic `{phase5['hard_gate_cases'][0]}` slices plus `{phase5['hard_gate_cases'][1]}`."
        ),
        normalize_markdown_text(
            f"Phase 6 and Phase 7 use `{phase_gate_mapping['Phase 6']['accepted_case']}` as the reduced nozzle acceptance case."
        ),
        normalize_markdown_text(
            f"Phase 8 uses `{phase8['routine_architecture_baseline_case']}` for routine architectural baselines, "
            f"`{phase8['production_shape_acceptance_case']}` for production-shape acceptance, and "
            f"`{phase8['backend_or_execution_parity_case']}` whenever backend or execution-mode parity is needed without Phase 6 nozzle-specific BCs."
        ),
    }


def build_reference_locked_default_lines(locked_defaults: dict[str, Any]) -> set[str]:
    return {
        normalize_markdown_text(
            f"Hard-gating `R0` case: `{locked_defaults['hard_gating_r0_case']}`."
        ),
        normalize_markdown_text(
            f"`{locked_defaults['shadow_reference_cases'][0]}` remains an optional shadow/reference case and is not a milestone-1 hard gate."
        ),
        normalize_markdown_text(
            "Pressure-drop is not a Phase 0 case-freeze selector. Formal later-phase pressure-drop gates, when present, are owned exclusively by `acceptance_manifest.md` / `acceptance_manifest.json`."
        ),
        normalize_markdown_text(
            f"`{locked_defaults['r1_core_required_case_id']}` is mandatory in the ladder and may not be replaced by a descriptive \"reduced generic case\" label."
        ),
    }


def build_support_global_policy_lines(global_policy: dict[str, Any]) -> set[str]:
    return {
        normalize_markdown_text("Static mesh only."),
        normalize_markdown_text("Single region only."),
        normalize_markdown_text(
            "No processor patches, cyclic/AMI patches, or arbitrary coded patch fields in milestone-1 production scope."
        ),
        normalize_markdown_text("Contact-angle is out of milestone-1 scope."),
        normalize_markdown_text("Surface-tension scope is constant `sigma` only."),
        normalize_markdown_text("Turbulence scope is laminar-only."),
        normalize_markdown_text(
            f"Default fallback policy is `{global_policy['default_fallback_policy']}`."
        ),
        normalize_markdown_text(
            "CPU/stage fallback, host patch execution, and host `setFields` startup are debug-only/bring-up-only modes and are forbidden in production acceptance."
        ),
        normalize_markdown_text(
            "In performance mode, only functionObjects classified `writeTimeOnly` are allowed. `debugOnly` entries are allowed only in explicit debug runs."
        ),
    }


def build_acceptance_soft_gate_lines(soft_gates: dict[str, Any]) -> set[str]:
    graph_launches = soft_gates["graph_launches_per_step"]["value"]
    kernel_regression = soft_gates["top_kernel_time_regression_pct"]["value"]
    return {
        normalize_markdown_text(f"`graph_launches_per_step <= {graph_launches}`"),
        normalize_markdown_text(
            f"`top_kernel_time_regression_pct <= {int(kernel_regression)}%` versus the locked baseline"
        ),
    }


def build_graph_global_capture_lines(global_capture_rules: dict[str, Any]) -> set[str]:
    failure_policy = global_capture_rules["capture_failure_policy"]
    if failure_policy != "downgrade_to_async_no_graph_with_logged_reason":
        raise AuthorityConflictError(
            f"unsupported capture failure policy {failure_policy!r}"
        )
    return {
        normalize_markdown_text("No post-warmup dynamic allocation."),
        normalize_markdown_text("No hidden CPU patch evaluation in capture-safe stages."),
        normalize_markdown_text("No silent host reads of device-authoritative fields."),
        normalize_markdown_text(
            "Any stage that fails capture/update/rebuild downgrades to `async_no_graph` with a logged reason."
        ),
    }


def normalize_allowed_kinds_cell(value: str) -> tuple[str, ...]:
    normalized = normalize_markdown_text(value)
    if "/" in normalized:
        return tuple(part.strip() for part in normalized.split("/"))
    return tuple(part.strip() for part in normalized.split(","))


def normalize_allowed_kind_sequence(values: list[str]) -> tuple[str, ...]:
    return tuple(normalize_markdown_text(value) for value in values)


def parse_prefixed_backticked_list(lines: list[str], prefix: str) -> set[str]:
    for line in lines:
        if line.startswith(prefix):
            return parse_backticked_tokens(line)
    raise AuthorityConflictError(f"missing bullet {prefix!r}")


def parse_backticked_tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"`([^`]+)`", value)}


def parse_case_variant_cell(value: str) -> tuple[str, str]:
    case_id, _, variant = normalize_markdown_text(value).partition(" / ")
    return case_id, variant


def normalize_markdown_admission(value: str) -> tuple[str, str | None]:
    normalized = normalize_markdown_text(value)
    if normalized == "Required":
        return "required", None
    if normalized.startswith("Optional benchmark-only"):
        bridge_mode = "DeviceDirect" if "DeviceDirect" in normalized else None
        return "optional_benchmark_only", bridge_mode
    raise AuthorityConflictError(f"unsupported acceptance admission text: {value}")


def normalize_yes_no(value: str) -> bool:
    normalized = normalize_markdown_text(value)
    if normalized == "Yes":
        return True
    if normalized == "No":
        return False
    raise AuthorityConflictError(f"unsupported Yes/No value: {value}")


def normalize_optional_markdown_value(value: str) -> str | None:
    normalized = strip_backticks(value).strip()
    if normalized == "None":
        return None
    return normalize_markdown_text(normalized)


def normalize_markdown_text(value: str) -> str:
    translated = (
        value.replace("`", "")
        .replace("“", "\"")
        .replace("”", "\"")
        .replace("’", "'")
    )
    return " ".join(translated.split())


def section_bullets(text: str, heading: str) -> list[str]:
    section = markdown_section(text, heading)
    lines: list[str] = []
    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("- ") or re.match(r"\d+\.\s", line):
            lines.append(line.removeprefix("- ").strip())
    return lines


def markdown_section(text: str, heading: str) -> str:
    pattern = re.compile(
        rf"^##?\s+{re.escape(heading)}\s*$",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        raise AuthorityConflictError(f"missing markdown section: {heading}")
    start = match.end()
    following = re.search(r"^##?\s+", text[start:], re.MULTILINE)
    end = start + following.start() if following else len(text)
    return text[start:end].strip()


def markdown_table_by_first_column(text: str, heading: str) -> dict[str, dict[str, str]]:
    rows = markdown_table_rows(text, heading)
    if not rows:
        raise AuthorityConflictError(f"empty markdown table: {heading}")
    first_key = next(iter(rows[0]))
    indexed_rows: dict[str, dict[str, str]] = {}
    for row in rows:
        row_key = row[first_key]
        if row_key in indexed_rows:
            raise AuthorityConflictError(
                f"duplicate markdown key {row_key!r} in section: {heading}"
            )
        indexed_rows[row_key] = row
    return indexed_rows


def markdown_table_rows(text: str, heading: str) -> list[dict[str, str]]:
    section = markdown_section(text, heading)
    table_lines = [line.strip() for line in section.splitlines() if line.strip().startswith("|")]
    if len(table_lines) < 3:
        raise AuthorityConflictError(f"missing markdown table rows in section: {heading}")
    headers = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    data_lines = table_lines[2:]
    rows: list[dict[str, str]] = []
    for line in data_lines:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(headers):
            raise AuthorityConflictError(f"malformed markdown table row in section: {heading}")
        rows.append(dict(zip(headers, cells, strict=True)))
    return rows


def parse_required_revalidation(text: str) -> list[str]:
    section = markdown_section(text, "Required Revalidation If This Manifest Changes")
    results: list[str] = []
    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        number, separator, content = line.partition(". ")
        if separator and number.isdigit():
            results.append(content)
    return results


def strip_backticks(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("`") and stripped.endswith("`"):
        return stripped[1:-1]
    return stripped


def extract_first_backticked_token(line: str) -> str:
    match = re.search(r"`([^`]+)`", line)
    if not match:
        raise AuthorityConflictError(f"expected backticked token in markdown line: {line}")
    return match.group(1)


def extract_metric_key_from_backticked_expression(line: str) -> str:
    expression = extract_first_backticked_token(line)
    return expression.split()[0]


def build_unique_index(
    items: list[dict[str, Any]],
    *,
    id_field: str,
    artifact_name: str,
    item_builder: Any,
) -> dict[str, Any]:
    indexed: dict[str, Any] = {}
    for item in items:
        item_id = str(item[id_field])
        if item_id in indexed:
            raise AuthorityConflictError(
                f"duplicate {id_field} {item_id!r} in {artifact_name}"
            )
        indexed[item_id] = item_builder(item)
    return indexed


def extract_unknown_phase_gate_cases(
    value: Any,
    known_case_ids: set[str],
    *,
    field_name: str | None = None,
) -> set[str]:
    if isinstance(value, dict):
        unknown: set[str] = set()
        for nested_field, nested_value in value.items():
            unknown.update(
                extract_unknown_phase_gate_cases(
                    nested_value,
                    known_case_ids,
                    field_name=str(nested_field),
                )
            )
        return unknown
    if isinstance(value, list):
        if field_name and field_name in CASE_REFERENCE_SEQUENCE_FIELDS:
            return {str(item) for item in value if str(item) not in known_case_ids}
        return set()
    if isinstance(value, str) and field_name and field_name in CASE_REFERENCE_SCALAR_FIELDS:
        return {value} if value not in known_case_ids else set()
    return set()


CASE_REFERENCE_SEQUENCE_FIELDS = {
    "ordered_case_ladder",
    "default_cases",
    "conditional_cases",
    "hard_gate_cases",
}


CASE_REFERENCE_SCALAR_FIELDS = {
    "accepted_case",
    "routine_architecture_baseline_case",
    "production_shape_acceptance_case",
    "backend_or_execution_parity_case",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=None,
        help="Repository root to load. Defaults to the current repo root.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the authority load report as JSON.",
    )
    return parser


def format_text_report(report: AuthorityLoadReport) -> str:
    lines = [
        "Authority bundle load report",
        f"Root: {report.root}",
        "Markdown artifacts:",
    ]
    lines.extend(f"- {filename}" for filename in report.loaded_markdown)
    lines.append("JSON companions:")
    lines.extend(
        f"- {artifact.filename} (schema {artifact.schema_version}, authority {artifact.authority_markdown})"
        for artifact in report.loaded_json
    )
    lines.append("Diagnostics:")
    lines.extend(f"- {diagnostic}" for diagnostic in report.diagnostics)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    bundle = load_authority_bundle(args.root)
    payload = bundle.report.as_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(format_text_report(bundle.report))
    return 0
