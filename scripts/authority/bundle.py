#!/usr/bin/env python3
"""Authority bundle loader and validator for the gpu_cfd roadmap docs."""

from __future__ import annotations

import argparse
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
    instrumentation: str


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

    diagnostics = validate_consistency(
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
    return PinManifest(
        reviewed_source_tuple_id=frozen_defaults["Reviewed source tuple ID"]["Frozen value"],
        runtime_base=frozen_defaults["Runtime base"]["Frozen value"],
        primary_toolkit_lane=frozen_defaults["Primary toolkit lane"]["Frozen value"],
        experimental_toolkit_lane=frozen_defaults["Experimental toolkit lane"]["Frozen value"],
        driver_floor=frozen_defaults["Driver floor"]["Frozen value"],
        gpu_target=frozen_defaults["GPU target"]["Frozen value"],
        instrumentation=frozen_defaults["Instrumentation"]["Frozen value"],
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
