"""Semantic source-audit helpers built on top of the frozen authority bundle."""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass

from .bundle import AuthorityBundle


SOURCE_AUDIT_TEMPLATE_PATH = pathlib.Path("docs/tasks/templates/source_audit_note.md")
SOURCE_AUDIT_AUTHORITY_PATH = "docs/authority/semantic_source_map.md"
SOURCE_AUDIT_REVIEWED_STATUS = "reviewed"


@dataclass(frozen=True)
class ResolvedSourceAuditSurface:
    contract_surface: str
    semantic_reference: str
    local_target_family: str
    ownership_scope: str
    notes: str


def resolve_source_audit_surfaces(
    bundle: AuthorityBundle,
    contract_surfaces: list[str] | tuple[str, ...],
) -> tuple[ResolvedSourceAuditSurface, ...]:
    normalized_surfaces = _normalize_requested_surfaces(contract_surfaces)
    available_surfaces = bundle.semantic_source_map.entries_by_surface
    unknown_surfaces = sorted(
        surface for surface in normalized_surfaces if surface not in available_surfaces
    )
    if unknown_surfaces:
        raise ValueError(
            "unknown semantic contract surfaces: " + ", ".join(unknown_surfaces)
        )

    return tuple(
        ResolvedSourceAuditSurface(
            contract_surface=surface,
            semantic_reference=available_surfaces[surface].semantic_reference,
            local_target_family=available_surfaces[surface].local_target_family,
            ownership_scope=bundle.semantic_source_map.owner_for(surface),
            notes=available_surfaces[surface].notes,
        )
        for surface in normalized_surfaces
    )


def render_source_audit_note(
    bundle: AuthorityBundle,
    *,
    touched_surfaces: list[str] | tuple[str, ...],
    review_status: str = SOURCE_AUDIT_REVIEWED_STATUS,
    note_title: str = "Source Audit Note",
    reviewer_notes: list[str] | tuple[str, ...] | None = None,
) -> str:
    _load_template_contract(bundle.root)
    resolved_surfaces = resolve_source_audit_surfaces(bundle, touched_surfaces)
    rendered_rows = [
        "| Contract surface | Semantic reference | Local target family | Ownership scope | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    rendered_rows.extend(
        [
            "| {contract_surface} | {semantic_reference} | {local_target_family} | "
            "{ownership_scope} | {notes} |".format(
                contract_surface=surface.contract_surface,
                semantic_reference=surface.semantic_reference,
                local_target_family=surface.local_target_family,
                ownership_scope=surface.ownership_scope,
                notes=surface.notes,
            )
            for surface in resolved_surfaces
        ]
    )
    reviewer_notes = list(reviewer_notes or ())
    if not reviewer_notes:
        reviewer_notes.append(
            "Reviewed against the frozen semantic source map before implementation planning."
        )

    lines = [
        f"# {note_title}",
        "",
        "## Review Status",
        "",
        f"- Review status: {review_status}",
        f"- Authority source: {SOURCE_AUDIT_AUTHORITY_PATH}",
        "- Contract note: patch the local SPUMA/v2412 target family named by the authority bundle, not an upstream analog path.",
        "",
        "## Semantic Surface Coverage",
        "",
        *rendered_rows,
        "",
        "## Reviewer Notes",
        "",
        *[f"- {note}" for note in reviewer_notes],
        "",
    ]
    return "\n".join(lines)


def validate_source_audit_note(
    bundle: AuthorityBundle,
    *,
    note_text: str,
    touched_surfaces: list[str] | tuple[str, ...],
) -> tuple[ResolvedSourceAuditSurface, ...]:
    _load_template_contract(bundle.root)
    resolved_surfaces = resolve_source_audit_surfaces(bundle, touched_surfaces)
    if _extract_review_status(note_text) != SOURCE_AUDIT_REVIEWED_STATUS:
        raise ValueError("source-audit note must be marked reviewed")

    note_rows = _parse_coverage_rows(note_text)
    missing_surfaces = [
        surface.contract_surface
        for surface in resolved_surfaces
        if surface.contract_surface not in note_rows
    ]
    if missing_surfaces:
        raise ValueError(
            "source-audit note is missing required semantic surfaces: "
            + ", ".join(missing_surfaces)
        )

    for resolved_surface in resolved_surfaces:
        observed_row = note_rows[resolved_surface.contract_surface]
        expected_row = {
            "contract_surface": resolved_surface.contract_surface,
            "semantic_reference": resolved_surface.semantic_reference,
            "local_target_family": resolved_surface.local_target_family,
            "ownership_scope": resolved_surface.ownership_scope,
        }
        comparable_row = {
            key: observed_row[key]
            for key in (
                "contract_surface",
                "semantic_reference",
                "local_target_family",
                "ownership_scope",
            )
        }
        if comparable_row != expected_row:
            raise ValueError(
                "source-audit note row for "
                f"{resolved_surface.contract_surface!r} does not match frozen semantic mapping"
            )
    return resolved_surfaces


def _normalize_requested_surfaces(
    contract_surfaces: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    normalized: list[str] = []
    for surface in contract_surfaces:
        cleaned = str(surface).strip()
        if not cleaned:
            continue
        if cleaned not in normalized:
            normalized.append(cleaned)
    if not normalized:
        raise ValueError("at least one semantic contract surface is required")
    return tuple(normalized)


def _extract_review_status(note_text: str) -> str | None:
    match = re.search(r"^- Review status:\s*`?([^`\n]+?)`?\s*$", note_text, re.MULTILINE)
    if not match:
        return None
    return match.group(1).strip()


def _parse_coverage_rows(note_text: str) -> dict[str, dict[str, str]]:
    coverage_section = _extract_markdown_section(note_text, "Semantic Surface Coverage")
    rows: dict[str, dict[str, str]] = {}
    for line in coverage_section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if stripped.startswith("| Contract surface |") or stripped.startswith("| ---"):
            continue
        parts = [part.strip() for part in stripped.strip("|").split("|")]
        if len(parts) != 5:
            continue
        row = {
            "contract_surface": parts[0],
            "semantic_reference": parts[1],
            "local_target_family": parts[2],
            "ownership_scope": parts[3],
            "notes": parts[4],
        }
        rows[row["contract_surface"]] = row
    return rows


def _extract_markdown_section(text: str, heading: str) -> str:
    pattern = re.compile(
        rf"^## {re.escape(heading)}\s*$\n(?P<body>.*?)(?=^## |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        raise ValueError(f"source-audit note is missing section '## {heading}'")
    return match.group("body").strip()


def _load_template_contract(root: pathlib.Path) -> str:
    template_path = root / SOURCE_AUDIT_TEMPLATE_PATH
    if not template_path.exists():
        raise FileNotFoundError(template_path.name)
    return template_path.read_text(encoding="utf-8")
