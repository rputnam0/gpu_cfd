#!/usr/bin/env python3
"""Phase-boundary cleanup helpers for the Symphony orchestration harness."""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass
from typing import Any

from scripts.symphony import linear_api


TASK_CARD_ID_PATTERN = re.compile(r"\b(?:FND-\d+|P\d+-\d+)\b", re.IGNORECASE)
PULL_REQUEST_URL_PATTERN = re.compile(r"^- Pull request:\s+(?P<url>https://\S+)\s*$", re.MULTILINE)
REVIEW_ARTIFACT_PATTERN = re.compile(
    r"^- Review artifact:\s+`(?P<artifact>[^`]+)`\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class SectionInfo:
    name: str
    label: str
    slug: str
    pr_ids: tuple[str, ...]
    authoritative_sources: tuple[str, ...]
    exports_to_next: tuple[str, ...]
    tracked_notes: tuple[str, ...]
    open_discontinuities: tuple[str, ...]


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def normalize_section_label(section_name: str) -> str:
    if "/" in section_name:
        return section_name.split("/", 1)[0].strip()
    if "—" in section_name:
        return section_name.split("—", 1)[0].strip()
    return section_name.strip()


def section_slug(section_name: str) -> str:
    normalized = normalize_section_label(section_name).lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    return normalized or "phase-cleanup"


def infer_pr_id(issue_title: str) -> str | None:
    match = TASK_CARD_ID_PATTERN.search(issue_title or "")
    if not match:
        return None
    return match.group(0).upper()


def load_sections(root: pathlib.Path | None = None) -> list[SectionInfo]:
    payload = load_backlog_payload(root)
    sections: list[SectionInfo] = []
    for entry in payload.get("sections", []):
        name = str(entry.get("section") or "").strip()
        sections.append(
            SectionInfo(
                name=name,
                label=normalize_section_label(name),
                slug=section_slug(name),
                pr_ids=tuple(str(pr.get("id") or "").upper() for pr in entry.get("prs", [])),
                authoritative_sources=tuple(str(item) for item in entry.get("authoritative_sources", [])),
                exports_to_next=tuple(str(item) for item in entry.get("exports_to_next", [])),
                tracked_notes=tuple(str(item) for item in entry.get("tracked_notes", [])),
                open_discontinuities=tuple(str(item) for item in entry.get("open_discontinuities", [])),
            )
        )
    return sections


def load_backlog_payload(root: pathlib.Path | None = None) -> dict[str, Any]:
    backlog_path = (root or repo_root()) / "docs" / "backlog" / "gpu_cfd_pr_backlog.json"
    return json.loads(backlog_path.read_text(encoding="utf-8"))


def load_pr_dependencies(root: pathlib.Path | None = None) -> dict[str, tuple[str, ...]]:
    payload = load_backlog_payload(root)
    dependencies: dict[str, tuple[str, ...]] = {}
    for section_payload in payload.get("sections", []):
        for pr_payload in section_payload.get("prs", []):
            pr_id = str(pr_payload.get("id") or "").upper()
            dependencies[pr_id] = tuple(
                str(dep).upper() for dep in pr_payload.get("depends_on", [])
            )
    return dependencies


def sections_by_pr_id(root: pathlib.Path | None = None) -> dict[str, SectionInfo]:
    mapping: dict[str, SectionInfo] = {}
    for section in load_sections(root):
        for pr_id in section.pr_ids:
            mapping[pr_id] = section
    return mapping


def build_issue_maps(team_issues: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_identifier: dict[str, dict[str, Any]] = {}
    by_pr_id: dict[str, dict[str, Any]] = {}
    for issue in team_issues:
        identifier = str(issue.get("identifier") or "").strip()
        if identifier:
            by_identifier[identifier] = issue
        pr_id = infer_pr_id(str(issue.get("title") or ""))
        if pr_id:
            by_pr_id[pr_id] = issue
    return by_identifier, by_pr_id


def issue_state(issue: dict[str, Any]) -> str:
    return str((issue.get("state") or {}).get("name") or "")


def issue_labels(issue: dict[str, Any]) -> list[str]:
    return [
        str(node.get("name") or "").strip()
        for node in (issue.get("labels") or {}).get("nodes", [])
        if str(node.get("name") or "").strip()
    ]


def find_cleanup_sweep(
    team_issues: list[dict[str, Any]],
    *,
    section_slug_value: str,
) -> dict[str, Any] | None:
    for issue in team_issues:
        marker = linear_api.extract_phase_cleanup_marker(str(issue.get("description") or ""))
        if marker and marker["section"] == section_slug_value:
            return issue
    return None


def section_boundary_reached(
    section: SectionInfo,
    implementation_issues: dict[str, dict[str, Any]],
) -> bool:
    return all(
        issue_state(implementation_issues.get(pr_id, {})) == "Done"
        for pr_id in section.pr_ids
    )


def section_cleanup_complete(
    section: SectionInfo,
    team_issues: list[dict[str, Any]],
    implementation_issues: dict[str, dict[str, Any]],
) -> bool:
    if not section_boundary_reached(section, implementation_issues):
        return False
    sweep = find_cleanup_sweep(team_issues, section_slug_value=section.slug)
    if sweep is None or issue_state(sweep) != "Done":
        return False
    sweep_identifier = str(sweep.get("identifier") or "")
    return not any(
        str(((issue.get("parent") or {}).get("identifier")) or "") == sweep_identifier
        and issue_state(issue) != "Done"
        for issue in team_issues
    )


def residual_parent_pr_id(issue: dict[str, Any]) -> str | None:
    marker = linear_api.extract_residual_followup_marker(str(issue.get("description") or ""))
    if marker is None:
        return None
    return marker["parent"]


def parse_residual_links(description: str) -> dict[str, str | None]:
    pr_match = PULL_REQUEST_URL_PATTERN.search(description or "")
    artifact_match = REVIEW_ARTIFACT_PATTERN.search(description or "")
    return {
        "pr_url": pr_match.group("url") if pr_match else None,
        "review_artifact": artifact_match.group("artifact") if artifact_match else None,
    }


def find_section_residual_followups(
    section: SectionInfo,
    team_issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    section_pr_ids = set(section.pr_ids)
    matches: list[dict[str, Any]] = []
    for issue in team_issues:
        parent_pr_id = residual_parent_pr_id(issue)
        if parent_pr_id in section_pr_ids and issue_state(issue) != "Done":
            matches.append(issue)
    return matches


def render_cleanup_sweep_description(
    section: SectionInfo,
    residual_followups: list[dict[str, Any]],
    implementation_issues: dict[str, dict[str, Any]],
) -> str:
    lines = [
        linear_api.render_phase_cleanup_marker(section.slug),
        "",
        f"Phase cleanup sweep for `{section.label}`.",
        "",
        "## Section Context",
        f"- Section: {section.name}",
        "- PR IDs: " + ", ".join(f"`{pr_id}`" for pr_id in section.pr_ids),
        "- Authoritative sources: "
        + (
            ", ".join(f"`{item}`" for item in section.authoritative_sources)
            if section.authoritative_sources
            else "None recorded"
        ),
        "",
        "## Residual Follow-ups",
    ]
    if residual_followups:
        for issue in residual_followups:
            metadata = parse_residual_links(str(issue.get("description") or ""))
            identifier = str(issue.get("identifier") or "").strip()
            title = str(issue.get("title") or "").strip()
            state = issue_state(issue) or "Unknown"
            lines.append(f"- {identifier}: {title} (`{state}`)")
            if metadata["pr_url"]:
                lines.append(f"- Originating PR: {metadata['pr_url']}")
            if metadata["review_artifact"]:
                lines.append(f"- Review artifact: `{metadata['review_artifact']}`")
    else:
        lines.append("- No residual local-review follow-up issues were present at boundary time.")
    lines.extend(
        [
            "",
            "## Cleanup Inputs",
            "- Exports to next: "
            + (
                ", ".join(section.exports_to_next)
                if section.exports_to_next
                else "None recorded"
            ),
            "- Tracked notes: "
            + (
                ", ".join(section.tracked_notes)
                if section.tracked_notes
                else "None recorded"
            ),
            "- Open discontinuities: "
            + (
                ", ".join(section.open_discontinuities)
                if section.open_discontinuities
                else "None recorded"
            ),
            "",
            "## Hygiene Brief",
            "- Garbage collect phase-local dead paths, duplicate helpers, and stale compatibility shims.",
            "- Simplify confusing code paths and messy patterns inside the section-owned surfaces.",
            "- Keep the cleanup bounded to the completed phase slice; do not widen into future sections.",
            "",
            "## Implementation Snapshot",
        ]
    )
    for pr_id in section.pr_ids:
        issue = implementation_issues.get(pr_id)
        identifier = str((issue or {}).get("identifier") or "unmapped")
        current_state = issue_state(issue or {}) or "Unknown"
        lines.append(f"- {pr_id}: {identifier} (`{current_state}`)")
    return "\n".join(lines).strip() + "\n"


def ensure_cleanup_sweep(
    *,
    section: SectionInfo,
    current_issue: dict[str, Any],
    team_issues: list[dict[str, Any]],
    implementation_issues: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    residual_followups = find_section_residual_followups(section, team_issues)
    description = render_cleanup_sweep_description(
        section,
        residual_followups,
        implementation_issues,
    )
    existing = find_cleanup_sweep(team_issues, section_slug_value=section.slug)
    created_or_updated: dict[str, Any]
    if existing is None:
        created_or_updated = {
            "action": "created",
            **linear_api.create_issue(
                team_id=str((current_issue.get("team") or {}).get("id") or ""),
                title=f"Phase cleanup: {section.label}",
                description=description,
                state_id=linear_api.resolve_state_id(current_issue, "Todo"),
                label_ids=linear_api.resolve_label_ids(["phase-cleanup"]),
            ),
        }
    else:
        updated = linear_api.update_issue(
            str(existing["identifier"]),
            target_state_name="Todo" if issue_state(existing) != "Done" else "Done",
            description=description,
            label_names=sorted(set(issue_labels(existing) + ["phase-cleanup"])),
        )
        created_or_updated = {
            "action": "updated",
            "identifier": existing["identifier"],
            "title": existing["title"],
            "url": existing.get("url"),
            **updated,
        }
    sweep_identifier = str(created_or_updated.get("identifier") or "")
    promoted_followups: list[dict[str, Any]] = []
    for followup in residual_followups:
        label_names = sorted(set(issue_labels(followup) + ["phase-cleanup"]))
        update_result = linear_api.update_issue(
            str(followup["identifier"]),
            target_state_name="Todo",
            parent_identifier=sweep_identifier,
            label_names=label_names,
        )
        promoted_followups.append(
            {
                "identifier": followup["identifier"],
                "previous_state": update_result["previous_state"],
                "current_state": update_result["current_state"],
                "previous_parent_identifier": update_result["previous_parent_identifier"],
                "current_parent_identifier": update_result["current_parent_identifier"],
                "current_labels": update_result["current_labels"],
                "changed": update_result["changed"],
            }
        )

    auto_closed = False
    has_cleanup_inputs = bool(
        residual_followups
        or section.exports_to_next
        or section.tracked_notes
        or section.open_discontinuities
    )
    if not has_cleanup_inputs and sweep_identifier:
        linear_api.update_issue(sweep_identifier, target_state_name="Done")
        linear_api.upsert_workpad_comment(
            sweep_identifier,
            linear_api.render_workpad_body(
                issue_identifier=sweep_identifier,
                issue_title=f"Phase cleanup: {section.label}",
                current_status="phase_cleanup_complete",
                validation=[
                    "Boundary audit completed with no residual bugs or hygiene work.",
                ],
                review_handoff_notes=[
                    "Phase cleanup auto-closed without a PR because the section audit was clean.",
                ],
            ),
        )
        auto_closed = True

    return {
        "section": section.slug,
        "sweep_identifier": sweep_identifier or None,
        "cleanup_issue": created_or_updated,
        "promoted_followups": promoted_followups,
        "auto_closed": auto_closed,
        "residual_followup_count": len(residual_followups),
    }


def resolve_section_for_issue(
    issue: dict[str, Any],
    team_issues: list[dict[str, Any]],
    *,
    root: pathlib.Path | None = None,
) -> SectionInfo | None:
    by_identifier, _implementation_issues = build_issue_maps(team_issues)
    section_map = sections_by_pr_id(root)
    available_sections = load_sections(root)

    cleanup_marker = linear_api.extract_phase_cleanup_marker(str(issue.get("description") or ""))
    if cleanup_marker:
        return next(
            (section for section in available_sections if section.slug == cleanup_marker["section"]),
            None,
        )

    parent_pr_id = residual_parent_pr_id(issue)
    if parent_pr_id and parent_pr_id in section_map:
        return section_map[parent_pr_id]

    pr_id = infer_pr_id(str(issue.get("title") or ""))
    if pr_id and pr_id in section_map:
        return section_map[pr_id]

    parent_identifier = str(((issue.get("parent") or {}).get("identifier")) or "").strip()
    if parent_identifier:
        parent_issue = by_identifier.get(parent_identifier)
        if parent_issue is not None:
            cleanup_marker = linear_api.extract_phase_cleanup_marker(
                str(parent_issue.get("description") or "")
            )
            if cleanup_marker:
                return next(
                    (
                        section
                        for section in available_sections
                        if section.slug == cleanup_marker["section"]
                    ),
                    None,
                )
    return None


def finalize_cleanup_sweep_if_ready(
    section: SectionInfo,
    team_issues: list[dict[str, Any]],
) -> dict[str, Any] | None:
    sweep = find_cleanup_sweep(team_issues, section_slug_value=section.slug)
    if sweep is None or issue_state(sweep) == "Done":
        return None
    sweep_identifier = str(sweep.get("identifier") or "")
    open_children = [
        issue
        for issue in team_issues
        if str(((issue.get("parent") or {}).get("identifier")) or "") == sweep_identifier
        and issue_state(issue) != "Done"
    ]
    if open_children:
        return None
    update_result = linear_api.update_issue(sweep_identifier, target_state_name="Done")
    return {
        "identifier": sweep_identifier,
        "previous_state": update_result["previous_state"],
        "current_state": update_result["current_state"],
        "changed": update_result["changed"],
    }


def promote_ready_backlog_issues(
    *,
    team_key: str,
    team_issues: list[dict[str, Any]] | None = None,
    root: pathlib.Path | None = None,
) -> list[dict[str, Any]]:
    live_issues = team_issues or linear_api.list_team_issues(team_key)
    _by_identifier, implementation_issues = build_issue_maps(live_issues)
    section_map = sections_by_pr_id(root)
    dependency_map = load_pr_dependencies(root)
    promoted: list[dict[str, Any]] = []
    for pr_id, issue in implementation_issues.items():
        if issue_state(issue) != "Backlog":
            continue
        if pr_id not in section_map:
            continue
        dependency_ids = set(dependency_map.get(pr_id, ()))
        ready = True
        for dependency in dependency_ids:
            dependency_issue = implementation_issues.get(dependency)
            if dependency_issue is None or issue_state(dependency_issue) != "Done":
                ready = False
                break
            dependency_section = section_map.get(dependency)
            if dependency_section and section_boundary_reached(
                dependency_section,
                implementation_issues,
            ) and not section_cleanup_complete(
                dependency_section,
                live_issues,
                implementation_issues,
            ):
                ready = False
                break
        if not ready:
            continue
        update_result = linear_api.update_issue_state(str(issue["identifier"]), "Todo")
        promoted.append(
            {
                "identifier": issue["identifier"],
                "previous_state": update_result["previous_state"],
                "current_state": update_result["current_state"],
                "changed": update_result["changed"],
            }
        )
    return promoted
