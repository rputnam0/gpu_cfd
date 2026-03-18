#!/usr/bin/env python3
"""Small Linear GraphQL helpers for the gpu_cfd Symphony workflow."""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

try:
    from scripts.symphony import trace
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    trace = None  # type: ignore[assignment]

LINEAR_GRAPHQL_URL = "https://api.linear.app/graphql"
ISSUE_IDENTIFIER_PATTERN = re.compile(
    r"\b(?P<prefix>PRO)-(?P<number>\d+)\b", re.IGNORECASE
)
LINKED_ISSUE_MARKER_PATTERN = re.compile(
    r"<!--\s*gpu-cfd-linear-issue:\s*(?P<identifier>[A-Z]+-\d+)\s*-->",
    re.IGNORECASE,
)
CLOSING_ISSUE_PATTERN = re.compile(
    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+(?P<identifier>[A-Z]+-\d+)\b",
    re.IGNORECASE,
)
BRANCH_ISSUE_PATTERN = re.compile(
    r"(?:^|/)(?P<identifier>PRO-\d+)(?:\b|-)",
    re.IGNORECASE,
)
WORKPAD_MARKER = "<!-- gpu-cfd-workpad:v1 -->"
WORKPAD_TITLE = "# GPU CFD Worker Workpad"
LEGACY_WORKPAD_MARKERS = (
    "<!-- codex:workpad -->",
    "# Canonical Workpad",
)
LEGACY_WORKPAD_SUPERSEDED_MARKER = "<!-- gpu-cfd-workpad-superseded:v1 -->"
WORKPAD_SECTIONS = (
    "Task Summary",
    "Current Status",
    "Execution Plan",
    "Scoped Sources",
    "Decisions and Rationale",
    "Validation",
    "Risks / Blockers",
    "Review / Handoff Notes",
)
RESIDUAL_FOLLOWUP_MARKER_PATTERN = re.compile(
    r"<!--\s*gpu-cfd-residual-followup:v1\s+parent=(?P<parent>[A-Z]+-\d+)\s+fingerprint=(?P<fingerprint>[a-f0-9]{12})\s*-->",
    re.IGNORECASE,
)
PHASE_CLEANUP_MARKER_PATTERN = re.compile(
    r"<!--\s*gpu-cfd-phase-cleanup:v1\s+section=(?P<section>[a-z0-9-]+)\s*-->",
    re.IGNORECASE,
)
_UNSET = object()

ISSUE_BY_IDENTIFIER_QUERY = """
query($teamKey: String!, $number: Float!) {
  issues(filter: { team: { key: { eq: $teamKey } }, number: { eq: $number } }) {
    nodes {
      id
      identifier
      title
      description
      branchName
      url
      labels {
        nodes {
          name
        }
      }
      parent {
        id
        identifier
      }
      state {
        id
        name
      }
      team {
        id
        key
        states {
          nodes {
            id
            name
          }
        }
      }
    }
  }
}
"""

ISSUE_COMMENTS_QUERY = """
query($teamKey: String!, $number: Float!, $after: String) {
  issues(filter: { team: { key: { eq: $teamKey } }, number: { eq: $number } }) {
    nodes {
      id
      identifier
      title
      comments(first: 100, after: $after) {
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          id
          body
          url
          createdAt
          updatedAt
        }
      }
    }
  }
}
"""

ISSUE_UPDATE_STATE_MUTATION = """
mutation($id: String!, $stateId: String!) {
  issueUpdate(id: $id, input: { stateId: $stateId }) {
    success
    issue {
      id
      identifier
      state {
        id
        name
      }
    }
  }
}
"""

ISSUE_UPDATE_MUTATION_SELECTION = """
    success
    issue {
      id
      identifier
      description
      parent {
        id
        identifier
      }
      labels {
        nodes {
          name
        }
      }
      state {
        id
        name
      }
    }
"""

ISSUE_UPDATE_DESCRIPTION_MUTATION = """
mutation($id: String!, $description: String!) {
  issueUpdate(id: $id, input: { description: $description }) {
    success
    issue {
      id
      identifier
      description
    }
  }
}
"""

ISSUE_CREATE_MUTATION = """
mutation(
  $teamId: String!,
  $title: String!,
  $description: String!,
  $parentId: String,
  $stateId: String,
  $priority: Int,
  $labelIds: [String!]
) {
  issueCreate(
    input: {
      teamId: $teamId,
      title: $title,
      description: $description,
      parentId: $parentId,
      stateId: $stateId,
      priority: $priority,
      labelIds: $labelIds
    }
  ) {
    success
    issue {
      id
      identifier
      title
      url
      state {
        id
        name
      }
    }
  }
}
"""

COMMENT_CREATE_MUTATION = """
mutation($issueId: String!, $body: String!) {
  commentCreate(input: { issueId: $issueId, body: $body }) {
    success
    comment {
      id
      body
      url
      createdAt
      updatedAt
    }
  }
}
"""

COMMENT_UPDATE_MUTATION = """
mutation($id: String!, $body: String!) {
  commentUpdate(id: $id, input: { body: $body }) {
    success
    comment {
      id
      body
      url
      createdAt
      updatedAt
    }
  }
}
"""

DIRECT_BLOCKS_QUERY = """
query($teamKey: String!, $number: Float!) {
  issues(filter: { team: { key: { eq: $teamKey } }, number: { eq: $number } }) {
    nodes {
      id
      identifier
      relations {
        nodes {
          type
          relatedIssue {
            id
            identifier
            state {
              id
              name
            }
            inverseRelations {
              nodes {
                type
                issue {
                  identifier
                  state {
                    name
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
"""

TEAM_ISSUES_QUERY = """
query($teamKey: String!, $after: String) {
  issues(
    filter: { team: { key: { eq: $teamKey } } }
    first: 100
    after: $after
  ) {
    pageInfo {
      hasNextPage
      endCursor
    }
    nodes {
      id
      identifier
      title
      description
      url
      parent {
        id
        identifier
      }
      labels {
        nodes {
          name
        }
      }
      state {
        name
      }
      team {
        key
      }
    }
  }
}
"""

ISSUE_LABELS_QUERY = """
query($after: String) {
  issueLabels(first: 100, after: $after) {
    pageInfo {
      hasNextPage
      endCursor
    }
    nodes {
      id
      name
    }
  }
}
"""


@dataclass(frozen=True)
class ParsedIssueIdentifier:
    team_key: str
    number: int


def parse_issue_identifier(issue_identifier: str) -> ParsedIssueIdentifier:
    normalized = issue_identifier.strip().upper()
    match = re.fullmatch(r"([A-Z]+)-(\d+)", normalized)
    if match is None:
        raise ValueError(f"unsupported Linear issue identifier: {issue_identifier}")
    return ParsedIssueIdentifier(team_key=match.group(1), number=int(match.group(2)))


def normalize_issue_identifier(issue_identifier: str) -> str:
    parsed = parse_issue_identifier(issue_identifier)
    return f"{parsed.team_key}-{parsed.number}"


def extract_issue_identifiers(*texts: str) -> list[str]:
    identifiers: list[str] = []
    for text in texts:
        for match in ISSUE_IDENTIFIER_PATTERN.finditer(text or ""):
            identifier = f"{match.group('prefix').upper()}-{match.group('number')}"
            if identifier not in identifiers:
                identifiers.append(identifier)
    return identifiers


def render_issue_link_marker(issue_identifier: str) -> str:
    return f"<!-- gpu-cfd-linear-issue: {normalize_issue_identifier(issue_identifier)} -->"


def extract_linked_issue_identifier(*texts: str) -> str | None:
    for text in texts:
        match = LINKED_ISSUE_MARKER_PATTERN.search(text or "")
        if match is None:
            continue
        return normalize_issue_identifier(match.group("identifier"))
    return None


def extract_closing_issue_identifier(*texts: str) -> str | None:
    for text in texts:
        match = CLOSING_ISSUE_PATTERN.search(text or "")
        if match is None:
            continue
        return normalize_issue_identifier(match.group("identifier"))
    return None


def extract_branch_issue_identifier(branch_name: str | None) -> str | None:
    match = BRANCH_ISSUE_PATTERN.search(branch_name or "")
    if match is None:
        return None
    return normalize_issue_identifier(match.group("identifier"))


def _render_bullet_lines(items: list[str]) -> list[str]:
    lines: list[str] = []
    for item in items:
        stripped = item.strip()
        if not stripped:
            continue
        if stripped.startswith(("- ", "* ")):
            lines.append(stripped)
        else:
            lines.append(f"- {stripped}")
    return lines


def _dedupe_preserve_order(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for line in lines:
        normalized = line.strip()
        if not normalized or normalized in seen:
            continue
        deduped.append(line)
        seen.add(normalized)
    return deduped


def _build_section(title: str, lines: list[str]) -> list[str]:
    content = _dedupe_preserve_order(lines)
    if not content:
        content = ["- Add notes here as the work evolves."]
    return [f"## {title}", *content, ""]


def graphql(query: str, variables: dict[str, Any]) -> dict[str, Any]:
    token = os.environ.get("LINEAR_API_KEY")
    if not token:
        raise ValueError("LINEAR_API_KEY is required")

    request = urllib.request.Request(
        LINEAR_GRAPHQL_URL,
        data=json.dumps({"query": query, "variables": variables}).encode("utf-8"),
        headers={
            "Authorization": token,
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ValueError(
            f"Linear GraphQL request failed with {exc.code}: {detail}"
        ) from exc

    if payload.get("errors"):
        raise ValueError(f"Linear GraphQL error: {payload['errors']}")

    return payload["data"]


def fetch_issue(issue_identifier: str) -> dict[str, Any]:
    parsed = parse_issue_identifier(issue_identifier)
    data = graphql(
        ISSUE_BY_IDENTIFIER_QUERY,
        {"teamKey": parsed.team_key, "number": parsed.number},
    )
    nodes = data.get("issues", {}).get("nodes", [])
    if not nodes:
        raise ValueError(f"Linear issue not found: {issue_identifier}")
    return nodes[0]


def fetch_issue_comments(
    issue_identifier: str,
    *,
    after: str | None = None,
) -> dict[str, Any]:
    parsed = parse_issue_identifier(issue_identifier)
    data = graphql(
        ISSUE_COMMENTS_QUERY,
        {"teamKey": parsed.team_key, "number": parsed.number, "after": after},
    )
    nodes = data.get("issues", {}).get("nodes", [])
    if not nodes:
        raise ValueError(f"Linear issue not found: {issue_identifier}")
    return nodes[0]


def resolve_state_id(issue: dict[str, Any], target_state_name: str) -> str:
    for state in issue.get("team", {}).get("states", {}).get("nodes", []):
        if state.get("name") == target_state_name:
            return str(state["id"])
    raise ValueError(
        f"state {target_state_name!r} not found on team "
        f"{issue.get('team', {}).get('key', '')}"
    )


def update_issue_state(issue_identifier: str, target_state_name: str) -> dict[str, Any]:
    issue = fetch_issue(issue_identifier)
    current_state = issue.get("state", {}).get("name")
    if current_state == target_state_name:
        return {
            "changed": False,
            "issue": issue,
            "previous_state": current_state,
            "current_state": current_state,
        }

    state_id = resolve_state_id(issue, target_state_name)
    mutation_data = graphql(
        ISSUE_UPDATE_STATE_MUTATION,
        {"id": issue["id"], "stateId": state_id},
    )
    updated_issue = mutation_data["issueUpdate"]["issue"]
    return {
        "changed": True,
        "issue": updated_issue,
        "previous_state": current_state,
        "current_state": updated_issue.get("state", {}).get("name"),
    }


def update_issue(
    issue_identifier: str,
    *,
    target_state_name: str | object = _UNSET,
    description: str | object = _UNSET,
    parent_identifier: str | None | object = _UNSET,
    label_names: list[str] | object = _UNSET,
) -> dict[str, Any]:
    issue = fetch_issue(issue_identifier)
    variables: dict[str, Any] = {"id": issue["id"]}
    changed = False

    previous_state = str((issue.get("state") or {}).get("name") or "")
    previous_description = str(issue.get("description") or "")
    previous_parent_identifier = str(((issue.get("parent") or {}).get("identifier")) or "")
    previous_labels = [
        str(node.get("name") or "").strip()
        for node in (issue.get("labels") or {}).get("nodes", [])
        if str(node.get("name") or "").strip()
    ]
    mutation_input: dict[str, Any] = {}

    if target_state_name is not _UNSET:
        if previous_state != target_state_name:
            mutation_input["stateId"] = resolve_state_id(issue, str(target_state_name))
            changed = True
    if description is not _UNSET:
        if previous_description != str(description):
            mutation_input["description"] = description
            changed = True
    if parent_identifier is not _UNSET:
        normalized_parent = (
            normalize_issue_identifier(str(parent_identifier))
            if isinstance(parent_identifier, str) and parent_identifier.strip()
            else None
        )
        if previous_parent_identifier != (normalized_parent or ""):
            mutation_input["parentId"] = (
                fetch_issue(normalized_parent)["id"] if normalized_parent else None
            )
            changed = True
    if label_names is not _UNSET:
        normalized_labels = sorted({name.strip() for name in list(label_names) if name.strip()})
        if sorted(previous_labels) != normalized_labels:
            mutation_input["labelIds"] = resolve_label_ids(normalized_labels)
            changed = True

    if not changed:
        return {
            "changed": False,
            "issue": issue,
            "previous_state": previous_state,
            "current_state": previous_state,
            "previous_description": previous_description,
            "current_description": previous_description,
            "previous_parent_identifier": previous_parent_identifier or None,
            "current_parent_identifier": previous_parent_identifier or None,
            "previous_labels": previous_labels,
            "current_labels": previous_labels,
        }

    variables.update(mutation_input)
    mutation_data = graphql(build_issue_update_mutation(mutation_input.keys()), variables)
    updated_issue = mutation_data["issueUpdate"]["issue"]
    updated_parent_identifier = str(
        ((updated_issue.get("parent") or {}).get("identifier")) or ""
    )
    updated_labels = [
        str(node.get("name") or "").strip()
        for node in (updated_issue.get("labels") or {}).get("nodes", [])
        if str(node.get("name") or "").strip()
    ]
    return {
        "changed": True,
        "issue": updated_issue,
        "previous_state": previous_state,
        "current_state": updated_issue.get("state", {}).get("name"),
        "previous_description": previous_description,
        "current_description": str(updated_issue.get("description") or ""),
        "previous_parent_identifier": previous_parent_identifier or None,
        "current_parent_identifier": updated_parent_identifier or None,
        "previous_labels": previous_labels,
        "current_labels": updated_labels,
    }


def build_issue_update_mutation(input_keys: Any) -> str:
    field_types = {
        "stateId": "String",
        "description": "String",
        "parentId": "String",
        "labelIds": "[String!]",
    }
    ordered_keys = [key for key in field_types if key in set(input_keys)]
    variable_definitions = ["$id: String!"] + [
        f"${key}: {field_types[key]}" for key in ordered_keys
    ]
    input_fields = ", ".join(f"{key}: ${key}" for key in ordered_keys)
    return (
        "mutation("
        + ", ".join(variable_definitions)
        + ") {\n"
        + "  issueUpdate(id: $id, input: { "
        + input_fields
        + " }) {\n"
        + ISSUE_UPDATE_MUTATION_SELECTION
        + "  }\n"
        + "}\n"
    )


def update_issue_description(issue_identifier: str, description: str) -> dict[str, Any]:
    issue = fetch_issue(issue_identifier)
    current_description = str(issue.get("description") or "")
    if current_description == description:
        return {
            "changed": False,
            "issue": issue,
            "previous_description": current_description,
            "current_description": current_description,
        }

    mutation_data = graphql(
        ISSUE_UPDATE_DESCRIPTION_MUTATION,
        {"id": issue["id"], "description": description},
    )
    updated_issue = mutation_data["issueUpdate"]["issue"]
    return {
        "changed": True,
        "issue": updated_issue,
        "previous_description": current_description,
        "current_description": str(updated_issue.get("description") or ""),
    }


def create_issue(
    *,
    team_id: str,
    title: str,
    description: str,
    parent_id: str | None = None,
    state_id: str | None = None,
    priority: int | None = None,
    label_ids: list[str] | None = None,
) -> dict[str, Any]:
    data = graphql(
        ISSUE_CREATE_MUTATION,
        {
            "teamId": team_id,
            "title": title,
            "description": description,
            "parentId": parent_id,
            "stateId": state_id,
            "priority": priority,
            "labelIds": label_ids or [],
        },
    )
    return data["issueCreate"]["issue"]


def list_issue_comments(issue_identifier: str) -> list[dict[str, Any]]:
    comments: list[dict[str, Any]] = []
    after: str | None = None
    while True:
        issue = fetch_issue_comments(issue_identifier, after=after)
        comments_payload = issue.get("comments", {})
        comments.extend(list(comments_payload.get("nodes", [])))
        page_info = comments_payload.get("pageInfo", {})
        if not page_info.get("hasNextPage"):
            return comments
        next_cursor = page_info.get("endCursor")
        if not next_cursor:
            return comments
        after = str(next_cursor)


def list_team_issues(team_key: str) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    after: str | None = None
    while True:
        data = graphql(TEAM_ISSUES_QUERY, {"teamKey": team_key.upper(), "after": after})
        issues_payload = data.get("issues", {})
        issues.extend(list(issues_payload.get("nodes", [])))
        page_info = issues_payload.get("pageInfo", {})
        if not page_info.get("hasNextPage"):
            return issues
        next_cursor = page_info.get("endCursor")
        if not next_cursor:
            return issues
        after = str(next_cursor)


def create_comment(issue_identifier: str, body: str) -> dict[str, Any]:
    issue = fetch_issue(issue_identifier)
    data = graphql(COMMENT_CREATE_MUTATION, {"issueId": issue["id"], "body": body})
    return data["commentCreate"]["comment"]


def update_comment(comment_id: str, body: str) -> dict[str, Any]:
    data = graphql(COMMENT_UPDATE_MUTATION, {"id": comment_id, "body": body})
    return data["commentUpdate"]["comment"]


def find_workpad_comment(issue_identifier: str) -> dict[str, Any] | None:
    canonical_comments: list[dict[str, Any]] = []
    legacy_comments: list[dict[str, Any]] = []
    for comment in list_issue_comments(issue_identifier):
        body = (comment.get("body") or "").strip()
        if WORKPAD_MARKER in body:
            canonical_comments.append(comment)
            continue
        if is_legacy_workpad_body(body):
            legacy_comments.append(comment)
    matched = canonical_comments or legacy_comments
    if not matched:
        return None
    matched.sort(
        key=lambda comment: (
            comment.get("updatedAt") or "",
            comment.get("createdAt") or "",
        )
    )
    return matched[-1]


def list_workpad_comments(issue_identifier: str) -> list[dict[str, Any]]:
    matched: list[dict[str, Any]] = []
    for comment in list_issue_comments(issue_identifier):
        body = (comment.get("body") or "").strip()
        if WORKPAD_MARKER in body or is_legacy_workpad_body(body):
            matched.append(comment)
    matched.sort(
        key=lambda comment: (
            comment.get("updatedAt") or "",
            comment.get("createdAt") or "",
        )
    )
    return matched


def is_legacy_workpad_body(body: str) -> bool:
    normalized = body.strip()
    return any(marker in normalized for marker in LEGACY_WORKPAD_MARKERS)


def render_workpad_body(
    *,
    issue_identifier: str,
    issue_title: str,
    current_status: str,
    task_summary: list[str] | None = None,
    execution_plan: list[str] | None = None,
    scoped_sources: list[str] | None = None,
    decisions_and_rationale: list[str] | None = None,
    validation: list[str] | None = None,
    risks_blockers: list[str] | None = None,
    review_handoff_notes: list[str] | None = None,
) -> str:
    task_summary_lines = [
        f"- Issue: {issue_identifier}",
        f"- Title: {issue_title}",
        "- Capture the scoped objective, branch/PR context, and any non-obvious constraints.",
    ]
    task_summary_lines.extend(_render_bullet_lines(task_summary or []))

    current_status_lines = [
        f"- Current status: {current_status}",
        "- Update this section as the task moves through planning, implementation, review, and rework.",
    ]

    execution_plan_lines = _render_bullet_lines(execution_plan or [])
    execution_plan_lines.extend(
        [
            "- Resolve the exact PR card before opening broader docs.",
            "- Record assumptions and scope limits before code edits.",
        ]
    )

    scoped_sources_lines = _render_bullet_lines(scoped_sources or [])
    scoped_sources_lines.extend(
        [
            "- Start with AGENTS.md, the issue context, and the exact task file and PR card.",
            "- Use docs/tasks/pr_inventory.md only if the PR ID, task file, or card location is unclear.",
            "- Expand outward only when the cited sources are insufficient.",
        ]
    )

    decisions_lines = _render_bullet_lines(decisions_and_rationale or [])
    decisions_lines.extend(
        [
            "- Tried X, rejected because Y in the task card/spec.",
            "- Boundary behavior differs from an earlier assumption.",
        ]
    )

    validation_lines = _render_bullet_lines(validation or [])
    validation_lines.extend(
        [
            "- Record the smallest direct validation first, then broader checks if needed.",
        ]
    )

    risks_lines = _render_bullet_lines(risks_blockers or [])
    risks_lines.extend(
        [
            "- Missing auth, missing credentials, or conflicting authority docs should be called out here.",
        ]
    )

    review_lines = _render_bullet_lines(review_handoff_notes or [])
    review_lines.extend(
        [
            "- Important for Rework: Devin comment likely points at file Z.",
            "- Future agent note: do not widen this scope without a cited authority update.",
        ]
    )

    lines = [
        WORKPAD_MARKER,
        "",
        WORKPAD_TITLE,
        "",
    ]
    lines.extend(_build_section("Task Summary", task_summary_lines))
    lines.extend(_build_section("Current Status", current_status_lines))
    lines.extend(_build_section("Execution Plan", execution_plan_lines))
    lines.extend(_build_section("Scoped Sources", scoped_sources_lines))
    lines.extend(_build_section("Decisions and Rationale", decisions_lines))
    lines.extend(_build_section("Validation", validation_lines))
    lines.extend(_build_section("Risks / Blockers", risks_lines))
    lines.extend(_build_section("Review / Handoff Notes", review_lines))
    return "\n".join(lines).strip() + "\n"


def _parse_workpad_sections(body: str) -> dict[str, list[str]]:
    if WORKPAD_MARKER not in body:
        return {}

    sections: dict[str, list[str]] = {}
    current_section: str | None = None
    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        if line == WORKPAD_MARKER or line == WORKPAD_TITLE:
            continue
        if line.startswith("## "):
            current_section = line[3:].strip()
            sections.setdefault(current_section, [])
            continue
        if current_section is None:
            continue
        sections[current_section].append(line)
    return sections


def _merge_section_lines(
    existing_lines: list[str], new_lines: list[str], *, replace_status: str | None = None
) -> list[str]:
    merged = [line for line in existing_lines if line.strip()]
    if replace_status is not None:
        merged = [
            line
            for line in merged
            if not line.strip().startswith("- Current status:")
        ]
        merged.insert(0, f"- Current status: {replace_status}")
    merged.extend(line for line in new_lines if line.strip())
    return _dedupe_preserve_order(merged)


def merge_workpad_body(
    existing_body: str | None,
    *,
    issue_identifier: str,
    issue_title: str,
    current_status: str,
    task_summary: list[str] | None = None,
    execution_plan: list[str] | None = None,
    scoped_sources: list[str] | None = None,
    decisions_and_rationale: list[str] | None = None,
    validation: list[str] | None = None,
    risks_blockers: list[str] | None = None,
    review_handoff_notes: list[str] | None = None,
) -> str:
    if not existing_body or WORKPAD_MARKER not in existing_body:
        return render_workpad_body(
            issue_identifier=issue_identifier,
            issue_title=issue_title,
            current_status=current_status,
            task_summary=task_summary,
            execution_plan=execution_plan,
            scoped_sources=scoped_sources,
            decisions_and_rationale=decisions_and_rationale,
            validation=validation,
            risks_blockers=risks_blockers,
            review_handoff_notes=review_handoff_notes,
        )

    existing_sections = _parse_workpad_sections(existing_body)
    section_updates = {
        "Task Summary": _render_bullet_lines(task_summary or []),
        "Current Status": [],
        "Execution Plan": _render_bullet_lines(execution_plan or []),
        "Scoped Sources": _render_bullet_lines(scoped_sources or []),
        "Decisions and Rationale": _render_bullet_lines(decisions_and_rationale or []),
        "Validation": _render_bullet_lines(validation or []),
        "Risks / Blockers": _render_bullet_lines(risks_blockers or []),
        "Review / Handoff Notes": _render_bullet_lines(review_handoff_notes or []),
    }

    merged_sections: dict[str, list[str]] = {}
    for section_name in WORKPAD_SECTIONS:
        existing_lines = existing_sections.get(section_name, [])
        replace_status = current_status if section_name == "Current Status" else None
        merged_sections[section_name] = _merge_section_lines(
            existing_lines,
            section_updates[section_name],
            replace_status=replace_status,
        )

    lines = [
        WORKPAD_MARKER,
        "",
        WORKPAD_TITLE,
        "",
    ]
    for section_name in WORKPAD_SECTIONS:
        lines.extend(_build_section(section_name, merged_sections[section_name]))
    return "\n".join(lines).strip() + "\n"


def upsert_workpad_comment(issue_identifier: str, body: str) -> dict[str, Any]:
    existing_comment = find_workpad_comment(issue_identifier)
    previous_body = (
        str(existing_comment.get("body") or "")
        if existing_comment is not None
        else None
    )
    if existing_comment is None:
        created = create_comment(issue_identifier, body)
        result = {"action": "created", **created}
    else:
        updated = update_comment(str(existing_comment["id"]), body)
        result = {"action": "updated", **updated}
    try:
        suppress_legacy_workpad_comments(issue_identifier)
    except Exception:
        pass
    if trace is not None and trace.is_enabled():
        context = trace.current_trace_context()
        if context["run_id"]:
            trace.capture_workpad_revision(
                issue_id=issue_identifier,
                run_id=str(context["run_id"]),
                previous_body=previous_body,
                current_body=body,
                action=str(result["action"]),
                comment_id=str(result.get("id") or ""),
                comment_url=str(result.get("url") or ""),
            )
    return result


def suppress_legacy_workpad_comments(issue_identifier: str) -> list[dict[str, Any]]:
    canonical_comment = None
    legacy_comments: list[dict[str, Any]] = []
    for comment in list_workpad_comments(issue_identifier):
        body = str(comment.get("body") or "")
        if WORKPAD_MARKER in body:
            canonical_comment = comment
            continue
        if is_legacy_workpad_body(body):
            legacy_comments.append(comment)

    if canonical_comment is None:
        return []

    suppressed: list[dict[str, Any]] = []
    canonical_url = str(canonical_comment.get("url") or "").strip()
    replacement = "\n".join(
        [
            LEGACY_WORKPAD_SUPERSEDED_MARKER,
            "",
            "# Workpad Superseded",
            "",
            (
                f"Superseded by the canonical worker workpad: {canonical_url}"
                if canonical_url
                else "Superseded by the canonical worker workpad comment."
            ),
            "",
        ]
    )
    for legacy_comment in legacy_comments:
        suppressed.append(update_comment(str(legacy_comment["id"]), replacement))
    return suppressed


def list_issue_labels() -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    after: str | None = None
    while True:
        data = graphql(ISSUE_LABELS_QUERY, {"after": after})
        payload = data.get("issueLabels", {})
        labels.extend(list(payload.get("nodes", [])))
        page_info = payload.get("pageInfo", {})
        if not page_info.get("hasNextPage"):
            return labels
        next_cursor = page_info.get("endCursor")
        if not next_cursor:
            return labels
        after = str(next_cursor)


def resolve_label_ids(label_names: list[str]) -> list[str]:
    requested = {name.strip() for name in label_names if name.strip()}
    if not requested:
        return []
    resolved: list[str] = []
    for label in list_issue_labels():
        name = str(label.get("name") or "").strip()
        if name in requested:
            resolved.append(str(label["id"]))
    return resolved


def render_residual_followup_marker(parent_identifier: str, fingerprint: str) -> str:
    return (
        "<!-- gpu-cfd-residual-followup:v1 "
        f"parent={normalize_issue_identifier(parent_identifier)} "
        f"fingerprint={fingerprint} -->"
    )


def extract_residual_followup_marker(description: str) -> dict[str, str] | None:
    match = RESIDUAL_FOLLOWUP_MARKER_PATTERN.search(description or "")
    if match is None:
        return None
    return {
        "parent": normalize_issue_identifier(match.group("parent")),
        "fingerprint": str(match.group("fingerprint")).lower(),
    }


def render_phase_cleanup_marker(section_slug: str) -> str:
    return f"<!-- gpu-cfd-phase-cleanup:v1 section={section_slug.strip().lower()} -->"


def extract_phase_cleanup_marker(description: str) -> dict[str, str] | None:
    match = PHASE_CLEANUP_MARKER_PATTERN.search(description or "")
    if match is None:
        return None
    return {"section": str(match.group("section")).strip().lower()}


def residual_followup_fingerprint(
    *,
    parent_identifier: str,
    priority_label: str,
    title: str,
    body: str,
    path: str | None = None,
    start_line: int | None = None,
) -> str:
    normalized_parts = [
        normalize_issue_identifier(parent_identifier),
        priority_label.strip(),
        title.strip(),
        body.strip(),
        (path or "").strip(),
        str(start_line or ""),
    ]
    digest = hashlib.sha256("\n".join(normalized_parts).encode("utf-8")).hexdigest()
    return digest[:12]


def find_residual_followup_issue(
    team_key: str,
    *,
    fingerprint: str,
) -> dict[str, Any] | None:
    for issue in list_team_issues(team_key):
        marker = extract_residual_followup_marker(str(issue.get("description") or ""))
        if marker and marker["fingerprint"] == fingerprint.lower():
            return issue
    return None


def render_residual_followup_description(
    *,
    parent_identifier: str,
    parent_title: str,
    fingerprint: str,
    priority_label: str,
    finding_title: str,
    finding_body: str,
    pr_url: str,
    branch: str,
    commit: str,
    artifact_path: str,
    source_path: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    location_lines: list[str] = []
    if source_path:
        location = source_path
        if start_line is not None:
            location += f":{start_line}"
            if end_line is not None and end_line != start_line:
                location += f"-{end_line}"
        location_lines.append(f"- Source location: `{location}`")
    return "\n".join(
        [
            render_residual_followup_marker(parent_identifier, fingerprint),
            "",
            f"Residual local-review finding deferred from `{parent_identifier}`.",
            "",
            "## Parent Context",
            f"- Parent issue: `{normalize_issue_identifier(parent_identifier)}`",
            f"- Parent title: {parent_title}",
            f"- Pull request: {pr_url}",
            f"- Branch: `{branch}`",
            f"- Commit: `{commit}`",
            f"- Review artifact: `{artifact_path}`",
            "",
            "## Finding",
            f"- Priority: `{priority_label}`",
            f"- Title: {finding_title}",
            *location_lines,
            "",
            "## Review Excerpt",
            finding_body.strip(),
            "",
        ]
    ).strip() + "\n"


def ensure_residual_followup_issue(
    *,
    parent_identifier: str,
    parent_title: str,
    finding_title: str,
    finding_body: str,
    priority: int,
    priority_label: str,
    pr_url: str,
    branch: str,
    commit: str,
    artifact_path: str,
    source_path: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    parent_issue = fetch_issue(parent_identifier)
    team_key = str(parent_issue.get("team", {}).get("key") or "")
    fingerprint = residual_followup_fingerprint(
        parent_identifier=parent_identifier,
        priority_label=priority_label,
        title=finding_title,
        body=finding_body,
        path=source_path,
        start_line=start_line,
    )
    existing = find_residual_followup_issue(team_key, fingerprint=fingerprint)
    if existing is not None:
        return {"action": "existing", **existing}

    backlog_state_id = resolve_state_id(parent_issue, "Backlog")
    effective_label_names = list(label_names or [])
    if not effective_label_names:
        if priority_label == "P0":
            effective_label_names = ["Bug"]
        elif priority_label in {"P1", "P2"}:
            effective_label_names = ["Bug"]
        else:
            effective_label_names = ["Improvement"]
    label_ids = resolve_label_ids(effective_label_names)
    created = create_issue(
        team_id=str(parent_issue["team"]["id"]),
        title=(
            f"Residual review follow-up: {normalize_issue_identifier(parent_identifier)} "
            f"[{priority_label}] {finding_title}"
        ),
        description=render_residual_followup_description(
            parent_identifier=parent_identifier,
            parent_title=parent_title,
            fingerprint=fingerprint,
            priority_label=priority_label,
            finding_title=finding_title,
            finding_body=finding_body,
            pr_url=pr_url,
            branch=branch,
            commit=commit,
            artifact_path=artifact_path,
            source_path=source_path,
            start_line=start_line,
            end_line=end_line,
        ),
        parent_id=str(parent_issue["id"]),
        state_id=backlog_state_id,
        priority=priority,
        label_ids=label_ids,
    )
    return {"action": "created", **created}


def fetch_direct_blocked_dependents(issue_identifier: str) -> list[dict[str, Any]]:
    parsed = parse_issue_identifier(issue_identifier)
    data = graphql(
        DIRECT_BLOCKS_QUERY,
        {"teamKey": parsed.team_key, "number": parsed.number},
    )
    issue_nodes = data.get("issues", {}).get("nodes", [])
    if not issue_nodes:
        raise ValueError(f"Linear issue not found: {issue_identifier}")

    dependents: list[dict[str, Any]] = []
    for relation in issue_nodes[0].get("relations", {}).get("nodes", []):
        if relation.get("type") != "blocks":
            continue
        related_issue = relation.get("relatedIssue")
        if isinstance(related_issue, dict):
            dependents.append(related_issue)
    return dependents


def dependent_is_unblocked(dependent_issue: dict[str, Any]) -> bool:
    inverse_relations = dependent_issue.get("inverseRelations", {}).get("nodes", [])
    blockers = [
        relation
        for relation in inverse_relations
        if relation.get("type") == "blocks"
        and isinstance(relation.get("issue"), dict)
    ]
    if not blockers:
        return True
    return all(
        relation["issue"].get("state", {}).get("name") == "Done"
        for relation in blockers
    )


def release_direct_unblocked_dependents(
    issue_identifier: str,
    *,
    source_state: str = "Backlog",
    target_state: str = "Todo",
) -> list[dict[str, Any]]:
    promoted: list[dict[str, Any]] = []
    for dependent in fetch_direct_blocked_dependents(issue_identifier):
        current_state = dependent.get("state", {}).get("name")
        identifier = dependent.get("identifier")
        if current_state != source_state or not isinstance(identifier, str):
            continue
        if not dependent_is_unblocked(dependent):
            continue
        update_result = update_issue_state(identifier, target_state)
        promoted.append(
            {
                "identifier": identifier,
                "previous_state": update_result["previous_state"],
                "current_state": update_result["current_state"],
                "changed": update_result["changed"],
            }
        )
    return promoted
