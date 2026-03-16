#!/usr/bin/env python3
"""Small Linear GraphQL helpers for the gpu_cfd Symphony workflow."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

LINEAR_GRAPHQL_URL = "https://api.linear.app/graphql"
ISSUE_IDENTIFIER_PATTERN = re.compile(
    r"\b(?P<prefix>PRO)-(?P<number>\d+)\b", re.IGNORECASE
)

ISSUE_BY_IDENTIFIER_QUERY = """
query($teamKey: String!, $number: Float!) {
  issues(filter: { team: { key: { eq: $teamKey } }, number: { eq: $number } }) {
    nodes {
      id
      identifier
      title
      state {
        id
        name
      }
      team {
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


def extract_issue_identifiers(*texts: str) -> list[str]:
    identifiers: list[str] = []
    for text in texts:
        for match in ISSUE_IDENTIFIER_PATTERN.finditer(text or ""):
            identifier = f"{match.group('prefix').upper()}-{match.group('number')}"
            if identifier not in identifiers:
                identifiers.append(identifier)
    return identifiers


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
