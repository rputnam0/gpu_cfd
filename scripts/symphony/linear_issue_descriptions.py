#!/usr/bin/env python3
"""Render, audit, and sync canonical Linear issue descriptions from repo task docs."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Any

try:
    from scripts.symphony import codex_dispatch, linear_api
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from scripts.symphony import codex_dispatch, linear_api


GITHUB_BLOB_ROOT = "https://github.com/rputnam0/gpu_cfd/blob/main"
TASK_CARD_ID_PATTERN = re.compile(r"\b(?:FND-\d+|P\d+-\d+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class IssueMetadata:
    pr_id: str
    title: str
    section_name: str
    section_label: str
    task_file: pathlib.Path
    scope: str
    done_when: str
    depends_on: list[str]


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def normalize_section_label(section_name: str) -> str:
    if "/" in section_name:
        return section_name.split("/", 1)[0].strip()
    if "—" in section_name:
        return section_name.split("—", 1)[0].strip()
    return section_name.strip()


def infer_pr_id(issue_title: str) -> str:
    match = TASK_CARD_ID_PATTERN.search(issue_title or "")
    if not match:
        raise ValueError(f"could not infer task card id from issue title: {issue_title!r}")
    return match.group(0).upper()


def load_issue_metadata(root: pathlib.Path | None = None) -> dict[str, IssueMetadata]:
    repo = (root or repo_root()).resolve()
    backlog = json.loads((repo / "docs" / "backlog" / "gpu_cfd_pr_backlog.json").read_text(encoding="utf-8"))
    inventory = codex_dispatch.parse_pr_inventory(repo)
    metadata: dict[str, IssueMetadata] = {}
    for section in backlog["sections"]:
        section_name = str(section["section"])
        section_label = normalize_section_label(section_name)
        for pr in section["prs"]:
            pr_id = str(pr["id"]).upper()
            task_file = inventory.get(pr_id)
            if task_file is None:
                raise ValueError(f"no owning task file found for {pr_id}")
            metadata[pr_id] = IssueMetadata(
                pr_id=pr_id,
                title=f"{pr_id}: {pr['title']}",
                section_name=section_name,
                section_label=section_label,
                task_file=task_file,
                scope=str(pr["scope"]),
                done_when=str(pr["done_when"]),
                depends_on=[str(item).upper() for item in pr.get("depends_on", [])],
            )
    return metadata


def github_doc_link(repo: pathlib.Path, path: pathlib.Path) -> str:
    relative = path.resolve().relative_to(repo.resolve()).as_posix()
    return f"[{relative}](<{GITHUB_BLOB_ROOT}/{relative}>)"


def format_dependencies(depends_on: list[str]) -> str:
    if not depends_on:
        return "None"
    return ", ".join(f"`{dependency}`" for dependency in depends_on)


def normalize_description_text(body: str) -> str:
    normalized_lines: list[str] = []
    for raw_line in body.strip().splitlines():
        line = raw_line.rstrip()
        bullet_match = re.match(r"^(?P<indent>\s*)\*\s+", line)
        if bullet_match is not None:
            line = re.sub(r"^(?P<indent>\s*)\*\s+", r"\g<indent>- ", line, count=1)
        normalized_lines.append(line)
    return "\n".join(normalized_lines).strip()


def render_issue_description(
    root: pathlib.Path | None,
    issue_identifier: str,
    issue_title: str,
) -> str:
    repo = (root or repo_root()).resolve()
    pr_id = infer_pr_id(issue_title)
    metadata = load_issue_metadata(repo)[pr_id]
    task_text = metadata.task_file.read_text(encoding="utf-8")
    card_markdown = codex_dispatch.extract_card_markdown(task_text, pr_id).strip()
    if not card_markdown:
        raise ValueError(f"task card {pr_id} not found in {metadata.task_file}")

    docs_readme = repo / "docs" / "README.md"
    tasks_readme = repo / "docs" / "tasks" / "README.md"
    authority_readme = repo / "docs" / "authority" / "README.md"
    pr_inventory = repo / "docs" / "tasks" / "pr_inventory.md"
    backlog_json = repo / "docs" / "backlog" / "gpu_cfd_pr_backlog.json"
    backlog_md = repo / "docs" / "backlog" / "gpu_cfd_pr_backlog.md"

    return (
        f"Execution task for backlog item `{pr_id}`. Linear is the operational tracker for this work; "
        "the repo documents below remain the technical source of truth.\n\n"
        "## Worker startup contract\n\n"
        "* Start with `AGENTS.md`.\n"
        "* Read this Linear issue and any attached PR/review context.\n"
        "* Open the owning task file and exact PR card.\n"
        "* Read only the sources cited by that card plus any directly relevant review comments or boundary docs.\n"
        f"* If the PR ID, task file, or card location is unclear, use `{pr_inventory.relative_to(repo).as_posix()}` as the fallback map.\n"
        "* Treat the PR card below as the execution contract; broader specs are supporting context only.\n"
        "* Write or update the canonical Linear workpad before edits.\n\n"
        "## Source docs\n\n"
        f"* Task card: {github_doc_link(repo, metadata.task_file)}\n"
        f"* PR inventory: {github_doc_link(repo, pr_inventory)}\n"
        f"* Backlog JSON: {github_doc_link(repo, backlog_json)}\n"
        f"* Backlog overview: {github_doc_link(repo, backlog_md)}\n"
        f"* Docs index: {github_doc_link(repo, docs_readme)}\n"
        f"* Tasks operating model: {github_doc_link(repo, tasks_readme)}\n"
        f"* Authority index: {github_doc_link(repo, authority_readme)}\n\n"
        "## Board gate\n\n"
        "* Keep this issue in `Backlog` until all Linear blockers are `Done`, then move it to `Todo`.\n"
        "* Move to `In Progress` only after implementation begins on a branch.\n"
        "* If pre-PR local review findings remain, `scripts/symphony/pr_handoff.py` parks the issue in `Ready to Merge` while the current worker fixes them; Symphony must not redispatch a fresh implementation worker from that state.\n"
        f"* Section: {metadata.section_label}\n"
        f"* Canonical backlog dependencies: {format_dependencies(metadata.depends_on)}\n"
        f"* Linear issue identifier: `{issue_identifier}`\n\n"
        "## Task card\n\n"
        f"{card_markdown}\n"
    )


def audit_live_issue_descriptions(
    root: pathlib.Path | None,
    *,
    team_key: str = "PRO",
) -> list[dict[str, Any]]:
    repo = (root or repo_root()).resolve()
    live_issues = linear_api.list_team_issues(team_key)
    results: list[dict[str, Any]] = []
    for issue in live_issues:
        issue_identifier = str(issue.get("identifier") or "")
        issue_title = str(issue.get("title") or "")
        live_description = str(issue.get("description") or "")
        try:
            pr_id = infer_pr_id(issue_title)
            expected_description = render_issue_description(
                repo,
                issue_identifier,
                issue_title,
            )
        except ValueError as exc:
            results.append(
                {
                    "issue_identifier": issue_identifier,
                    "issue_title": issue_title,
                    "status": "skipped",
                    "changed": False,
                    "reason": str(exc),
                }
            )
            continue
        changed = (
            normalize_description_text(live_description)
            != normalize_description_text(expected_description)
        )
        results.append(
            {
                "issue_identifier": issue_identifier,
                "issue_title": issue_title,
                "pr_id": pr_id,
                "status": "drifted" if changed else "in_sync",
                "changed": changed,
                "live_description": live_description,
                "expected_description": expected_description,
            }
        )
    return results


def sync_live_issue_descriptions(
    root: pathlib.Path | None,
    *,
    team_key: str = "PRO",
) -> list[dict[str, Any]]:
    results = audit_live_issue_descriptions(root, team_key=team_key)
    synced_results: list[dict[str, Any]] = []
    for result in results:
        if result["status"] != "drifted":
            synced_results.append(result)
            continue
        update_result = linear_api.update_issue_description(
            str(result["issue_identifier"]),
            str(result["expected_description"]),
        )
        synced_results.append(
            {
                **result,
                "status": "updated" if update_result["changed"] else "in_sync",
                "update_result": {
                    "changed": update_result["changed"],
                    "previous_description": update_result["previous_description"],
                    "current_description": update_result["current_description"],
                },
            }
        )
    return synced_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("render", "audit", "sync"),
        default="render",
        help="Render issue descriptions, audit live Linear descriptions, or sync them.",
    )
    parser.add_argument(
        "--team",
        default="PRO",
        help="Linear team key for audit/sync modes. Defaults to PRO.",
    )
    parser.add_argument(
        "--issue",
        nargs=2,
        action="append",
        metavar=("ISSUE_ID", "ISSUE_TITLE"),
        help="Render a description for the given Linear issue id and title.",
    )
    parser.add_argument(
        "--issues-json",
        help="Path to a JSON file containing [{'id': ..., 'title': ...}, ...].",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the rendered descriptions as JSON.",
    )
    return parser.parse_args()


def load_issue_specs(args: argparse.Namespace) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    if args.issue:
        specs.extend((issue_id, issue_title) for issue_id, issue_title in args.issue)
    if args.issues_json:
        payload = json.loads(pathlib.Path(args.issues_json).read_text(encoding="utf-8"))
        specs.extend((str(item["id"]), str(item["title"])) for item in payload)
    if not specs:
        raise SystemExit("provide at least one --issue pair or --issues-json")
    return specs


def main() -> int:
    args = parse_args()
    repo = repo_root()
    if args.mode == "render":
        payload: Any = {
            issue_id: render_issue_description(repo, issue_id, issue_title)
            for issue_id, issue_title in load_issue_specs(args)
        }
        exit_code = 0
    elif args.mode == "audit":
        payload = audit_live_issue_descriptions(repo, team_key=args.team)
        exit_code = 1 if any(result["status"] == "drifted" for result in payload) else 0
    else:
        payload = sync_live_issue_descriptions(repo, team_key=args.team)
        exit_code = 0
    body = json.dumps(payload, indent=2) + "\n"
    if args.output:
        pathlib.Path(args.output).write_text(body, encoding="utf-8")
    else:
        print(body, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
