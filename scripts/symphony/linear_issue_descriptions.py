#!/usr/bin/env python3
"""Render canonical Linear issue descriptions from the repo task docs."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from dataclasses import dataclass

try:
    from scripts.symphony import codex_dispatch
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from scripts.symphony import codex_dispatch


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
        f"* Section: {metadata.section_label}\n"
        f"* Canonical backlog dependencies: {format_dependencies(metadata.depends_on)}\n"
        f"* Linear issue identifier: `{issue_identifier}`\n\n"
        "## Task card\n\n"
        f"{card_markdown}\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    rendered = {
        issue_id: render_issue_description(repo, issue_id, issue_title)
        for issue_id, issue_title in load_issue_specs(args)
    }
    body = json.dumps(rendered, indent=2) + "\n"
    if args.output:
        pathlib.Path(args.output).write_text(body, encoding="utf-8")
    else:
        print(body, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
