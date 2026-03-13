#!/usr/bin/env python3
"""Launch Codex with a repo-owned runtime profile."""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

try:
    from scripts.symphony import runtime_config
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from scripts.symphony import runtime_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "profile", help="Runtime profile from scripts/symphony/runtime_config.toml."
    )
    parser.add_argument(
        "codex_args", nargs=argparse.REMAINDER, help="Arguments forwarded to codex."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.codex_args:
        raise SystemExit(
            "expected a codex subcommand, for example: implementation app-server"
        )
    command = runtime_config.build_codex_command(args.profile, list(args.codex_args))
    os.execv(command[0], command)


if __name__ == "__main__":
    sys.exit(main())
