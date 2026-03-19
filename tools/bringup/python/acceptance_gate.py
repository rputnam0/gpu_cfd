#!/usr/bin/env python3
"""Compatibility shim for the Phase 1 acceptance report entrypoint."""

from __future__ import annotations

import pathlib
import sys


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3]


sys.path.append(str(_repo_root()))

from scripts.authority.phase1_acceptance import main


if __name__ == "__main__":
    raise SystemExit(main(["report", *sys.argv[1:]]))
