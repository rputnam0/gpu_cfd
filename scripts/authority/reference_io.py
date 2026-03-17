"""Phase 0 reference I/O normalization helpers."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shlex
from typing import Any, Mapping


REFERENCE_IO_STAGE_NAME = "reference_io_normalization"
REFERENCE_IO_STAGE_KIND = "compare-prep"
REFERENCE_IO_OVERLAY_ARTIFACT = "reference_freeze_overlay.json"
REFERENCE_IO_COMPARISON_SCOPE = "same_baseline_rerun"
REFERENCE_IO_CONTROL_DICT = "system/controlDict"

REFERENCE_IO_POLICY = {
    "write_format": "ascii",
    "write_compression": "off",
    "write_precision": 12,
    "time_precision": 12,
}

_CONTROL_DICT_DIRECTIVES = (
    ("writeFormat", "write_format"),
    ("writeCompression", "write_compression"),
    ("writePrecision", "write_precision"),
    ("timePrecision", "time_precision"),
)

_CANONICAL_DIRECTIVE_LINES = {
    "writeFormat": "writeFormat     ascii;",
    "writeCompression": "writeCompression off;",
    "writePrecision": "writePrecision  12;",
    "timePrecision": "timePrecision   12;",
}


def build_reference_io_normalization_payload(
    *,
    overlay_artifact: str = REFERENCE_IO_OVERLAY_ARTIFACT,
) -> dict[str, Any]:
    return {
        "stage_name": REFERENCE_IO_STAGE_NAME,
        "stage_kind": REFERENCE_IO_STAGE_KIND,
        "overlay_artifact": overlay_artifact,
        "comparison_scope": REFERENCE_IO_COMPARISON_SCOPE,
        "preserves_numerics": True,
        "policy": dict(REFERENCE_IO_POLICY),
    }


def reference_io_overlay_command(
    *,
    case_dir: str = ".",
    json_out: str = REFERENCE_IO_OVERLAY_ARTIFACT,
) -> str:
    script_path = pathlib.Path(__file__).resolve()
    return " ".join(
        [
            "python3",
            shlex.quote(script_path.as_posix()),
            "--case-dir",
            shlex.quote(case_dir),
            "--json-out",
            shlex.quote(json_out),
        ]
    )


def reference_io_overlay_stage(
    *,
    overlay_artifact: str = REFERENCE_IO_OVERLAY_ARTIFACT,
) -> dict[str, Any]:
    return {
        "name": REFERENCE_IO_STAGE_NAME,
        "cmd": reference_io_overlay_command(json_out=overlay_artifact),
        "cwd": ".",
        "stage_kind": REFERENCE_IO_STAGE_KIND,
        "overlay_artifact": overlay_artifact,
    }


def apply_reference_io_overlay(
    case_dir: pathlib.Path | str,
    *,
    json_out: pathlib.Path | str | None = None,
    control_dict_relative_path: str = REFERENCE_IO_CONTROL_DICT,
) -> dict[str, Any]:
    case_dir_path = pathlib.Path(case_dir)
    control_dict_path = case_dir_path / control_dict_relative_path
    if not control_dict_path.exists():
        raise FileNotFoundError(f"missing OpenFOAM controlDict: {control_dict_path}")

    original_lines = control_dict_path.read_text(encoding="utf-8").splitlines()
    changes: dict[str, dict[str, Any]] = {
        foam_key: {"before": None, "after": _policy_value_string(policy_key)}
        for foam_key, policy_key in _CONTROL_DICT_DIRECTIVES
    }

    rewritten_lines: list[str] = []
    seen_keys: set[str] = set()
    for line in original_lines:
        matched_key = None
        for foam_key, policy_key in _CONTROL_DICT_DIRECTIVES:
            pattern = re.compile(
                rf"^(?P<indent>\s*){re.escape(foam_key)}\s+(?P<value>[^;]+);(?P<suffix>\s*(?://.*)?)$"
            )
            match = pattern.match(line)
            if match is None:
                continue
            matched_key = foam_key
            if changes[foam_key]["before"] is None:
                changes[foam_key]["before"] = match.group("value").strip()
            suffix = match.group("suffix") or ""
            rewritten_lines.append(_CANONICAL_DIRECTIVE_LINES[foam_key] + suffix)
            seen_keys.add(foam_key)
            break
        if matched_key is None:
            rewritten_lines.append(line)

    for foam_key, _policy_key in _CONTROL_DICT_DIRECTIVES:
        if foam_key not in seen_keys:
            rewritten_lines.append(_CANONICAL_DIRECTIVE_LINES[foam_key])

    control_dict_path.write_text("\n".join(rewritten_lines) + "\n", encoding="utf-8")

    overlay_artifact = (
        pathlib.Path(json_out).name if json_out is not None else REFERENCE_IO_OVERLAY_ARTIFACT
    )
    payload = build_reference_io_normalization_payload(overlay_artifact=overlay_artifact)
    payload["control_dict"] = control_dict_relative_path
    payload["changes"] = changes

    if json_out is not None:
        json_out_path = pathlib.Path(json_out)
        if not json_out_path.is_absolute():
            json_out_path = case_dir_path / json_out_path
        json_out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True, help="Case directory containing system/controlDict")
    parser.add_argument(
        "--json-out",
        help="Optional JSON output path for the emitted reference I/O overlay metadata",
    )
    args = parser.parse_args(argv)
    payload = apply_reference_io_overlay(
        args.case_dir,
        json_out=args.json_out,
    )
    if args.json_out is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _policy_value_string(policy_key: str) -> str:
    value = REFERENCE_IO_POLICY[policy_key]
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
