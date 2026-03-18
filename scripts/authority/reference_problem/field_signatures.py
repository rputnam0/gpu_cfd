from __future__ import annotations

import hashlib
import math
import pathlib
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from .models import ARTIFACT_SCHEMA_VERSION, FIELD_SIGNATURES_ARTIFACT_NAME


STEADY_REQUIRED_FIELDS = ("U",)
STEADY_OPTIONAL_FIELDS = ("k", "omega", "nut")
TRANSIENT_REQUIRED_FIELDS = ("alpha.water", "U", "p_rgh")
TRANSIENT_OPTIONAL_FIELDS = ("rho", "phi")


def compute_field_signatures(
    *,
    normalized_steady_root: pathlib.Path | str,
    normalized_transient_root: pathlib.Path | str,
    case_identity: Mapping[str, Any],
    mesh_counts: Mapping[str, int] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    case_role = str(case_identity.get("case_role", "")).strip()
    steady_time_dir = _resolve_latest_time_dir(pathlib.Path(normalized_steady_root))
    transient_time_dir = _resolve_latest_time_dir(pathlib.Path(normalized_transient_root))
    if steady_time_dir is None:
        if case_role != "R2":
            raise ValueError(
                f"no OpenFOAM time directories found under {pathlib.Path(normalized_steady_root).as_posix()}"
            )
        steady_payload = {
            "available": False,
            "latest_time": None,
            "fields": {},
            "missing_optional_fields": list(STEADY_OPTIONAL_FIELDS),
        }
    else:
        steady_payload = _build_time_window_payload(
            steady_time_dir,
            label="steady",
            required_fields=STEADY_REQUIRED_FIELDS,
            optional_fields=STEADY_OPTIONAL_FIELDS,
            mesh_counts=mesh_counts,
        )
    transient_payload = _build_time_window_payload(
        transient_time_dir,
        label="transient",
        required_fields=TRANSIENT_REQUIRED_FIELDS,
        optional_fields=TRANSIENT_OPTIONAL_FIELDS,
        mesh_counts=mesh_counts,
    )
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "canonical_name": FIELD_SIGNATURES_ARTIFACT_NAME,
        **dict(case_identity),
        "steady_precondition": steady_payload,
        "transient_latest": transient_payload,
        "provenance": dict(provenance or {}),
    }


def _resolve_latest_time_dir(root: pathlib.Path) -> pathlib.Path | None:
    if not root.exists() or not root.is_dir():
        return None
    candidates = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        try:
            order_key = Decimal(path.name)
        except InvalidOperation:
            continue
        candidates.append((order_key, path.name, path))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][2]


def _build_time_window_payload(
    time_dir: pathlib.Path,
    *,
    label: str,
    required_fields: tuple[str, ...],
    optional_fields: tuple[str, ...],
    mesh_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    missing_required = [
        field_name for field_name in required_fields if not (time_dir / field_name).is_file()
    ]
    if missing_required:
        raise ValueError(
            f"missing required {label} field(s): {', '.join(missing_required)}"
        )

    fields = {
        field_name: _build_field_signature(time_dir / field_name, mesh_counts=mesh_counts)
        for field_name in required_fields + optional_fields
        if (time_dir / field_name).is_file()
    }
    missing_optional = [
        field_name for field_name in optional_fields if not (time_dir / field_name).is_file()
    ]
    return {
        "available": True,
        "latest_time": time_dir.name,
        "fields": fields,
        "missing_optional_fields": missing_optional,
    }


def _build_field_signature(
    path: pathlib.Path,
    *,
    mesh_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    field_kind, values, sample_count = _parse_internal_field(text, mesh_counts=mesh_counts)
    normalized_text = _normalize_text(text)
    payload = {
        "field_kind": field_kind,
        "normalized_file_sha256": hashlib.sha256(normalized_text.encode("utf-8")).hexdigest(),
        "sample_count": sample_count,
        "source_path": path.as_posix(),
    }
    if field_kind == "scalar":
        scalar_values = [float(value) for value in values]
        scalar_sum = float(sum(scalar_values))
        if sample_count != len(scalar_values):
            scalar_sum = float(scalar_values[0] * sample_count)
        payload["stats"] = {
            "min": min(scalar_values),
            "max": max(scalar_values),
            "sum": scalar_sum,
            "mean": float(scalar_sum / sample_count),
        }
    else:
        magnitudes = [
            math.sqrt((x * x) + (y * y) + (z * z))
            for x, y, z in values
        ]
        l2_magnitude = math.sqrt(sum(value * value for value in magnitudes))
        if sample_count != len(magnitudes):
            l2_magnitude = math.sqrt(sample_count * (magnitudes[0] ** 2))
        payload["stats"] = {
            "min_magnitude": min(magnitudes),
            "max_magnitude": max(magnitudes),
            "l2_magnitude": l2_magnitude,
        }
    return payload


def _parse_internal_field(
    text: str,
    *,
    mesh_counts: Mapping[str, int] | None = None,
) -> tuple[str, list[float] | list[tuple[float, float, float]], int]:
    uniform_match = re.search(r"internalField\s+uniform\s+([^;]+);", text, flags=re.DOTALL)
    if uniform_match is not None:
        token = uniform_match.group(1).strip()
        sample_count = _uniform_sample_count(text, mesh_counts=mesh_counts)
        if token.startswith("("):
            return "vector", [_parse_vector(token)], sample_count
        return "scalar", [float(token)], sample_count

    nonuniform_match = re.search(
        r"internalField\s+nonuniform\s+List<(?P<kind>scalar|vector)>\s*"
        r"(?P<count>\d+)\s*\(\s*(?P<body>.*?)\s*\)\s*;",
        text,
        flags=re.DOTALL,
    )
    if nonuniform_match is None:
        raise ValueError("field file does not contain a parseable internalField")

    field_kind = nonuniform_match.group("kind")
    body = nonuniform_match.group("body")
    expected_count = int(nonuniform_match.group("count"))
    if field_kind == "scalar":
        values = [float(value) for value in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", body)]
    else:
        values = [
            _parse_vector(f"({match})")
            for match in re.findall(r"\(([^()]+)\)", body)
        ]
    if len(values) != expected_count:
        raise ValueError(
            f"field file declared {expected_count} values but contained {len(values)}"
        )
    return field_kind, values, len(values)


def _uniform_sample_count(text: str, *, mesh_counts: Mapping[str, int] | None) -> int:
    if mesh_counts is None:
        raise ValueError("mesh_counts are required to expand uniform internalField values")
    field_class_match = re.search(r"\bclass\s+([A-Za-z0-9]+)\s*;", text)
    if field_class_match is None:
        raise ValueError("field file is missing a class declaration")
    field_class = field_class_match.group(1)
    if field_class.startswith("vol"):
        return int(mesh_counts["cells"])
    if field_class.startswith("surface"):
        return int(mesh_counts["internal_faces"])
    raise ValueError(f"unsupported OpenFOAM field class for uniform expansion: {field_class}")


def _parse_vector(token: str) -> tuple[float, float, float]:
    values = [float(value) for value in token.strip("() ").split()]
    if len(values) != 3:
        raise ValueError("vector field values must contain three components")
    return (values[0], values[1], values[2])


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").strip() + "\n"
