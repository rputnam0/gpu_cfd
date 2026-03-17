from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
from typing import Any, Mapping

from .models import ARTIFACT_SCHEMA_VERSION, BUILD_FINGERPRINT_ARTIFACT_NAME


MESH_FILE_NAMES = ("points", "faces", "owner", "neighbour", "boundary")


def compute_mesh_patch_fingerprint(
    case_dir: pathlib.Path | str,
    *,
    case_identity: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    poly_mesh_dir = pathlib.Path(case_dir) / "constant" / "polyMesh"
    contents = {
        file_name: _read_text(poly_mesh_dir / file_name)
        for file_name in MESH_FILE_NAMES
    }
    points = _parse_points(contents["points"])
    face_count = _parse_face_count(contents["faces"])
    owner = _parse_integer_list(contents["owner"])
    neighbour = _parse_integer_list(contents["neighbour"])
    patches = _parse_boundary(contents["boundary"])

    mesh_counts = {
        "cells": _compute_cell_count(owner, neighbour),
        "faces": face_count,
        "points": len(points),
        "internal_faces": len(neighbour),
        "patches": len(patches),
    }
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "canonical_name": BUILD_FINGERPRINT_ARTIFACT_NAME,
        **dict(case_identity),
        "mesh_files": {
            file_name: {
                "path": (poly_mesh_dir / file_name).as_posix(),
                "sha256": hashlib.sha256(contents[file_name].encode("utf-8")).hexdigest(),
            }
            for file_name in MESH_FILE_NAMES
        },
        "semantic_patches": patches,
        "mesh_counts": mesh_counts,
        "bounding_box": _compute_bounding_box(points),
        "provenance": dict(provenance or {}),
    }
    payload["fingerprint_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "mesh_files": {name: payload["mesh_files"][name]["sha256"] for name in MESH_FILE_NAMES},
                "semantic_patches": patches,
                "mesh_counts": mesh_counts,
                "bounding_box": payload["bounding_box"],
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return payload


def _read_text(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"missing required mesh file: {path.as_posix()}") from exc


def _parse_points(text: str) -> list[tuple[float, float, float]]:
    body = _extract_list_body(text)
    points = []
    for match in re.finditer(r"\(([^()]+)\)", body):
        values = [float(value) for value in match.group(1).split()]
        if len(values) != 3:
            raise ValueError("points file must contain three-component coordinates")
        points.append((values[0], values[1], values[2]))
    if not points:
        raise ValueError("points file did not contain any coordinates")
    return points


def _parse_face_count(text: str) -> int:
    body = _extract_list_body(text)
    return len(re.findall(r"^\s*\d+\s*\([^()]*\)\s*$", body, flags=re.MULTILINE))


def _parse_integer_list(text: str) -> list[int]:
    body = _extract_list_body(text)
    return [int(value) for value in re.findall(r"[-+]?\d+", body)]


def _parse_boundary(text: str) -> list[dict[str, Any]]:
    body = _extract_list_body(text)
    matches = re.finditer(r"^\s*(\S+)\s*\n\s*\{(.*?)^\s*\}", body, flags=re.MULTILINE | re.DOTALL)
    patches = []
    for match in matches:
        entries = match.group(2)
        patch_type = _extract_boundary_value(entries, key="type")
        n_faces = _extract_boundary_value(entries, key="nFaces")
        start_face = _extract_boundary_value(entries, key="startFace")
        patches.append(
            {
                "name": match.group(1),
                "type": patch_type,
                "nFaces": int(n_faces),
                "startFace": int(start_face),
            }
        )
    if not patches:
        raise ValueError("boundary file did not contain any patch entries")
    return patches


def _extract_boundary_value(entries: str, *, key: str) -> str:
    match = re.search(rf"{re.escape(key)}\s+([^;]+);", entries)
    if match is None:
        raise ValueError(f"boundary file is missing {key}")
    return match.group(1).strip()


def _extract_list_body(text: str) -> str:
    match = re.search(r"\n\s*\d+\s*\(\s*(.*?)\s*\)\s*;?\s*$", text, flags=re.DOTALL)
    if match is None:
        raise ValueError("OpenFOAM list payload could not be parsed")
    return match.group(1)


def _compute_cell_count(owner: list[int], neighbour: list[int]) -> int:
    if not owner and not neighbour:
        return 0
    return max(owner + neighbour) + 1


def _compute_bounding_box(points: list[tuple[float, float, float]]) -> dict[str, list[float]]:
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    for point in points:
        for index, value in enumerate(point):
            mins[index] = min(mins[index], value)
            maxs[index] = max(maxs[index], value)
    return {"min": [float(value) for value in mins], "max": [float(value) for value in maxs]}
