from __future__ import annotations

import pathlib
from dataclasses import dataclass


ARTIFACT_SCHEMA_VERSION = "1.0.0"
BUILD_FINGERPRINT_ARTIFACT_NAME = "build_fingerprint.json"
FIELD_SIGNATURES_ARTIFACT_NAME = "field_signatures.json"
METRICS_ARTIFACT_NAME = "metrics.json"


@dataclass(frozen=True)
class ReferenceProblemArtifacts:
    build_fingerprint_path: pathlib.Path
    field_signatures_path: pathlib.Path
    metrics_path: pathlib.Path | None = None
