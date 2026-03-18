from __future__ import annotations

from typing import Any, Mapping

from .models import ARTIFACT_SCHEMA_VERSION, METRICS_ARTIFACT_NAME


def build_feature_extractor_metrics_json(
    *,
    case_identity: Mapping[str, Any],
    metrics: Mapping[str, Any],
    metric_sources: Mapping[str, str],
    time_windows: Mapping[str, Any] | None = None,
    angle_source: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_angle_source = angle_source
    metric_values = dict(metrics)
    if resolved_angle_source is None and "spray_angle_source" in metric_values:
        resolved_angle_source = str(metric_values.pop("spray_angle_source"))
    missing_sources = [name for name in metric_values if name not in metric_sources]
    if missing_sources:
        raise ValueError(
            "missing metric_source metadata for metric(s): " + ", ".join(sorted(missing_sources))
        )
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "canonical_name": METRICS_ARTIFACT_NAME,
        **dict(case_identity),
        "metrics": {
            metric_name: {
                "value": metric_value,
                "metric_source": metric_sources[metric_name],
            }
            for metric_name, metric_value in metric_values.items()
        },
        "time_windows": dict(time_windows or {}),
        "angle_source": resolved_angle_source,
        "provenance": dict(provenance or {}),
    }
