"""Shared authority loader for gpu_cfd foundation tooling."""

from .bundle import (
    AuthorityBundle,
    AuthorityConflictError,
    AuthorityLoadError,
    AuthorityLoadReport,
    AuthoritySchemaError,
    load_authority_bundle,
    main,
)
from .cases import (
    AuthoritySelectionError,
    ResolvedReferenceCase,
    allowed_phase_gate_case_roles,
    case_meta_schema,
    resolve_phase_gate_case,
    resolve_reference_case,
    resolve_reference_case_by_frozen_id,
    stage_plan_schema,
    validate_case_meta,
    validate_frozen_ladder,
    validate_stage_plan,
)
from .pins import (
    ConsumerPinResolution,
    EmittedEnvironmentManifests,
    emit_environment_manifests,
    resolve_consumer_pin_manifest,
)

__all__ = [
    "AuthorityBundle",
    "AuthorityConflictError",
    "AuthorityLoadError",
    "AuthorityLoadReport",
    "AuthoritySchemaError",
    "AuthoritySelectionError",
    "ConsumerPinResolution",
    "EmittedEnvironmentManifests",
    "ResolvedReferenceCase",
    "allowed_phase_gate_case_roles",
    "case_meta_schema",
    "emit_environment_manifests",
    "load_authority_bundle",
    "main",
    "resolve_phase_gate_case",
    "resolve_reference_case",
    "resolve_reference_case_by_frozen_id",
    "resolve_consumer_pin_manifest",
    "stage_plan_schema",
    "validate_case_meta",
    "validate_frozen_ladder",
    "validate_stage_plan",
]
