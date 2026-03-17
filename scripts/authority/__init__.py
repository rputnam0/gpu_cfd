"""Shared authority loader for gpu_cfd foundation tooling."""

from .acceptance import (
    AcceptanceClassResult,
    AcceptanceEvaluationContext,
    AcceptanceVerdict,
    AcceptanceWaiver,
    AcceptanceWaiverHook,
    AcceptedTupleResolution,
    evaluate_acceptance,
    resolve_accepted_tuple,
)
from .bundle import (
    AuthorityBundle,
    AuthorityConflictError,
    AuthorityLoadError,
    AuthorityLoadReport,
    AuthoritySchemaError,
    load_authority_bundle,
    main,
)
from .pins import (
    ConsumerPinResolution,
    EmittedEnvironmentManifests,
    emit_environment_manifests,
    resolve_consumer_pin_manifest,
)
from .source_audit import (
    ResolvedSourceAuditSurface,
    SOURCE_AUDIT_OWNERSHIP_SCOPE_PLACEHOLDER,
    render_source_audit_note,
    resolve_source_audit_surfaces,
    validate_source_audit_note,
)

__all__ = [
    "AcceptanceClassResult",
    "AcceptanceEvaluationContext",
    "AcceptanceVerdict",
    "AcceptanceWaiver",
    "AcceptanceWaiverHook",
    "AcceptedTupleResolution",
    "AuthorityBundle",
    "AuthorityConflictError",
    "AuthorityLoadError",
    "AuthorityLoadReport",
    "AuthoritySchemaError",
    "ConsumerPinResolution",
    "EmittedEnvironmentManifests",
    "ResolvedSourceAuditSurface",
    "SOURCE_AUDIT_OWNERSHIP_SCOPE_PLACEHOLDER",
    "emit_environment_manifests",
    "evaluate_acceptance",
    "load_authority_bundle",
    "main",
    "render_source_audit_note",
    "resolve_accepted_tuple",
    "resolve_consumer_pin_manifest",
    "resolve_source_audit_surfaces",
    "validate_source_audit_note",
]
