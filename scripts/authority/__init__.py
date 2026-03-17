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
    "evaluate_acceptance",
    "emit_environment_manifests",
    "load_authority_bundle",
    "main",
    "resolve_accepted_tuple",
    "resolve_consumer_pin_manifest",
]
