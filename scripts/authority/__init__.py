"""Shared authority loader for gpu_cfd foundation tooling."""

from .bundle import (
    AuthorityBundle,
    AuthorityConflictError,
    AuthorityLoadError,
    AuthorityLoadReport,
    AuthoritySchemaError,
    load_authority_bundle,
    main,
    repo_root,
)
from .pins import (
    ConsumerPinResolution,
    EmittedEnvironmentManifests,
    emit_environment_manifests,
    resolve_consumer_pin_manifest,
)
from .support_scanner import (
    SupportBoundaryCondition,
    SupportFunctionObject,
    SupportScanIssue,
    SupportScanRejected,
    SupportScanRequest,
    SupportScanReport,
    SupportStartupSeedFieldValue,
    SupportStartupSeedRegion,
    SupportStartupSeedSpec,
    enforce_support_scan,
    scan_support_matrix,
)

__all__ = [
    "AuthorityBundle",
    "AuthorityConflictError",
    "AuthorityLoadError",
    "AuthorityLoadReport",
    "AuthoritySchemaError",
    "ConsumerPinResolution",
    "EmittedEnvironmentManifests",
    "SupportBoundaryCondition",
    "SupportFunctionObject",
    "SupportScanIssue",
    "SupportScanRejected",
    "SupportScanRequest",
    "SupportScanReport",
    "SupportStartupSeedFieldValue",
    "SupportStartupSeedRegion",
    "SupportStartupSeedSpec",
    "emit_environment_manifests",
    "enforce_support_scan",
    "load_authority_bundle",
    "main",
    "repo_root",
    "resolve_consumer_pin_manifest",
    "scan_support_matrix",
]
