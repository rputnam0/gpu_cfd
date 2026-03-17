"""Authority-driven support scanner for startup admissibility checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .bundle import AuthorityBundle


GLOBAL_POLICY_CITATION = "docs/authority/support_matrix.json#global_policy"
SCHEME_CITATION = "docs/authority/support_matrix.json#exact_audited_scheme_tuple"
FUNCTION_OBJECT_CITATION = "docs/authority/support_matrix.json#function_object_policy"
BOUNDARY_CITATION = "docs/authority/support_matrix.json#phase6_nozzle_specific_envelope"
STARTUP_SEED_CITATION = "docs/authority/support_matrix.json#startup_seed_dsl"
BACKEND_CITATION = "docs/authority/support_matrix.json#backend_operational_policy"
CONTINUITY_CITATION = "docs/authority/continuity_ledger.md#1-frozen-global-decisions"
DEBUG_FALLBACK_POLICY = "debugOnlyFallback"
FAIL_FAST_POLICY = "failFast"


@dataclass(frozen=True)
class SupportFunctionObject:
    name: str
    class_name: str
    execute_control: str


@dataclass(frozen=True)
class SupportBoundaryCondition:
    patch_role: str
    field: str
    kind: str
    value: str | None = None


@dataclass(frozen=True)
class SupportStartupSeedFieldValue:
    value_class: str
    field: str
    value: Any


@dataclass(frozen=True)
class SupportStartupSeedRegion:
    region_type: str
    field_values: tuple[SupportStartupSeedFieldValue, ...] = ()


@dataclass(frozen=True)
class SupportStartupSeedSpec:
    enabled: bool
    force_reseed: bool
    precedence: str
    default_field_values: tuple[SupportStartupSeedFieldValue, ...] = ()
    regions: tuple[SupportStartupSeedRegion, ...] = ()
    extra_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class SupportScanRequest:
    execution_mode: str
    fallback_policy: str = FAIL_FAST_POLICY
    backend: str = "native"
    backend_pressure_mode: str | None = None
    mesh_mode: str = "static"
    region_count: int = 1
    processor_patches_present: bool = False
    cyclic_or_ami_patches_present: bool = False
    arbitrary_coded_patch_fields_present: bool = False
    turbulence_model: str = "laminar"
    contact_angle_enabled: bool = False
    surface_tension_model: str = "constant_sigma"
    schemes: dict[str, dict[str, str]] = field(default_factory=dict)
    function_objects: tuple[SupportFunctionObject, ...] = ()
    boundary_conditions: tuple[SupportBoundaryCondition, ...] = ()
    startup_seed: SupportStartupSeedSpec | None = None


@dataclass(frozen=True)
class SupportScanIssue:
    code: str
    message: str
    citations: tuple[str, ...]
    detail: dict[str, Any] = field(default_factory=dict)
    debug_fallback_eligible: bool = False

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "code": self.code,
            "message": self.message,
            "citations": list(self.citations),
        }
        if self.detail:
            payload["detail"] = dict(self.detail)
        return payload


@dataclass(frozen=True)
class SupportScanReport:
    startup_allowed: bool
    production_eligible: bool
    mode_label: str
    fallback_policy: str
    authority_citations: tuple[str, ...]
    issues: tuple[SupportScanIssue, ...]

    @property
    def reject_reasons(self) -> tuple[dict[str, Any], ...]:
        return tuple(issue.as_dict() for issue in self.issues)

    def as_dict(self) -> dict[str, Any]:
        return {
            "startup_allowed": self.startup_allowed,
            "production_eligible": self.production_eligible,
            "mode_label": self.mode_label,
            "fallback_policy": self.fallback_policy,
            "authority_citations": list(self.authority_citations),
            "reject_reasons": [issue.as_dict() for issue in self.issues],
        }


class SupportScanRejected(RuntimeError):
    """Raised when a startup request is not admitted by the support scanner."""

    def __init__(self, report: SupportScanReport) -> None:
        self.report = report
        codes = ", ".join(issue.code for issue in report.issues) or "unknown"
        super().__init__(f"Startup support scan rejected: {codes}")


def scan_support_matrix(bundle: AuthorityBundle, request: SupportScanRequest) -> SupportScanReport:
    issues = list(_scan_issues(bundle, request))
    explicit_debug_fallback = (
        request.execution_mode == "debug" and request.fallback_policy == DEBUG_FALLBACK_POLICY
    )
    fallback_allowed = explicit_debug_fallback and all(
        issue.debug_fallback_eligible for issue in issues
    )
    startup_allowed = not issues or fallback_allowed
    production_eligible = request.execution_mode == "production" and not issues
    mode_label = "debug-only-fallback" if fallback_allowed and issues else request.execution_mode
    return SupportScanReport(
        startup_allowed=startup_allowed,
        production_eligible=production_eligible,
        mode_label=mode_label,
        fallback_policy=request.fallback_policy,
        authority_citations=(
            GLOBAL_POLICY_CITATION,
            SCHEME_CITATION,
            FUNCTION_OBJECT_CITATION,
            BOUNDARY_CITATION,
            STARTUP_SEED_CITATION,
            BACKEND_CITATION,
            CONTINUITY_CITATION,
        ),
        issues=tuple(sorted(issues, key=lambda issue: issue.code)),
    )


def enforce_support_scan(bundle: AuthorityBundle, request: SupportScanRequest) -> SupportScanReport:
    report = scan_support_matrix(bundle, request)
    if not report.startup_allowed:
        raise SupportScanRejected(report)
    return report


def _scan_issues(bundle: AuthorityBundle, request: SupportScanRequest) -> tuple[SupportScanIssue, ...]:
    issues: list[SupportScanIssue] = []
    global_policy = bundle.support.raw["global_policy"]

    if request.execution_mode == "production" and request.fallback_policy == DEBUG_FALLBACK_POLICY:
        issues.append(
            SupportScanIssue(
                code="production_debug_fallback_forbidden",
                message="debugOnlyFallback is a non-production mode and cannot be carried on production requests.",
                citations=(GLOBAL_POLICY_CITATION, CONTINUITY_CITATION),
            )
        )
    if request.mesh_mode != global_policy["mesh_mode"]:
        issues.append(
            SupportScanIssue(
                code="mesh_mode_violation",
                message="Only static meshes are admitted in milestone-1 scope.",
                citations=(GLOBAL_POLICY_CITATION, CONTINUITY_CITATION),
                detail={"requested": request.mesh_mode, "expected": global_policy["mesh_mode"]},
            )
        )
    if request.region_count != int(global_policy["region_count"]):
        issues.append(
            SupportScanIssue(
                code="region_count_violation",
                message="Only single-region cases are admitted in milestone-1 scope.",
                citations=(GLOBAL_POLICY_CITATION,),
                detail={"requested": request.region_count, "expected": global_policy["region_count"]},
            )
        )
    if request.processor_patches_present:
        issues.append(
            SupportScanIssue(
                code="processor_patch_violation",
                message="Processor patches are not admitted in milestone-1 production scope.",
                citations=(GLOBAL_POLICY_CITATION,),
            )
        )
    if request.cyclic_or_ami_patches_present:
        issues.append(
            SupportScanIssue(
                code="cyclic_or_ami_patch_violation",
                message="Cyclic or AMI patches are not admitted in milestone-1 production scope.",
                citations=(GLOBAL_POLICY_CITATION,),
            )
        )
    if request.arbitrary_coded_patch_fields_present:
        issues.append(
            SupportScanIssue(
                code="coded_patch_field_violation",
                message="Arbitrary coded patch fields are not admitted in milestone-1 scope.",
                citations=(GLOBAL_POLICY_CITATION,),
            )
        )
    if request.turbulence_model != "laminar":
        issues.append(
            SupportScanIssue(
                code="turbulence_scope_violation",
                message="Turbulence scope is laminar-only.",
                citations=(GLOBAL_POLICY_CITATION,),
                detail={"requested": request.turbulence_model},
            )
        )
    if request.contact_angle_enabled:
        issues.append(
            SupportScanIssue(
                code="contact_angle_scope_violation",
                message="Contact-angle support is out of milestone-1 scope.",
                citations=(GLOBAL_POLICY_CITATION, CONTINUITY_CITATION),
            )
        )
    if request.surface_tension_model != "constant_sigma":
        issues.append(
            SupportScanIssue(
                code="surface_tension_scope_violation",
                message="Only constant sigma surface tension is admitted in milestone-1 scope.",
                citations=(GLOBAL_POLICY_CITATION,),
                detail={"requested": request.surface_tension_model},
            )
        )

    issues.extend(_scan_backend(request))
    issues.extend(_scan_schemes(bundle, request))
    issues.extend(_scan_function_objects(bundle, request))
    issues.extend(_scan_boundary_conditions(bundle, request))
    issues.extend(_scan_startup_seed(bundle, request))
    return tuple(issues)


def _scan_backend(request: SupportScanRequest) -> tuple[SupportScanIssue, ...]:
    if request.backend == "native":
        return ()
    if request.backend != "amgx":
        return (
            SupportScanIssue(
                code="backend_not_admitted",
                message="Backend eligibility is limited to the native baseline and the Phase 4 AmgX bridge.",
                citations=(BACKEND_CITATION, CONTINUITY_CITATION),
                detail={"backend": request.backend},
            ),
        )
    if request.execution_mode == "production" and request.backend_pressure_mode != "DeviceDirect":
        return (
            SupportScanIssue(
                code="backend_mode_not_admitted",
                message="AmgX production claims require DeviceDirect; PinnedHost is bring-up only.",
                citations=(BACKEND_CITATION, CONTINUITY_CITATION),
                detail={"backend": request.backend, "pressure_mode": request.backend_pressure_mode},
            ),
        )
    if request.execution_mode == "debug" and request.backend_pressure_mode == "PinnedHost":
        return (
            SupportScanIssue(
                code="backend_debug_only_mode_enabled",
                message="PinnedHost AmgX mode is correctness-only bring-up and must be labeled non-production.",
                citations=(BACKEND_CITATION, CONTINUITY_CITATION),
                detail={"backend": request.backend, "pressure_mode": request.backend_pressure_mode},
                debug_fallback_eligible=True,
            ),
        )
    return ()


def _scan_schemes(bundle: AuthorityBundle, request: SupportScanRequest) -> tuple[SupportScanIssue, ...]:
    expected = {
        block: {key: value for key, value in values.items() if key not in {"notes", "unsupported_examples"}}
        for block, values in bundle.support.raw["exact_audited_scheme_tuple"].items()
        if isinstance(values, dict)
    }
    observed = {
        block: { _normalize_scheme_key(key): value for key, value in request.schemes.get(block, {}).items() }
        for block in expected
    }
    if observed != expected:
        return (
            SupportScanIssue(
                code="unsupported_scheme_tuple",
                message="The startup scheme tuple must match the exact audited authority tuple.",
                citations=(SCHEME_CITATION,),
                detail={"observed": observed, "expected": expected},
            ),
        )
    return ()


def _scan_function_objects(
    bundle: AuthorityBundle, request: SupportScanRequest
) -> tuple[SupportScanIssue, ...]:
    policy = bundle.support.raw["function_object_policy"]
    classifications = {
        row["class_name"]: row["classification"] for row in policy["classes"]
    }
    issues: list[SupportScanIssue] = []
    for function_object in request.function_objects:
        classification = classifications.get(function_object.class_name, policy["default_classification"])
        if classification == "unsupported":
            issues.append(
                SupportScanIssue(
                    code="function_object_unsupported_class",
                    message="Unsupported functionObject classes must be rejected before startup.",
                    citations=(FUNCTION_OBJECT_CITATION,),
                    detail={"class_name": function_object.class_name, "name": function_object.name},
                )
            )
            continue
        if classification == "writeTimeOnly" and function_object.execute_control != "writeTime":
            issues.append(
                SupportScanIssue(
                    code="function_object_not_write_time_only",
                    message="writeTimeOnly functionObjects may execute only at write times.",
                    citations=(FUNCTION_OBJECT_CITATION,),
                    detail={
                        "class_name": function_object.class_name,
                        "name": function_object.name,
                        "execute_control": function_object.execute_control,
                    },
                )
            )
            continue
        if classification != "debugOnly":
            continue
        if request.execution_mode == "production":
            issues.append(
                SupportScanIssue(
                    code="function_object_debug_only_in_production",
                    message="debugOnly functionObjects are not admitted in production runs.",
                    citations=(FUNCTION_OBJECT_CITATION, GLOBAL_POLICY_CITATION),
                    detail={"class_name": function_object.class_name, "name": function_object.name},
                )
            )
            continue
        issues.append(
            SupportScanIssue(
                code="function_object_debug_only_enabled",
                message="debugOnly functionObjects require explicit debug-only fallback labeling.",
                citations=(FUNCTION_OBJECT_CITATION, GLOBAL_POLICY_CITATION),
                detail={"class_name": function_object.class_name, "name": function_object.name},
                debug_fallback_eligible=True,
            )
        )
    return tuple(issues)


def _scan_boundary_conditions(
    bundle: AuthorityBundle, request: SupportScanRequest
) -> tuple[SupportScanIssue, ...]:
    allowed_rows = {
        (row["patch_role"], row["field"]): row for row in bundle.support.raw["phase6_nozzle_specific_envelope"]
    }
    issues: list[SupportScanIssue] = []
    for condition in request.boundary_conditions:
        row = allowed_rows.get((condition.patch_role, condition.field))
        if row is None and condition.patch_role == "Symmetry / empty":
            row = allowed_rows.get((condition.patch_role, "all relevant fields"))
        if row is None:
            issues.append(
                SupportScanIssue(
                    code="unsupported_boundary_condition",
                    message="Boundary-condition tuple is not present in the frozen Phase 6 envelope.",
                    citations=(BOUNDARY_CITATION,),
                    detail={
                        "patch_role": condition.patch_role,
                        "field": condition.field,
                        "kind": condition.kind,
                    },
                )
            )
            continue
        normalized_kind = _normalize_boundary_kind(condition, row)
        if normalized_kind not in row["allowed_kinds"]:
            issues.append(
                SupportScanIssue(
                    code="unsupported_boundary_condition",
                    message="Boundary-condition kind is not admitted for the frozen nozzle envelope.",
                    citations=(BOUNDARY_CITATION,),
                    detail={
                        "patch_role": condition.patch_role,
                        "field": condition.field,
                        "kind": condition.kind,
                    },
                )
            )
    return tuple(issues)


def _scan_startup_seed(bundle: AuthorityBundle, request: SupportScanRequest) -> tuple[SupportScanIssue, ...]:
    spec = request.startup_seed
    if spec is None or not spec.enabled:
        return ()

    seed_policy = bundle.support.raw["startup_seed_dsl"]
    issues: list[SupportScanIssue] = []
    if spec.precedence not in seed_policy["precedence_policy"]["allowed_values"]:
        issues.append(
            SupportScanIssue(
                code="startup_seed_precedence_not_admitted",
                message="Startup-seed precedence must use the frozen lastWins policy.",
                citations=(STARTUP_SEED_CITATION,),
                detail={"requested": spec.precedence},
            )
        )
    for extra_key in spec.extra_keys:
        issues.append(
            SupportScanIssue(
                code="startup_seed_unknown_key",
                message="Unknown startup-seed top-level keys are rejected.",
                citations=(STARTUP_SEED_CITATION,),
                detail={"key": extra_key},
            )
        )
    allowed_field_entries = {
        (row["value_class"], row["field"]) for row in seed_policy["allowed_field_value_entries"]
    }
    allowed_region_types = {
        row["region_type"] for row in seed_policy["supported_region_types"]
    }
    for field_value in spec.default_field_values:
        if (field_value.value_class, field_value.field) not in allowed_field_entries:
            issues.append(
                SupportScanIssue(
                    code="startup_seed_field_value_not_admitted",
                    message="Startup-seed field-value entries must match the frozen DSL.",
                    citations=(STARTUP_SEED_CITATION,),
                    detail={
                        "value_class": field_value.value_class,
                        "field": field_value.field,
                    },
                )
            )
    for region in spec.regions:
        if region.region_type not in allowed_region_types:
            issues.append(
                SupportScanIssue(
                    code="startup_seed_region_not_admitted",
                    message="Startup-seed region families must match the frozen DSL.",
                    citations=(STARTUP_SEED_CITATION,),
                    detail={"region_type": region.region_type},
                )
            )
        for field_value in region.field_values:
            if (field_value.value_class, field_value.field) not in allowed_field_entries:
                issues.append(
                    SupportScanIssue(
                        code="startup_seed_field_value_not_admitted",
                        message="Startup-seed field-value entries must match the frozen DSL.",
                        citations=(STARTUP_SEED_CITATION,),
                        detail={
                            "value_class": field_value.value_class,
                            "field": field_value.field,
                            "region_type": region.region_type,
                        },
                    )
                )
    return tuple(issues)


def _normalize_scheme_key(key: str) -> str:
    return (
        key.replace("alpha.water", "alpha1")
        .replace("alpha.air", "alpha1")
        .replace("grad(alpha1)", "grad(alpha1)")
    )


def _normalize_boundary_kind(condition: SupportBoundaryCondition, row: dict[str, Any]) -> str:
    if condition.kind in row["allowed_kinds"]:
        return condition.kind
    for alias in row.get("compatibility_aliases", ()):
        if (
            condition.kind == alias["kind"]
            and condition.value == alias.get("required_value")
        ):
            return str(alias["normalizes_to"])
    return condition.kind
