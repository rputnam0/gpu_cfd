"""Reference-case resolution and metadata-schema helpers for foundation tooling."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .bundle import AuthorityBundle, ReferenceCase
from .stage_runner import StageRunnerContext


MANIFEST_SCHEMA_VERSION = "1.0.0"
CANONICAL_CASE_META_NAME = "case_meta.json"
CANONICAL_STAGE_PLAN_NAME = "stage_plan.json"
REQUIRED_PROVENANCE_FIELDS = ("probe_payload", "host_env", "manifest_refs")


class AuthoritySelectionError(ValueError):
    """Raised when a caller selects an unknown or out-of-scope authority case."""


@dataclass(frozen=True)
class ResolvedReferenceCase:
    case_role: str
    frozen_id: str
    purpose: str
    frozen_default_contract: dict[str, Any]
    ladder_position: int
    phase_gates: tuple[str, ...]


@dataclass(frozen=True)
class EmittedCaseArtifacts:
    case_meta_path: pathlib.Path
    stage_plan_path: pathlib.Path


def resolve_reference_case(bundle: AuthorityBundle, *, case_role: str) -> ResolvedReferenceCase:
    reference_case = bundle.cases.by_case_id.get(case_role)
    if reference_case is None:
        raise AuthoritySelectionError(f"unknown case role {case_role!r}")
    return _build_resolved_case(bundle, case_role=case_role, reference_case=reference_case)


def resolve_reference_case_by_frozen_id(
    bundle: AuthorityBundle, *, frozen_id: str
) -> ResolvedReferenceCase:
    for case_role, reference_case in bundle.cases.by_case_id.items():
        if reference_case.frozen_id == frozen_id:
            return _build_resolved_case(bundle, case_role=case_role, reference_case=reference_case)
    raise AuthoritySelectionError(f"unknown frozen case id {frozen_id!r}")


def validate_frozen_ladder(
    bundle: AuthorityBundle, ordered_case_roles: Sequence[str]
) -> tuple[str, ...]:
    normalized = tuple(ordered_case_roles)
    if normalized != bundle.ladder.ordered_case_ids:
        raise AuthoritySelectionError(
            "validation ladder must remain "
            + " -> ".join(bundle.ladder.ordered_case_ids)
        )
    return normalized


def allowed_phase_gate_case_roles(
    bundle: AuthorityBundle, *, phase_gate: str, include_conditional: bool = False
) -> tuple[str, ...]:
    return _allowed_phase_gate_case_roles(
        bundle,
        phase_gate=phase_gate,
        include_conditional=include_conditional,
    )


def _allowed_phase_gate_case_roles(
    bundle: AuthorityBundle, *, phase_gate: str, include_conditional: bool
) -> tuple[str, ...]:
    phase_gate_mapping = bundle.cases.phase_gate_mapping.get(phase_gate)
    if phase_gate_mapping is None:
        raise AuthoritySelectionError(f"unknown phase gate {phase_gate!r}")

    ordered_roles: list[str] = []
    for key in (
        "ordered_case_ladder",
        "default_cases",
        "hard_gate_cases",
    ):
        ordered_roles.extend(str(case_role) for case_role in phase_gate_mapping.get(key, ()))
    if include_conditional:
        ordered_roles.extend(
            str(case_role) for case_role in phase_gate_mapping.get("conditional_cases", ())
        )
    for key in (
        "accepted_case",
        "routine_architecture_baseline_case",
        "production_shape_acceptance_case",
        "backend_or_execution_parity_case",
    ):
        case_role = phase_gate_mapping.get(key)
        if case_role is not None:
            ordered_roles.append(str(case_role))

    deduped_roles = tuple(dict.fromkeys(ordered_roles))
    if not deduped_roles:
        raise AuthoritySelectionError(f"phase gate {phase_gate!r} has no resolved case roles")
    return deduped_roles


def resolve_phase_gate_case(
    bundle: AuthorityBundle, *, phase_gate: str, case_role: str, allow_conditional: bool = False
) -> ResolvedReferenceCase:
    allowed_case_roles = _allowed_phase_gate_case_roles(
        bundle,
        phase_gate=phase_gate,
        include_conditional=allow_conditional,
    )
    if case_role not in allowed_case_roles:
        conditional_case_roles = set(
            _allowed_phase_gate_case_roles(bundle, phase_gate=phase_gate, include_conditional=True)
        ) - set(allowed_phase_gate_case_roles(bundle, phase_gate=phase_gate))
        if not allow_conditional and case_role in conditional_case_roles:
            conditional_rule = bundle.cases.phase_gate_mapping[phase_gate].get(
                "conditional_case_rule",
                "authority-defined conditional rule",
            )
            raise AuthoritySelectionError(
                f"phase gate {phase_gate!r} allows case role {case_role!r} only conditionally: "
                f"{conditional_rule}"
            )
        raise AuthoritySelectionError(
            f"phase gate {phase_gate!r} does not allow case role {case_role!r}"
        )
    return resolve_reference_case(bundle, case_role=case_role)


def case_meta_schema(bundle: AuthorityBundle) -> dict[str, Any]:
    ordered_case_roles = list(bundle.ladder.ordered_case_ids)
    schema = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "canonical_name": CANONICAL_CASE_META_NAME,
        "type": "object",
        "required": [
            "schema_version",
            "case_id",
            "case_role",
            "ladder_position",
            "phase_gates",
            "baseline",
            "runtime_base",
            "reviewed_source_tuple_id",
            "requested_vof_solver_mode",
            "resolved_vof_solver_exec",
            "resolved_pressure_backend",
            "openfoam_bashrc_used",
            "available_commands",
            "mesh_full_360",
            "mesh_resolution_scale",
            "hydraulic_domain_mode",
            "near_field_radius_d",
            "near_field_length_d",
            "steady_end_time_iter",
            "steady_write_interval_iter",
            "steady_turbulence_model",
            "vof_turbulence_model",
            "delta_t_s",
            "write_interval_s",
            "end_time_s",
            "max_co",
            "max_alpha_co",
            "resolved_direct_slot_numerics",
            "startup_fill_extension_d",
            "air_core_seed_radius_d_requested",
            "air_core_seed_radius_m_resolved",
            "air_core_seed_cap_applied",
            "fill_radius_m_resolved",
            "fill_z_start_m",
            "fill_z_stop_m",
            "DeltaP_Pa",
            "DeltaP_effective_Pa",
            "check_valve_loss_applied",
            "provenance",
        ],
        "properties": {
            "schema_version": {"const": MANIFEST_SCHEMA_VERSION},
            "case_id": {
                "enum": [
                    bundle.cases.by_case_id[case_role].frozen_id for case_role in ordered_case_roles
                ]
            },
            "case_role": {"enum": ordered_case_roles},
            "ladder_position": {
                "type": "integer",
                "minimum": 1,
                "maximum": len(ordered_case_roles),
            },
            "phase_gates": {
                "type": "array",
                "minItems": 1,
                "items": {"enum": list(bundle.cases.phase_gate_mapping)},
                "uniqueItems": True,
            },
            "baseline": {"type": "string", "minLength": 1, "pattern": r".*\S.*"},
            "runtime_base": {"type": ["string", "null"]},
            "reviewed_source_tuple_id": {"type": ["string", "null"]},
            "requested_vof_solver_mode": {"type": "string", "minLength": 1, "pattern": r".*\S.*"},
            "resolved_vof_solver_exec": {"type": "string", "minLength": 1, "pattern": r".*\S.*"},
            "resolved_pressure_backend": {"type": "string", "minLength": 1, "pattern": r".*\S.*"},
            "openfoam_bashrc_used": {"type": "string", "minLength": 1, "pattern": r".*\S.*"},
            "available_commands": {
                "type": "object",
                "minProperties": 1,
                "additionalProperties": {
                    "type": "string",
                    "minLength": 1,
                    "pattern": r".*\S.*",
                },
            },
            "mesh_full_360": {"type": "integer", "enum": [0, 1]},
            "mesh_resolution_scale": {"type": "number"},
            "hydraulic_domain_mode": {"type": "string", "minLength": 1, "pattern": r".*\S.*"},
            "near_field_radius_d": {"type": ["number", "null"]},
            "near_field_length_d": {"type": ["number", "null"]},
            "steady_end_time_iter": {"type": ["integer", "null"]},
            "steady_write_interval_iter": {"type": ["integer", "null"]},
            "steady_turbulence_model": {"type": ["string", "null"]},
            "vof_turbulence_model": {"type": ["string", "null"]},
            "delta_t_s": {"type": "number"},
            "write_interval_s": {"type": "number"},
            "end_time_s": {"type": "number"},
            "max_co": {"type": "number"},
            "max_alpha_co": {"type": "number"},
            "resolved_direct_slot_numerics": {"type": "object", "minProperties": 1},
            "startup_fill_extension_d": {"type": "number"},
            "air_core_seed_radius_d_requested": {"type": "number"},
            "air_core_seed_radius_m_resolved": {"type": "number"},
            "air_core_seed_cap_applied": {"type": "boolean"},
            "fill_radius_m_resolved": {"type": "number"},
            "fill_z_start_m": {"type": "number"},
            "fill_z_stop_m": {"type": "number"},
            "DeltaP_Pa": {"type": "number"},
            "DeltaP_effective_Pa": {"type": "number"},
            "check_valve_loss_applied": {"type": "boolean"},
            "provenance": {
                "type": "object",
                "required": list(REQUIRED_PROVENANCE_FIELDS),
                "properties": {
                    "probe_payload": {"type": "string", "minLength": 1, "pattern": r".*\S.*"},
                    "host_env": {"type": "string", "minLength": 1, "pattern": r".*\S.*"},
                    "manifest_refs": {"type": "string", "minLength": 1, "pattern": r".*\S.*"},
                },
            },
        },
    }
    schema["allOf"] = [{"oneOf": _case_meta_variants(bundle)}]
    return schema


def validate_case_meta(bundle: AuthorityBundle, payload: dict[str, Any]) -> dict[str, Any]:
    _validate_required_fields(
        payload,
        required_fields=(
            "schema_version",
            "case_id",
            "case_role",
            "ladder_position",
            "phase_gates",
            "baseline",
            "runtime_base",
            "reviewed_source_tuple_id",
            "requested_vof_solver_mode",
            "resolved_vof_solver_exec",
            "resolved_pressure_backend",
            "openfoam_bashrc_used",
            "available_commands",
            "mesh_full_360",
            "mesh_resolution_scale",
            "hydraulic_domain_mode",
            "near_field_radius_d",
            "near_field_length_d",
            "steady_end_time_iter",
            "steady_write_interval_iter",
            "steady_turbulence_model",
            "vof_turbulence_model",
            "delta_t_s",
            "write_interval_s",
            "end_time_s",
            "max_co",
            "max_alpha_co",
            "resolved_direct_slot_numerics",
            "startup_fill_extension_d",
            "air_core_seed_radius_d_requested",
            "air_core_seed_radius_m_resolved",
            "air_core_seed_cap_applied",
            "fill_radius_m_resolved",
            "fill_z_start_m",
            "fill_z_stop_m",
            "DeltaP_Pa",
            "DeltaP_effective_Pa",
            "check_valve_loss_applied",
            "provenance",
        ),
        artifact_name=CANONICAL_CASE_META_NAME,
    )
    _validate_schema_version(payload, artifact_name=CANONICAL_CASE_META_NAME)

    resolved_case = resolve_reference_case(bundle, case_role=str(payload["case_role"]))
    case_id = str(payload["case_id"])
    if case_id != resolved_case.frozen_id:
        raise AuthoritySelectionError(
            f"{CANONICAL_CASE_META_NAME} case_role {resolved_case.case_role!r} "
            f"must resolve to case_id {resolved_case.frozen_id!r}"
        )

    ladder_position = payload["ladder_position"]
    if not isinstance(ladder_position, int) or isinstance(ladder_position, bool):
        raise AuthoritySelectionError(f"{CANONICAL_CASE_META_NAME} ladder_position must be an integer")
    if ladder_position != resolved_case.ladder_position:
        raise AuthoritySelectionError(
            f"{CANONICAL_CASE_META_NAME} ladder_position {ladder_position} "
            f"must match authority ladder position {resolved_case.ladder_position} "
            f"for case role {resolved_case.case_role!r}"
        )

    phase_gates_value = payload["phase_gates"]
    if not isinstance(phase_gates_value, list):
        raise AuthoritySelectionError(
            f"{CANONICAL_CASE_META_NAME} phase_gates must be a list of phase-gate names"
        )
    if any(not isinstance(phase_gate, str) for phase_gate in phase_gates_value):
        raise AuthoritySelectionError(
            f"{CANONICAL_CASE_META_NAME} phase_gates must contain only strings"
        )
    if len(phase_gates_value) != len(set(phase_gates_value)):
        raise AuthoritySelectionError(
            f"{CANONICAL_CASE_META_NAME} phase_gates must not contain duplicates"
        )
    phase_gates = set(phase_gates_value)
    expected_phase_gates = set(resolved_case.phase_gates)
    if phase_gates != expected_phase_gates:
        raise AuthoritySelectionError(
            f"{CANONICAL_CASE_META_NAME} phase_gates for case role {resolved_case.case_role!r} "
            f"must remain {list(resolved_case.phase_gates)!r}"
        )

    _validate_non_empty_string_field(payload, "baseline", artifact_name=CANONICAL_CASE_META_NAME)
    _validate_optional_string_field(
        payload,
        "runtime_base",
        artifact_name=CANONICAL_CASE_META_NAME,
    )
    _validate_optional_string_field(
        payload,
        "reviewed_source_tuple_id",
        artifact_name=CANONICAL_CASE_META_NAME,
    )
    for field_name in (
        "requested_vof_solver_mode",
        "resolved_vof_solver_exec",
        "resolved_pressure_backend",
        "openfoam_bashrc_used",
        "hydraulic_domain_mode",
    ):
        _validate_non_empty_string_field(payload, field_name, artifact_name=CANONICAL_CASE_META_NAME)
    _validate_string_mapping_field(payload, "available_commands", artifact_name=CANONICAL_CASE_META_NAME)
    _validate_int_enum_field(
        payload,
        "mesh_full_360",
        allowed_values=(0, 1),
        artifact_name=CANONICAL_CASE_META_NAME,
    )
    for field_name in (
        "mesh_resolution_scale",
        "delta_t_s",
        "write_interval_s",
        "end_time_s",
        "max_co",
        "max_alpha_co",
        "startup_fill_extension_d",
        "air_core_seed_radius_d_requested",
        "air_core_seed_radius_m_resolved",
        "fill_radius_m_resolved",
        "fill_z_start_m",
        "fill_z_stop_m",
        "DeltaP_Pa",
        "DeltaP_effective_Pa",
    ):
        _validate_numeric_field(payload, field_name, artifact_name=CANONICAL_CASE_META_NAME)
    for field_name in ("near_field_radius_d", "near_field_length_d"):
        _validate_optional_numeric_field(payload, field_name, artifact_name=CANONICAL_CASE_META_NAME)
    for field_name in ("steady_end_time_iter", "steady_write_interval_iter"):
        _validate_optional_integer_field(payload, field_name, artifact_name=CANONICAL_CASE_META_NAME)
    for field_name in ("steady_turbulence_model", "vof_turbulence_model"):
        _validate_optional_string_field(payload, field_name, artifact_name=CANONICAL_CASE_META_NAME)
    for field_name in ("air_core_seed_cap_applied", "check_valve_loss_applied"):
        _validate_boolean_field(payload, field_name, artifact_name=CANONICAL_CASE_META_NAME)
    _validate_mapping_field(
        payload,
        "resolved_direct_slot_numerics",
        artifact_name=CANONICAL_CASE_META_NAME,
    )
    _validate_provenance_payload(payload, artifact_name=CANONICAL_CASE_META_NAME)

    return payload


def stage_plan_schema(bundle: AuthorityBundle) -> dict[str, Any]:
    ordered_case_roles = list(bundle.ladder.ordered_case_ids)
    schema = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "canonical_name": CANONICAL_STAGE_PLAN_NAME,
        "type": "object",
        "required": [
            "schema_version",
            "case_id",
            "case_role",
            "phase_gate",
            "phase_gate_selection",
            "stages",
            "baseline",
            "runtime_base",
            "reviewed_source_tuple_id",
            "provenance",
        ],
        "properties": {
            "schema_version": {"const": MANIFEST_SCHEMA_VERSION},
            "case_id": {
                "enum": [
                    bundle.cases.by_case_id[case_role].frozen_id for case_role in ordered_case_roles
                ]
            },
            "case_role": {"enum": ordered_case_roles},
            "phase_gate": {"enum": list(bundle.cases.phase_gate_mapping)},
            "baseline": {"type": "string", "minLength": 1, "pattern": r".*\S.*"},
            "runtime_base": {"type": ["string", "null"]},
            "reviewed_source_tuple_id": {"type": ["string", "null"]},
            "provenance": {
                "type": "object",
                "required": list(REQUIRED_PROVENANCE_FIELDS),
                "properties": {
                    "probe_payload": {"type": "string", "minLength": 1, "pattern": r".*\S.*"},
                    "host_env": {"type": "string", "minLength": 1, "pattern": r".*\S.*"},
                    "manifest_refs": {"type": "string", "minLength": 1, "pattern": r".*\S.*"},
                },
            },
            "phase_gate_selection": {
                "type": "object",
                "required": [
                    "selected_case_role",
                    "available_case_roles",
                    "ordered_ladder",
                    "conditional_selection",
                ],
                "properties": {
                    "selected_case_role": {"enum": ordered_case_roles},
                    "available_case_roles": {
                        "type": "array",
                        "items": {"enum": ordered_case_roles},
                        "uniqueItems": True,
                    },
                    "ordered_ladder": {"const": ordered_case_roles},
                    "conditional_selection": {"type": "boolean"},
                    "conditional_reason": {
                        "type": "string",
                        "minLength": 1,
                        "pattern": r".*\S.*",
                    },
                },
                "allOf": [
                    {
                        "if": {
                            "properties": {"conditional_selection": {"const": True}},
                            "required": ["conditional_selection"],
                        },
                        "then": {"required": ["conditional_reason"]},
                    },
                    {
                        "if": {
                            "properties": {"conditional_selection": {"const": False}},
                            "required": ["conditional_selection"],
                        },
                        "then": {"not": {"required": ["conditional_reason"]}},
                    }
                ],
            },
            "stages": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["name", "cmd"],
                    "properties": {
                        "name": {"type": "string", "minLength": 1, "pattern": r".*\S.*"},
                        "cmd": {"type": "string", "minLength": 1, "pattern": r".*\S.*"},
                        "cwd": {"type": "string"},
                    },
                },
            },
        },
    }
    schema["allOf"] = [{"oneOf": _stage_plan_variants(bundle)}]
    return schema


def validate_stage_plan(bundle: AuthorityBundle, payload: dict[str, Any]) -> dict[str, Any]:
    _validate_required_fields(
        payload,
        required_fields=(
            "schema_version",
            "case_id",
            "case_role",
            "phase_gate",
            "phase_gate_selection",
            "stages",
            "baseline",
            "runtime_base",
            "reviewed_source_tuple_id",
            "provenance",
        ),
        artifact_name=CANONICAL_STAGE_PLAN_NAME,
    )
    _validate_schema_version(payload, artifact_name=CANONICAL_STAGE_PLAN_NAME)

    resolved_case = resolve_reference_case(bundle, case_role=str(payload["case_role"]))
    case_id = str(payload["case_id"])
    if case_id != resolved_case.frozen_id:
        raise AuthoritySelectionError(
            f"{CANONICAL_STAGE_PLAN_NAME} case_role {resolved_case.case_role!r} "
            f"must resolve to case_id {resolved_case.frozen_id!r}"
        )

    phase_gate = str(payload["phase_gate"])
    conditional_case_roles = tuple(
        case_role
        for case_role in allowed_phase_gate_case_roles(
            bundle,
            phase_gate=phase_gate,
            include_conditional=True,
        )
        if case_role not in allowed_phase_gate_case_roles(bundle, phase_gate=phase_gate)
    )
    unconditional_case_roles = allowed_phase_gate_case_roles(bundle, phase_gate=phase_gate)

    phase_gate_selection = payload["phase_gate_selection"]
    if not isinstance(phase_gate_selection, dict):
        raise AuthoritySelectionError(
            f"{CANONICAL_STAGE_PLAN_NAME} phase_gate_selection must be an object"
        )
    _validate_required_fields(
        phase_gate_selection,
        required_fields=(
            "selected_case_role",
            "available_case_roles",
            "ordered_ladder",
            "conditional_selection",
        ),
        artifact_name=f"{CANONICAL_STAGE_PLAN_NAME} phase_gate_selection",
    )

    selected_case_role = str(phase_gate_selection["selected_case_role"])
    if selected_case_role != resolved_case.case_role:
        raise AuthoritySelectionError(
            f"{CANONICAL_STAGE_PLAN_NAME} selected_case_role {selected_case_role!r} "
            f"must match case_role {resolved_case.case_role!r}"
        )

    conditional_selection = phase_gate_selection["conditional_selection"]
    if not isinstance(conditional_selection, bool):
        raise AuthoritySelectionError(
            f"{CANONICAL_STAGE_PLAN_NAME} conditional_selection must be a boolean"
        )
    conditional_reason = phase_gate_selection.get("conditional_reason")
    if conditional_selection:
        if selected_case_role not in conditional_case_roles:
            raise AuthoritySelectionError(
                f"{CANONICAL_STAGE_PLAN_NAME} conditional_selection is only valid for "
                f"authority-conditional case roles in phase gate {phase_gate!r}"
            )
        if not isinstance(conditional_reason, str) or not conditional_reason.strip():
            raise AuthoritySelectionError(
                f"{CANONICAL_STAGE_PLAN_NAME} conditional_reason must be a non-empty string "
                "when conditional_selection is true"
            )
        allowed_case_roles = allowed_phase_gate_case_roles(
            bundle,
            phase_gate=phase_gate,
            include_conditional=True,
        )
        resolve_phase_gate_case(
            bundle,
            phase_gate=phase_gate,
            case_role=selected_case_role,
            allow_conditional=True,
        )
    else:
        if conditional_reason is not None:
            raise AuthoritySelectionError(
                f"{CANONICAL_STAGE_PLAN_NAME} conditional_reason is only allowed when "
                "conditional_selection is true"
            )
        allowed_case_roles = unconditional_case_roles
        resolve_phase_gate_case(bundle, phase_gate=phase_gate, case_role=selected_case_role)

    available_case_roles_value = phase_gate_selection["available_case_roles"]
    if not isinstance(available_case_roles_value, list):
        raise AuthoritySelectionError(
            f"{CANONICAL_STAGE_PLAN_NAME} available_case_roles must be a list of case roles"
        )
    if any(not isinstance(case_role, str) for case_role in available_case_roles_value):
        raise AuthoritySelectionError(
            f"{CANONICAL_STAGE_PLAN_NAME} available_case_roles must contain only strings"
        )
    if len(available_case_roles_value) != len(set(available_case_roles_value)):
        raise AuthoritySelectionError(
            f"{CANONICAL_STAGE_PLAN_NAME} available_case_roles must not contain duplicates"
        )
    available_case_roles = set(available_case_roles_value)
    if available_case_roles != set(allowed_case_roles):
        raise AuthoritySelectionError(
            f"{CANONICAL_STAGE_PLAN_NAME} available_case_roles for phase gate {phase_gate!r} "
            f"must remain {list(allowed_case_roles)!r}"
        )

    ordered_ladder_value = phase_gate_selection["ordered_ladder"]
    if not isinstance(ordered_ladder_value, list):
        raise AuthoritySelectionError(
            f"{CANONICAL_STAGE_PLAN_NAME} ordered_ladder must be a list of case roles"
        )
    if any(not isinstance(case_role, str) for case_role in ordered_ladder_value):
        raise AuthoritySelectionError(
            f"{CANONICAL_STAGE_PLAN_NAME} ordered_ladder must contain only strings"
        )
    ordered_ladder = tuple(ordered_ladder_value)
    try:
        validate_frozen_ladder(bundle, ordered_ladder)
    except AuthoritySelectionError as exc:
        raise AuthoritySelectionError(
            f"{CANONICAL_STAGE_PLAN_NAME} ordered_ladder must remain "
            + " -> ".join(bundle.ladder.ordered_case_ids)
        ) from exc

    _validate_non_empty_string_field(payload, "baseline", artifact_name=CANONICAL_STAGE_PLAN_NAME)
    _validate_optional_string_field(
        payload,
        "runtime_base",
        artifact_name=CANONICAL_STAGE_PLAN_NAME,
    )
    _validate_optional_string_field(
        payload,
        "reviewed_source_tuple_id",
        artifact_name=CANONICAL_STAGE_PLAN_NAME,
    )
    _validate_provenance_payload(payload, artifact_name=CANONICAL_STAGE_PLAN_NAME)

    stages = payload["stages"]
    if not isinstance(stages, list) or not stages:
        raise AuthoritySelectionError(f"{CANONICAL_STAGE_PLAN_NAME} stages must be a non-empty list")
    for stage in stages:
        if (
            not isinstance(stage, dict)
            or not isinstance(stage.get("name"), str)
            or not stage["name"].strip()
            or not isinstance(stage.get("cmd"), str)
            or not stage["cmd"].strip()
        ):
            raise AuthoritySelectionError(
                f"{CANONICAL_STAGE_PLAN_NAME} each stage must define string name and cmd values"
            )
        if "cwd" in stage and not isinstance(stage["cwd"], str):
            raise AuthoritySelectionError(
                f"{CANONICAL_STAGE_PLAN_NAME} stage cwd must be a string when provided"
            )

    return payload


def build_case_meta_payload(
    bundle: AuthorityBundle,
    *,
    context: StageRunnerContext,
    case_role: str,
    requested_vof_solver_mode: str,
    resolved_vof_solver_exec: str,
    resolved_pressure_backend: str,
    mesh: Mapping[str, Any],
    numerics: Mapping[str, Any],
    startup: Mapping[str, Any],
    pressure: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_case = resolve_reference_case(bundle, case_role=case_role)
    probe_payload = _read_json_dict(context.probe_payload_path)
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "case_id": resolved_case.frozen_id,
        "case_role": resolved_case.case_role,
        "ladder_position": resolved_case.ladder_position,
        "phase_gates": list(resolved_case.phase_gates),
        "baseline": context.baseline_name,
        "runtime_base": context.runtime_base,
        "reviewed_source_tuple_id": context.reviewed_source_tuple_id,
        "requested_vof_solver_mode": requested_vof_solver_mode,
        "resolved_vof_solver_exec": resolved_vof_solver_exec,
        "resolved_pressure_backend": resolved_pressure_backend,
        "openfoam_bashrc_used": context.bashrc_path,
        "available_commands": _extract_available_commands(probe_payload),
        "mesh_full_360": _require_mapping_value(mesh, "mesh_full_360", artifact_name=CANONICAL_CASE_META_NAME),
        "mesh_resolution_scale": _require_mapping_value(mesh, "mesh_resolution_scale", artifact_name=CANONICAL_CASE_META_NAME),
        "hydraulic_domain_mode": _require_mapping_value(mesh, "hydraulic_domain_mode", artifact_name=CANONICAL_CASE_META_NAME),
        "near_field_radius_d": _require_mapping_value(mesh, "near_field_radius_d", artifact_name=CANONICAL_CASE_META_NAME),
        "near_field_length_d": _require_mapping_value(mesh, "near_field_length_d", artifact_name=CANONICAL_CASE_META_NAME),
        "steady_end_time_iter": _require_mapping_value(mesh, "steady_end_time_iter", artifact_name=CANONICAL_CASE_META_NAME),
        "steady_write_interval_iter": _require_mapping_value(mesh, "steady_write_interval_iter", artifact_name=CANONICAL_CASE_META_NAME),
        "steady_turbulence_model": _require_mapping_value(mesh, "steady_turbulence_model", artifact_name=CANONICAL_CASE_META_NAME),
        "vof_turbulence_model": _require_mapping_value(mesh, "vof_turbulence_model", artifact_name=CANONICAL_CASE_META_NAME),
        "delta_t_s": _require_mapping_value(numerics, "delta_t_s", artifact_name=CANONICAL_CASE_META_NAME),
        "write_interval_s": _require_mapping_value(numerics, "write_interval_s", artifact_name=CANONICAL_CASE_META_NAME),
        "end_time_s": _require_mapping_value(numerics, "end_time_s", artifact_name=CANONICAL_CASE_META_NAME),
        "max_co": _require_mapping_value(numerics, "max_co", artifact_name=CANONICAL_CASE_META_NAME),
        "max_alpha_co": _require_mapping_value(numerics, "max_alpha_co", artifact_name=CANONICAL_CASE_META_NAME),
        "resolved_direct_slot_numerics": _require_mapping_value(
            numerics,
            "resolved_direct_slot_numerics",
            artifact_name=CANONICAL_CASE_META_NAME,
        ),
        "startup_fill_extension_d": _require_mapping_value(
            startup,
            "startup_fill_extension_d",
            artifact_name=CANONICAL_CASE_META_NAME,
        ),
        "air_core_seed_radius_d_requested": _require_mapping_value(
            startup,
            "air_core_seed_radius_d_requested",
            artifact_name=CANONICAL_CASE_META_NAME,
        ),
        "air_core_seed_radius_m_resolved": _require_mapping_value(
            startup,
            "air_core_seed_radius_m_resolved",
            artifact_name=CANONICAL_CASE_META_NAME,
        ),
        "air_core_seed_cap_applied": _require_mapping_value(
            startup,
            "air_core_seed_cap_applied",
            artifact_name=CANONICAL_CASE_META_NAME,
        ),
        "fill_radius_m_resolved": _require_mapping_value(
            startup,
            "fill_radius_m_resolved",
            artifact_name=CANONICAL_CASE_META_NAME,
        ),
        "fill_z_start_m": _require_mapping_value(startup, "fill_z_start_m", artifact_name=CANONICAL_CASE_META_NAME),
        "fill_z_stop_m": _require_mapping_value(startup, "fill_z_stop_m", artifact_name=CANONICAL_CASE_META_NAME),
        "DeltaP_Pa": _require_mapping_value(pressure, "DeltaP_Pa", artifact_name=CANONICAL_CASE_META_NAME),
        "DeltaP_effective_Pa": _require_mapping_value(
            pressure,
            "DeltaP_effective_Pa",
            artifact_name=CANONICAL_CASE_META_NAME,
        ),
        "check_valve_loss_applied": _require_mapping_value(
            pressure,
            "check_valve_loss_applied",
            artifact_name=CANONICAL_CASE_META_NAME,
        ),
        "provenance": _build_provenance_payload(context, extra=provenance),
    }
    return validate_case_meta(bundle, payload)


def build_stage_plan_payload(
    bundle: AuthorityBundle,
    *,
    context: StageRunnerContext,
    case_role: str,
    phase_gate: str,
    stages: Sequence[Mapping[str, Any]],
    conditional_reason: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_case = resolve_reference_case(bundle, case_role=case_role)
    unconditional_case_roles = allowed_phase_gate_case_roles(bundle, phase_gate=phase_gate)
    allowed_case_roles = allowed_phase_gate_case_roles(
        bundle,
        phase_gate=phase_gate,
        include_conditional=True,
    )
    conditional_selection = case_role not in unconditional_case_roles
    phase_gate_selection = {
        "selected_case_role": resolved_case.case_role,
        "available_case_roles": list(
            allowed_case_roles if conditional_selection else unconditional_case_roles
        ),
        "ordered_ladder": list(bundle.ladder.ordered_case_ids),
        "conditional_selection": conditional_selection,
    }
    if conditional_reason is not None:
        phase_gate_selection["conditional_reason"] = conditional_reason

    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "case_id": resolved_case.frozen_id,
        "case_role": resolved_case.case_role,
        "phase_gate": phase_gate,
        "baseline": context.baseline_name,
        "runtime_base": context.runtime_base,
        "reviewed_source_tuple_id": context.reviewed_source_tuple_id,
        "provenance": _build_provenance_payload(context, extra=provenance),
        "phase_gate_selection": phase_gate_selection,
        "stages": [dict(stage) for stage in stages],
    }
    return validate_stage_plan(bundle, payload)


def emit_case_bundle(
    bundle: AuthorityBundle,
    *,
    output_dir: pathlib.Path | str,
    case_meta: Mapping[str, Any],
    stage_plan: Mapping[str, Any],
) -> EmittedCaseArtifacts:
    validated_case_meta = validate_case_meta(bundle, dict(case_meta))
    validated_stage_plan = validate_stage_plan(bundle, dict(stage_plan))
    for field_name in ("case_id", "case_role", "baseline"):
        if validated_case_meta[field_name] != validated_stage_plan[field_name]:
            raise AuthoritySelectionError(
                f"case bundle {field_name} mismatch between {CANONICAL_CASE_META_NAME} "
                f"and {CANONICAL_STAGE_PLAN_NAME}"
            )

    target_dir = pathlib.Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    case_meta_path = target_dir / CANONICAL_CASE_META_NAME
    stage_plan_path = target_dir / CANONICAL_STAGE_PLAN_NAME
    _write_json(case_meta_path, validated_case_meta)
    _write_json(stage_plan_path, validated_stage_plan)
    return EmittedCaseArtifacts(
        case_meta_path=case_meta_path,
        stage_plan_path=stage_plan_path,
    )


def _build_resolved_case(
    bundle: AuthorityBundle, *, case_role: str, reference_case: ReferenceCase
) -> ResolvedReferenceCase:
    return ResolvedReferenceCase(
        case_role=case_role,
        frozen_id=reference_case.frozen_id,
        purpose=reference_case.purpose,
        frozen_default_contract=dict(reference_case.frozen_default_contract),
        ladder_position=bundle.ladder.ordered_case_ids.index(case_role) + 1,
        phase_gates=_phase_gates_for_case_role(bundle, case_role=case_role),
    )


def _phase_gates_for_case_role(bundle: AuthorityBundle, *, case_role: str) -> tuple[str, ...]:
    phase_gates = []
    for phase_gate in bundle.cases.phase_gate_mapping:
        if case_role in allowed_phase_gate_case_roles(
            bundle,
            phase_gate=phase_gate,
            include_conditional=True,
        ):
            phase_gates.append(phase_gate)
    return tuple(phase_gates)


def _case_meta_variants(bundle: AuthorityBundle) -> list[dict[str, Any]]:
    variants = []
    for case_role in bundle.ladder.ordered_case_ids:
        resolved_case = resolve_reference_case(bundle, case_role=case_role)
        variants.append(
            {
                "properties": {
                    "case_role": {"const": resolved_case.case_role},
                    "case_id": {"const": resolved_case.frozen_id},
                    "ladder_position": {"const": resolved_case.ladder_position},
                    "phase_gates": _exact_membership_array_schema(resolved_case.phase_gates),
                }
            }
        )
    return variants


def _stage_plan_variants(bundle: AuthorityBundle) -> list[dict[str, Any]]:
    variants = []
    for phase_gate in bundle.cases.phase_gate_mapping:
        unconditional_case_roles = allowed_phase_gate_case_roles(bundle, phase_gate=phase_gate)
        for case_role in unconditional_case_roles:
            resolved_case = resolve_phase_gate_case(bundle, phase_gate=phase_gate, case_role=case_role)
            variants.append(
                {
                    "properties": {
                        "phase_gate": {"const": phase_gate},
                        "case_role": {"const": resolved_case.case_role},
                        "case_id": {"const": resolved_case.frozen_id},
                        "phase_gate_selection": {
                            "properties": {
                                "selected_case_role": {"const": resolved_case.case_role},
                                "available_case_roles": _exact_membership_array_schema(
                                    unconditional_case_roles
                                ),
                                "conditional_selection": {"const": False},
                            }
                        },
                    }
                }
            )

        conditional_case_roles = tuple(
            case_role
            for case_role in allowed_phase_gate_case_roles(
                bundle,
                phase_gate=phase_gate,
                include_conditional=True,
            )
            if case_role not in unconditional_case_roles
        )
        allowed_with_conditionals = allowed_phase_gate_case_roles(
            bundle,
            phase_gate=phase_gate,
            include_conditional=True,
        )
        for case_role in conditional_case_roles:
            resolved_case = resolve_phase_gate_case(
                bundle,
                phase_gate=phase_gate,
                case_role=case_role,
                allow_conditional=True,
            )
            variants.append(
                {
                    "properties": {
                        "phase_gate": {"const": phase_gate},
                        "case_role": {"const": resolved_case.case_role},
                        "case_id": {"const": resolved_case.frozen_id},
                        "phase_gate_selection": {
                            "properties": {
                                "selected_case_role": {"const": resolved_case.case_role},
                                "available_case_roles": _exact_membership_array_schema(
                                    allowed_with_conditionals
                                ),
                                "conditional_selection": {"const": True},
                            }
                        },
                    }
                }
            )
    return variants


def _exact_membership_array_schema(values: Sequence[str]) -> dict[str, Any]:
    ordered_values = list(values)
    return {
        "type": "array",
        "minItems": len(ordered_values),
        "maxItems": len(ordered_values),
        "uniqueItems": True,
        "items": {"enum": ordered_values},
        "allOf": [{"contains": {"const": value}} for value in ordered_values],
    }


def _validate_required_fields(
    payload: dict[str, Any], *, required_fields: Sequence[str], artifact_name: str
) -> None:
    missing_fields = [field_name for field_name in required_fields if field_name not in payload]
    if missing_fields:
        raise AuthoritySelectionError(
            f"{artifact_name} is missing required fields: {', '.join(missing_fields)}"
        )


def _validate_schema_version(payload: dict[str, Any], *, artifact_name: str) -> None:
    schema_version = str(payload["schema_version"])
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise AuthoritySelectionError(
            f"{artifact_name} has unsupported schema_version {schema_version!r}; "
            f"expected {MANIFEST_SCHEMA_VERSION!r}"
        )


def _require_mapping_value(
    payload: Mapping[str, Any],
    field_name: str,
    *,
    artifact_name: str,
) -> Any:
    if field_name not in payload:
        raise AuthoritySelectionError(
            f"{artifact_name} is missing required fields: {field_name}"
        )
    return payload[field_name]


def _validate_non_empty_string_field(
    payload: Mapping[str, Any],
    field_name: str,
    *,
    artifact_name: str,
) -> None:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise AuthoritySelectionError(
            f"{artifact_name} {field_name} must be a non-empty string"
        )


def _validate_optional_string_field(
    payload: Mapping[str, Any],
    field_name: str,
    *,
    artifact_name: str,
) -> None:
    value = payload.get(field_name)
    if value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise AuthoritySelectionError(
            f"{artifact_name} {field_name} must be a non-empty string when provided"
        )


def _validate_numeric_field(
    payload: Mapping[str, Any],
    field_name: str,
    *,
    artifact_name: str,
) -> None:
    value = payload.get(field_name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise AuthoritySelectionError(f"{artifact_name} {field_name} must be numeric")


def _validate_optional_numeric_field(
    payload: Mapping[str, Any],
    field_name: str,
    *,
    artifact_name: str,
) -> None:
    value = payload.get(field_name)
    if value is None:
        return
    _validate_numeric_field(payload, field_name, artifact_name=artifact_name)


def _validate_optional_integer_field(
    payload: Mapping[str, Any],
    field_name: str,
    *,
    artifact_name: str,
) -> None:
    value = payload.get(field_name)
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool):
        raise AuthoritySelectionError(f"{artifact_name} {field_name} must be an integer")


def _validate_boolean_field(
    payload: Mapping[str, Any],
    field_name: str,
    *,
    artifact_name: str,
) -> None:
    if not isinstance(payload.get(field_name), bool):
        raise AuthoritySelectionError(f"{artifact_name} {field_name} must be a boolean")


def _validate_int_enum_field(
    payload: Mapping[str, Any],
    field_name: str,
    *,
    allowed_values: Sequence[int],
    artifact_name: str,
) -> None:
    value = payload.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool) or value not in allowed_values:
        raise AuthoritySelectionError(
            f"{artifact_name} {field_name} must be one of {list(allowed_values)!r}"
        )


def _validate_mapping_field(
    payload: Mapping[str, Any],
    field_name: str,
    *,
    artifact_name: str,
) -> None:
    value = payload.get(field_name)
    if not isinstance(value, dict) or not value:
        raise AuthoritySelectionError(f"{artifact_name} {field_name} must be a non-empty object")


def _validate_string_mapping_field(
    payload: Mapping[str, Any],
    field_name: str,
    *,
    artifact_name: str,
) -> None:
    value = payload.get(field_name)
    if not isinstance(value, dict) or not value:
        raise AuthoritySelectionError(f"{artifact_name} {field_name} must be a non-empty object")
    for key, entry in value.items():
        if not isinstance(key, str) or not key.strip():
            raise AuthoritySelectionError(
                f"{artifact_name} {field_name} must use non-empty string keys"
            )
        if not isinstance(entry, str) or not entry.strip():
            raise AuthoritySelectionError(
                f"{artifact_name} {field_name} must use non-empty string values"
            )


def _validate_provenance_payload(payload: Mapping[str, Any], *, artifact_name: str) -> None:
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise AuthoritySelectionError(f"{artifact_name} provenance must be an object")
    _validate_required_fields(
        provenance,
        required_fields=REQUIRED_PROVENANCE_FIELDS,
        artifact_name=f"{artifact_name} provenance",
    )
    for field_name in REQUIRED_PROVENANCE_FIELDS:
        _validate_non_empty_string_field(
            provenance,
            field_name,
            artifact_name=f"{artifact_name} provenance",
        )


def _build_provenance_payload(
    context: StageRunnerContext,
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "probe_payload": context.probe_payload_path.as_posix(),
        "host_env": context.host_env_path.as_posix(),
        "manifest_refs": context.manifest_refs_path.as_posix(),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def _extract_available_commands(probe_payload: Mapping[str, Any]) -> dict[str, str]:
    commands = probe_payload.get("commands")
    if not isinstance(commands, Mapping):
        return {}
    filtered_commands: dict[str, str] = {}
    for raw_name, raw_path in commands.items():
        name = str(raw_name).strip()
        if isinstance(raw_path, Mapping):
            path = str(raw_path.get("path", "")).strip()
        else:
            path = str(raw_path).strip()
        if name and path:
            filtered_commands[name] = path
    return filtered_commands


def _read_json_dict(path: pathlib.Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AuthoritySelectionError(f"unable to read JSON payload: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AuthoritySelectionError(f"invalid JSON payload: {path}") from exc
    if not isinstance(payload, dict):
        raise AuthoritySelectionError(f"JSON payload must be an object: {path}")
    return payload


def _write_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
