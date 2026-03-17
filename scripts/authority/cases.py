"""Reference-case resolution and metadata-schema helpers for foundation tooling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .bundle import AuthorityBundle, ReferenceCase


MANIFEST_SCHEMA_VERSION = "1.0.0"
CANONICAL_CASE_META_NAME = "case_meta.json"
CANONICAL_STAGE_PLAN_NAME = "stage_plan.json"


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
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "canonical_name": CANONICAL_CASE_META_NAME,
        "type": "object",
        "required": [
            "schema_version",
            "case_id",
            "case_role",
            "ladder_position",
            "phase_gates",
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
                "items": {"enum": list(bundle.cases.phase_gate_mapping)},
                "uniqueItems": True,
            },
        },
    }


def validate_case_meta(bundle: AuthorityBundle, payload: dict[str, Any]) -> dict[str, Any]:
    _validate_required_fields(
        payload,
        required_fields=(
            "schema_version",
            "case_id",
            "case_role",
            "ladder_position",
            "phase_gates",
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

    return payload


def stage_plan_schema(bundle: AuthorityBundle) -> dict[str, Any]:
    ordered_case_roles = list(bundle.ladder.ordered_case_ids)
    return {
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
                    "conditional_reason": {"type": "string"},
                },
            },
            "stages": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["name", "cmd"],
                    "properties": {
                        "name": {"type": "string"},
                        "cmd": {"type": "string"},
                        "cwd": {"type": "string"},
                    },
                },
            },
        },
    }


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
