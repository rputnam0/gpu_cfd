# Reference Case Contract

This file freezes the case ladder consumed across Phases 0-8. It is the authoritative source for case IDs, roles, and phase-gate usage. The machine-readable companion is `reference_case_contract.json`.

## Frozen Cases

| Case | Frozen ID | Purpose | Frozen default contract |
|---|---|---|---|
| `R0` | `phase0_r0_57_28_1000_full360_v1` | Representative production reference | `57-28 @ 1000 psi`, full 360, algebraic `incompressibleVoF` / explicit MULES / PIMPLE, `hydraulic_cfd_mode=vof_transient_preconditioned`, `geometry_mode=sk_standard_shared_body_with_coeffs_v2`, `startup_fill_extension_d=0.0`, `air_core_seed_radius_d=1.0`, startup conditioning enabled, compact near-field plume. |
| `R1` | `phase0_r1_57_28_1000_internal_v1` | Reduced nozzle development reference | Same nozzle and solver family as `R0`, `hydraulic_domain_mode=internal_only`, reduced resolution for fast iteration, startup conditioning retained, nozzle-specific boundary behavior retained. |
| `R1-core` | `phase0_r1_core_57_28_1000_internal_generic_v1` | Phase-5-friendly reduced generic VOF case | Same algebraic solver family and internal-only geometry family as `R1`, but restricted to the generic Phase 5 support subset: no custom nozzle inlet BC, no Phase 6-only startup seeding dependency, no contact-angle, no coupled/processor patches, no Phase 6-only pressure or ambient/open BC requirements. |
| `R2` | `phase0_r2_dambreak_reference_v1` | Generic VOF verification anchor | Existing `cfd/smoke/damBreak_amgx` intent, with a documented native-CPU pressure fallback on the SPUMA/v2412 CPU comparison line if the external solver stack is unavailable. |

## Phase-Gate Mapping

- Phase 0 freezes `R2 -> R1-core -> R1 -> R0` on Baseline A and Baseline B.
- Phase 2 uses `R2` plus `R1-core` by default. `R1` is used only when nozzle-specific topology or patch-manifest coverage is intentionally under test.
- Phase 5 hard gates use two generic `R2` slices plus `R1-core`.
- Phase 6 and Phase 7 use `R1` as the reduced nozzle acceptance case.
- Phase 8 uses `R1` for routine architectural baselines, `R0` for production-shape acceptance, and `R1-core` whenever backend or execution-mode parity is needed without Phase 6 nozzle-specific BCs.

## Locked Defaults

- Hard-gating `R0` case: `57-28 @ 1000 psi`.
- `40-28 @ 1000 psi` remains an optional shadow/reference case and is not a milestone-1 hard gate.
- Pressure-drop is not a Phase 0 case-freeze selector. Formal later-phase pressure-drop gates, when present, are owned exclusively by `acceptance_manifest.md` / `acceptance_manifest.json`.
- `R1-core` is mandatory in the ladder and may not be replaced by a descriptive “reduced generic case” label.
