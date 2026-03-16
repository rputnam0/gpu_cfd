# Phase Specs

This folder holds the phase-by-phase implementation specs that consume the authority bundle.

Do not read the full spec set by default. Start from the assigned PR card, then open only the exact
spec subsections cited by that card or by an unresolved boundary issue.

## How To Use These Specs

1. Resolve the PR card from `docs/tasks/pr_inventory.md`.
2. Treat the exact PR card as the execution contract.
3. Open only the spec subsections cited by that card.
4. Expand to adjacent sections only when the cited text is insufficient to resolve the task safely.

## Spec Guide

- `phase0_reference_problem_spec.md`: CPU reference freeze, baseline contracts, and Phase 0 execution details
- `phase1_blackwell_bringup_spec.md`: Blackwell bring-up, build validation, and early profiling groundwork
- `phase2_gpu_memory_spec.md`: memory model, visibility rules, staging, and residency contracts
- `phase3_execution_model_spec.md`: execution modes, stage taxonomy, stream/graph orchestration, and launch rules
- `phase4_linear_algebra_spec.md`: pressure linear-algebra path, AmgX bridge, and matrix staging
- `phase5_spuma_nozzle_spec.md`: generic VOF core, pressure coupling, and milestone-1 solver semantics
- `phase6_pressure_swirl_nozzle_bc_spec.md`: nozzle boundary conditions, startup semantics, and Phase 6 acceptance
- `phase7_custom_cuda_kernel_spec.md`: custom kernel scope, hotspot replacement, and graph-safety cleanup
- `phase8_profiling_performance_acceptance_spec.md`: profiling substrate, acceptance, and baseline lock criteria

