# Authority Bundle

This folder is the frozen authority layer for repo-wide decisions. Treat it as the source of truth
for support scope, runtime defaults, validation policy, stage taxonomy, case roles, and semantic
ownership.

Do not load every authority document by default. Start from the exact PR card and read only the
authority docs or JSON companions cited by that card.

## Authority Order

When multiple docs are relevant, use this order to orient yourself:

1. `continuity_ledger.md`
2. `master_pin_manifest.md`
3. `reference_case_contract.md`
4. `validation_ladder.md`
5. `support_matrix.md`
6. `acceptance_manifest.md`
7. `graph_capture_support_matrix.md`
8. `semantic_source_map.md`

## File Guide

- `continuity_ledger.md`: frozen global decisions, ownership boundaries, and package consumption rules
- `master_pin_manifest.md`: toolchain, runtime, source, and profiler pins
- `reference_case_contract.md` and `.json`: canonical case IDs, case roles, and phase-gate mapping
- `validation_ladder.md`: frozen `R2 -> R1-core -> R1 -> R0` ladder alias and usage rule
- `support_matrix.md` and `.json`: supported tuples, runtime policy, and fail-fast rules
- `acceptance_manifest.md` and `.json`: tuple-level acceptance, thresholds, gates, and disposition rules
- `graph_capture_support_matrix.md` and `.json`: canonical stage IDs, run modes, and graph-policy rules
- `semantic_source_map.md`: semantic target mapping for SPUMA/v2412 patch planning

## Consumption Rules

- Authority docs and JSON companions override conflicting phase-local prose.
- JSON companions are the machine-readable source of truth for automation and validators.
- Phase specs consume authority decisions; they do not redefine them.
- Task cards should cite the narrowest exact authority subsection that governs the current work.

