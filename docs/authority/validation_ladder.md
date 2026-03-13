# Validation Ladder

This file is the package-level navigation alias for the frozen validation ladder.

The authoritative owner of case IDs, case roles, and phase-gate usage is [reference_case_contract.md](reference_case_contract.md).

## Frozen Ladder

1. `R2` — generic VOF verification anchor
2. `R1-core` — Phase-5-friendly reduced generic case
3. `R1` — reduced nozzle development reference
4. `R0` — representative production reference

## Usage Rule

- When a phase document references the “validation ladder,” it means this frozen `R2 -> R1-core -> R1 -> R0` sequence as defined in `reference_case_contract.md`.
- Phase documents may select which rung they consume, but they may not rename, reorder, or replace ladder members locally.
