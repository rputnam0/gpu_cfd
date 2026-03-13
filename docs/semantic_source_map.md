# Semantic Source Map

This file is the authoritative semantic handoff for Phases 4-7. Public Foundation/OpenCFD references remain algorithmic guides only; implementation patches target the local SPUMA/v2412 family named here.

## Frozen Mapping

| Contract surface | Semantic reference | Local implementation target family | Notes |
|---|---|---|---|
| Solver family | `incompressibleVoF` / explicit MULES / PIMPLE | SPUMA/OpenFOAM-v2412 incompressible VOF runtime line | Milestone-1 canonical GPU family. |
| Alpha transport | `twoPhaseSolver::alphaPredictor()` | local `alphaPredictor.C` path plus `DeviceAlphaTransport.*` / `DeviceMULES.*` | Preserve alpha subcycling, previous-correction flux handling, and MULES ordering. |
| Momentum predictor | solver momentum stage | local momentum stage path plus `DeviceMomentumPredictor.*` | Keep sequencing relative to mixture/interface updates. |
| Pressure corrector | `twoPhaseSolver::pressureCorrector()` | local `pressureCorrector.C` path plus `DevicePressureCorrector.*` | Native baseline; AmgX enters only through the Phase 4 bridge. |
| Interface properties | `interfaceProperties` and `surfaceTensionModel` | local interface-properties path plus `DeviceSurfaceTension.*` | Milestone-1 scope is constant `sigma` and no contact-angle. |
| Pressure bridge | linear-solver boundary and matrix staging | `PressureMatrixCache`, `packDeviceStaging(...)`, `DeviceDirect`, existing `foamExternalSolvers` AmgX bridge | `PinnedHost` is correctness-only. |
| Nozzle BC runtime selection | pressure-swirl inlet and pressure boundary handling | `gpuPressureSwirlInletVelocityFvPatchVectorField.*`, `BoundaryExecutor.*`, `PressureBoundaryState.*`, `StartupSeeder.*` | Physics/math is frozen by Phase 6 and imported by Phase 7. |
| Profiling/instrumentation touch points | stage taxonomy and runtime config | `deviceIncompressibleVoF.C`, `deviceAlphaPredictor.C`, `deviceMomentumPredictor.C`, `devicePressureCorrector.C`, `deviceSurfaceTension.C`, `deviceNozzleBoundary.C`, `GpuProfilingConfig.*` | Required stage-level parent ranges must use the stage IDs from [graph_capture_support_matrix.md](graph_capture_support_matrix.md) exactly. Finer-grained child ranges may nest beneath those parents for diagnosis, but they are never acceptance keys. |

## Implementation Rule

- When public reference file paths differ from the checked-out SPUMA/v2412 tree, patch the local SPUMA/v2412 equivalents named by responsibility, not the public reference source.
- Any future expansion of this map must preserve the milestone-1 solver family, runtime schema, and backend/default policies frozen elsewhere in `final/`.
