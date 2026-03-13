# Support Matrix

This file is the authoritative package-level support envelope for milestone-1 implementation. Phase files consume it; they do not reopen it locally. The machine-readable companion is `support_matrix.json`; automation must consume that JSON companion rather than scrape these Markdown tables.

## Global Policy

- Static mesh only.
- Single region only.
- No processor patches, cyclic/AMI patches, or arbitrary coded patch fields in milestone-1 production scope.
- Contact-angle is out of milestone-1 scope.
- Surface-tension scope is constant `sigma` only.
- Turbulence scope is laminar-only.
- Default fallback policy is `failFast`.
- CPU/stage fallback, host patch execution, and host `setFields` startup are debug-only/bring-up-only modes and are forbidden in production acceptance.
- In performance mode, only functionObjects classified `writeTimeOnly` are allowed. `debugOnly` entries are allowed only in explicit debug runs.

## Phase 5 Generic VOF Envelope

- Scalars: `fixedValue`, `zeroGradient`, `calculated`.
- Vectors: `fixedValue`, `noSlip`, `slip`, `symmetryPlane`, `calculated`.
- Geometry patch families: `wall`, `patch`, `symmetryPlane`, `empty`, `wedge`.
- Schemes/runtime policy: Euler ddt, generic BC/scheme subset only, no contact-angle patch fields, no nozzle-specific swirl inlet logic, no `pressureInletOutletVelocity`, no `fixedFluxPressure`, no processor/coupled patches.

## Exact Audited Scheme Tuple

The startup support scanner and the Phase 7 custom-kernel path must treat any deviation from this tuple as unsupported in milestone-1 production mode.

| Block | Exact allowed entry | Scope |
|---|---|---|
| `ddtSchemes.default` | `Euler` | all accepted milestone-1 cases |
| `gradSchemes.default` | `Gauss linear` | generic gradient fallback for accepted cases |
| `gradSchemes.grad(alpha1)` | `Gauss linear` | alpha / interface pipeline |
| `interpolationSchemes.default` | `linear` | accepted milestone-1 cases |
| `interpolationSchemes.interpolate(grad(alpha1))` | `linear` | interface-normal construction |
| `interpolationSchemes.interpolate(sigmaK)` | `linear` | surface-tension force path |
| `divSchemes.div(phi,alpha1)` | `Gauss upwind` | accepted alpha-convection baseline |
| `divSchemes.div(phirb,alpha1)` | `Gauss interfaceCompression` | accepted compressive alpha path |

Notes:

- `alpha1` here is generic transport shorthand. In the frozen nozzle family (`R0`, `R1`, `R1-core`), the liquid volume-fraction field is `alpha.water`.
- `localEuler`, `CrankNicolson`, alternative alpha convection schemes, and any unlisted `grad` / `interpolate` / `div` entry are unsupported in milestone-1 production mode.
- Momentum and pressure-side numerics outside the alpha/interface tuple remain whatever is frozen in the checked-in Phase 0 case bundles; they are not widened locally by later GPU phases.

## FunctionObject Classification

The support scanner must classify functionObjects exactly as follows. Any class not listed here is `unsupported`.

| FunctionObject class | Classification | Production-mode rule |
|---|---|---|
| `fieldMinMax` | `writeTimeOnly` | allowed only when execution is restricted to write times |
| `volFieldValue` | `writeTimeOnly` | allowed only when execution is restricted to write times |
| `surfaceFieldValue` | `writeTimeOnly` | allowed only when execution is restricted to write times |
| `writeObjects` | `writeTimeOnly` | allowed only when execution is restricted to write times |
| `solverInfo` | `debugOnly` | allowed only in explicit debug / bring-up runs |
| `residuals` | `debugOnly` | allowed only in explicit debug / bring-up runs |
| `probes` | `debugOnly` | allowed only in explicit debug / bring-up runs |
| `patchProbes` | `debugOnly` | allowed only in explicit debug / bring-up runs |
| `forces` | `debugOnly` | allowed only in explicit debug / bring-up runs |
| `forceCoeffs` | `debugOnly` | allowed only in explicit debug / bring-up runs |
| `vorticity` | `debugOnly` | allowed only in explicit debug / bring-up runs |
| `Q` | `debugOnly` | allowed only in explicit debug / bring-up runs |
| `lambda2` | `debugOnly` | allowed only in explicit debug / bring-up runs |
| `streamLines` | `debugOnly` | allowed only in explicit debug / bring-up runs |
| `isoSurface` | `debugOnly` | allowed only in explicit debug / bring-up runs |
| `yPlus` | `debugOnly` | allowed only in explicit debug / bring-up runs |
| `coded`, `systemCall`, `python`, any unlisted class | `unsupported` | reject before the first timestep |

Rules:

- In performance mode, `writeTimeOnly` means the object executes only on accepted write events; any every-timestep or substep execution is unsupported.
- `debugOnly` objects may appear only in explicit debug/bring-up runs and automatically disqualify production acceptance if they trigger host commits inside a timed window.

## Phase 6 Nozzle-Specific Envelope

The frozen milestone-1 `R1` / `R0` nozzle tuple is:
- swirl inlet `U` = `gpuPressureSwirlInletVelocity`
- swirl inlet `p_rgh` = `prghTotalHydrostaticPressure`
- wall `U` = `noSlip`
- wall `p_rgh` = `fixedFluxPressure`
- ambient/open `U` = `pressureInletOutletVelocity`
- ambient/open `p_rgh` = `prghPressure`

| Patch role | Field | Allowed milestone-1 kinds |
|---|---|---|
| Swirl inlet (`swirlInletA/B`) | `U` | `gpuPressureSwirlInletVelocity` |
| Swirl inlet (`swirlInletA/B`) | `p_rgh` | `prghTotalHydrostaticPressure` |
| Swirl inlet (`swirlInletA/B`) | `alpha.water` | `fixedValue` |
| Wall | `U` | `noSlip` |
| Wall | `p_rgh` | `fixedFluxPressure` |
| Wall | `alpha.water` | `zeroGradient` |
| Ambient/open | `U` | `pressureInletOutletVelocity` |
| Ambient/open | `p_rgh` | `prghPressure` |
| Ambient/open | `alpha.water` | `inletOutlet` |
| Symmetry / empty | all relevant fields | pass-through / no-op semantics |

Notes:
- The frozen production token for wall `U` in the `R1` / `R0` nozzle family is `noSlip`.
- A legacy `fixedValue (0 0 0)` wall may be normalized as a compatibility alias to `noSlip`, but it is not a second independently admitted machine-readable kind.

## Canonical Startup-Seed DSL

The centralized support matrix also owns the allowed Phase 6 startup-seeding grammar. Phase 6 imports and implements this DSL; it does not widen it locally. Automation should consume `support_matrix.json` rather than duplicating phase-local prose.

| Element | Frozen rule | Notes |
|---|---|---|
| Canonical owner | `system/gpuRuntimeDict` -> `gpuRuntime.startupSeed` | `gpuStartupSeedDict` is compatibility-shim input only. |
| Top-level keys | `enabled`, `forceReseed`, `precedence`, `defaultFieldValues`, `regions` | Any unlisted top-level key is unsupported in milestone-1 production scope. |
| Precedence policy | `lastWins` only | Apply `defaultFieldValues` first, then apply `regions` in listed order. |
| Default timing | apply once before the first timestep | Restart reseeding is forbidden unless `forceReseed yes`. |
| Allowed field-value entries | `volScalarFieldValue alpha.water <scalar>`; `volVectorFieldValue U (<x> <y> <z>)`; `volScalarFieldValue p_rgh <scalar>` | Any other field name or value class is unsupported. |
| Supported region families | `cylinderToCell`, `frustumToCell`, `boxToCell`, `sphereToCell`, `halfSpaceToCell` | Geometry semantics are the constrained Phase 6 parser/kernel subset and may not be expanded locally. |

Region semantics:
- `cylinderToCell`: axial segment plus constant radius inclusion.
- `frustumToCell`: axial segment plus linearly interpolated `radius1 -> radius2`.
- `boxToCell`: axis-aligned bounds inclusion.
- `sphereToCell`: center-radius inclusion.
- `halfSpaceToCell`: signed plane-distance inclusion.

## Backend and Operational Policy

- Native pressure is the required baseline on every accepted case.
- AmgX is a supported secondary backend only through the Phase 4 bridge. Any AmgX production claim requires `DeviceDirect`; `PinnedHost` is correctness-only bring-up.
- Unsupported cases must fail during startup support scanning before the first timestep.
