# Phase 6 — Pressure-swirl nozzle boundary conditions and startup

Phase 6 owns `P6-01..P6-10` for nozzle-specific boundary conditions and startup seeding on top of the frozen Phase 5 generic core. This section consumes upstream contracts as fixed inputs, keeps `R1-core` generic-only, and exports nozzle semantics as non-widening contracts for Phase 7 optimization and Phase 8 acceptance.

## P6-01 Boundary support report and patch classifier

- Objective:
  - Implement deterministic startup support classification and boundary manifest substrate for the frozen `R1`/`R0` nozzle patch tuple with fail-fast rejection of unsupported configurations.
- Exact citations:
  - Authority:
    - `docs/authority/support_matrix.md` -> `## Global Policy`
    - `docs/authority/support_matrix.md` -> `## Phase 6 Nozzle-Specific Envelope`
    - `docs/authority/support_matrix.json` -> `phase6_nozzle_specific_envelope`
    - `docs/authority/support_matrix.json` -> `accepted_nozzle_case_family_boundary_tuple`
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (support-matrix ownership, forbidden host patch polymorphism in production acceptance)
    - `docs/authority/reference_case_contract.md` -> `## Phase-Gate Mapping` (Phase 6 accepted case = `R1`)
    - `docs/tasks/boundary_matrix.md` -> `### Phase 5 -> Phase 6`
    - `docs/tasks/07_phase5_generic_vof_core.md` -> `## P5-11 Write-time commit, validation artifacts, and Phase 5 baseline freeze`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `### Design decisions` (`D3`, `D4`, `D9`)
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### Supported boundary-condition matrix`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### 1. Module boundaries`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### Step 1 — Add startup validation and support reporting`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `### Acceptance checklist`
  - Backlog scope:
    - "Implement the boundary support report, patch classifier, and manifest substrate for the frozen nozzle patch set."
  - Backlog done_when:
    - "The branch can classify the R1/R0 boundary set up front and reject unsupported nozzle configurations deterministically."
- Depends on (backlog IDs):
  - `FND-04`
  - `P5-11`
- Prerequisites:
  - Foundation support scanner/fail-fast taxonomy from `FND-04`.
  - Phase 5 baseline freeze packet and seam handoff from `P5-11`, including `case_meta.json` patch-role metadata and generic/nozzle boundary guardrails.
- Concrete task slices:
  1. Implement startup boundary-role classifier for `swirlInletA/B`, `wall`, `ambient/open`, `symmetry`, and `empty` against support-matrix-admitted field/kind tuples only.
  2. Emit deterministic `BoundarySupportReport` with admitted/denied patch rows, fail-fast reason codes, and explicit fallback-policy fields.
  3. Build host-side manifest substrate schema (patch ranges, role-kind bindings, parameter table indexes) consumed by downstream upload and kernel dispatch.
  4. Reject unsupported patch families (`processor`, `cyclic/AMI`, arbitrary coded patch fields) before first timestep in production configuration.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `BoundarySupportReport`
    - boundary-role classifier contract
    - manifest-substrate schema for Phase 6 boundary execution
  - Consumed:
    - `FND-04` support taxonomy and deterministic fail-fast policy
    - `P5-11` seam packet and generic-only boundary ownership statement
- Validation:
  - `R1` and `R0` startup classifications cover the entire patch set deterministically with zero unsupported rows in accepted configurations.
  - Deliberately unsupported patch tuples fail before first timestep with stable reason codes.
  - Production mode records no CPU boundary fallback permission by default.
- Done criteria:
  - Boundary support report and classifier are live and deterministic for frozen nozzle cases.
  - Unsupported nozzle configurations are rejected up front with no silent host fallback path.
- Exports to later PRs/phases:
  - Boundary role/kind manifest substrate consumed by `P6-02`, `P6-03`, `P6-05..P6-09`.
  - Deterministic support-report artifact consumed by `P6-10` acceptance evidence and Phase 7 seam review.

## P6-02 Flat boundary spans and boundary metadata upload

- Objective:
  - Expose contiguous device boundary spans and uploadable metadata views so all Phase 6 boundary execution writes directly into authoritative field storage without host patch polymorphism.
- Exact citations:
  - Authority:
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (no field-scale host evaluation in hot stages; normalized runtime ownership)
    - `docs/authority/support_matrix.md` -> `## Global Policy`
    - `docs/authority/graph_capture_support_matrix.md` -> `## Global Capture Rules`
    - `docs/tasks/boundary_matrix.md` -> `### Phase 5 -> Phase 6`
    - `docs/tasks/04_phase2_gpu_memory_model.md` -> `## P2-08 Mesh mirror and startup registration`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `### Design decisions` (`D4`, `D5`, `D11`, `D12`)
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### 2. Required immutable device arrays`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### 4. Data-layout decisions`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### Step 2 — Expose flat boundary spans in the device field layer`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### Step 3 — Implement host-side patch manifest builder`
  - Backlog scope:
    - "Add the flat boundary-span representation and uploadable metadata views required by device-side boundary execution."
  - Backlog done_when:
    - "Supported nozzle cases expose flat boundary spans and patch metadata without falling back to host patch polymorphism."
- Depends on (backlog IDs):
  - `P6-01`
  - `P2-08`
- Prerequisites:
  - `P6-01` patch classification and manifest-substrate schema.
  - `P2-08` mesh mirror/startup registration contracts and boundary-index stability guarantees.
- Concrete task slices:
  1. Extend Phase 5 field abstractions with contiguous boundary-span views for `U`, `p_rgh`, `alpha.water`, `phi`, `rho`, and `rhorAU`/`rAU` boundary faces.
  2. Build and upload `DeviceBoundaryManifestView` (SoA geometry + range tables + field-boundary flat offsets) from startup host manifest.
  3. Add patch-local-to-flat-index mapping validation to prevent stale or aliased boundary writes.
  4. Integrate allocator and upload lifetime rules so manifests persist and remain capture-safe across steady-state steps.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `DeviceBoundaryManifestView`
    - flat boundary-span mapping tables and upload contract
  - Consumed:
    - `P2-08` mesh mirror and startup registration substrate
    - `P6-01` boundary manifest-substrate and support report outputs
- Validation:
  - Device test writes and reads one boundary face via flat spans with no hidden host copy path.
  - Nsight traces for boundary-span tests show no unexpected HtoD/DtoH transfers and no CPU patch execution.
  - Uploaded metadata indices match per-patch face counts and field offsets for `R1` and `R0`.
- Done criteria:
  - Flat boundary spans and uploadable metadata are available for all supported nozzle patch roles.
  - Device-side boundary execution can proceed without host patch polymorphism fallback.
- Exports to later PRs/phases:
  - Boundary manifest/device-view contract consumed by `P6-05`, `P6-06`, `P6-07`, `P6-08`, and `P6-10`.
  - Flat-span contract exported unchanged to Phase 7 custom-kernel replacement scope.

## P6-03 Constrained profile grammar and compiler

- Objective:
  - Implement deterministic parser/compiler for the frozen inlet profile subset used by nozzle swirl boundary execution and CPU/GPU snapshot parity.
- Exact citations:
  - Authority:
    - `docs/authority/support_matrix.md` -> `## Phase 6 Nozzle-Specific Envelope`
    - `docs/authority/support_matrix.md` -> `## Global Policy` (phase-local support widening forbidden)
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (phase docs consume centralized support scope)
    - `docs/authority/reference_case_contract.md` -> `## Frozen Cases` (`R1`/`R0` nozzle family ownership)
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `### Design decisions` (`D7`)
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### 5. How profiles are represented`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `##### 5.1 \`constant\``
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `##### 5.2 \`radialPolynomial\``
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `##### 5.3 \`radialTable\``
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `##### 5.4 \`separableProfile\``
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### Step 4 — Implement constrained profile parser/compiler`
  - Backlog scope:
    - "Implement the constrained inlet-profile grammar/parser/compiler used by the frozen nozzle boundary contract."
  - Backlog done_when:
    - "Accepted inlet profiles compile into a machine-readable representation that the CPU reference and device path both understand."
- Depends on (backlog IDs):
  - `P6-01`
- Prerequisites:
  - `P6-01` support report and patch-role classifier for admissibility checks.
  - Frozen boundary-role envelope for `R1`/`R0` from support matrix.
- Concrete task slices:
  1. Implement parser and validator for admitted profile kinds only: `constant`, `radialPolynomial`, `radialTable`, `separableProfile`.
  2. Compile profiles into deterministic coefficient/table pools and index metadata consumable by CPU snapshot and device kernels.
  3. Enforce unsupported-keyword and non-monotonic-table rejection with clear diagnostics and stable error codes.
  4. Add canonical serialization/readback helpers so snapshot tests and runtime compilation use one representation.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - profile IR/compiler output schema
    - inlet-profile validation and error taxonomy
  - Consumed:
    - `P6-01` support-report/classifier outputs for admissible role/kind binding
- Validation:
  - Unit tests cover all admitted profile families and rejected grammar variants.
  - Compiled profile outputs are bit-stable under repeated parse/compile on same inputs.
  - CPU and device evaluators consume the same compiled profile representation.
- Done criteria:
  - Constrained profile grammar is implemented and compile outputs are shared by CPU and GPU boundary paths.
- Exports to later PRs/phases:
  - Compiled profile IR contract consumed by `P6-04` CPU snapshot path and `P6-07` swirl inlet kernel dispatch.
  - Validation fixtures reused by `P6-10` acceptance package.

## P6-04 Custom `gpuPressureSwirlInletVelocity` type and CPU snapshot path

- Objective:
  - Add custom nozzle inlet runtime type with CPU snapshot/reference execution so inlet semantics are frozen before GPU kernel integration.
- Exact citations:
  - Authority:
    - `docs/authority/support_matrix.md` -> `## Phase 6 Nozzle-Specific Envelope`
    - `docs/authority/support_matrix.json` -> `accepted_nozzle_case_family_boundary_tuple.swirl_inlet`
    - `docs/authority/reference_case_contract.md` -> `## Frozen Cases` (`R1` and `R0` nozzle-specific boundary behavior)
    - `docs/tasks/07_phase5_generic_vof_core.md` -> `## P5-01 Runtime schema, support scanner integration, and field-state contract`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `### Design decisions` (`D1`, `D6`)
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `##### 7.1 Required boundary dictionary for nozzle inlet \`U\``
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### 3. Detailed algorithm for \`gpuPressureSwirlInletVelocity\``
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### Step 5 — Add the custom patch type \`gpuPressureSwirlInletVelocity\``
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `##### 2.1 Swirl inlet patch snapshot test`
  - Backlog scope:
    - "Add the custom nozzle inlet runtime type plumbing together with the CPU snapshot/reference path used to validate inlet semantics before device execution."
  - Backlog done_when:
    - "The custom inlet BC can be instantiated, and CPU-side snapshot tests prove the frozen inlet math before GPU kernels land."
- Depends on (backlog IDs):
  - `P6-03`
  - `P5-01`
- Prerequisites:
  - `P6-03` compiled profile grammar and evaluator.
  - `P5-01` normalized runtime schema and support-scanner plumbing for BC-kind admission.
- Concrete task slices:
  1. Add and register `gpuPressureSwirlInletVelocity` patch type with canonical constructor/clone/write behavior required by runtime selection.
  2. Implement CPU reference evaluation path using the same compiled profile representation and flux-sign handling as the planned device kernel.
  3. Build CPU snapshot harness for frozen field-state inputs (`phi`, `rho`, owner `U`, time, geometry basis) and expected boundary outputs.
  4. Integrate startup validation so unsupported nozzle inlet configs fail before any timestep.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `gpuPressureSwirlInletVelocity` runtime type contract
    - CPU snapshot/reference harness for swirl inlet semantics
  - Consumed:
    - `P6-03` profile compiler IR
    - `P5-01` runtime schema/support-scanner integration
- Validation:
  - CPU-only startup accepts the custom BC and executes snapshot/reference path.
  - Snapshot tests verify flux-derived normal component and swirl superposition behavior on frozen fields.
  - Unsupported inlet grammar/kind combinations are rejected deterministically.
- Done criteria:
  - Custom inlet BC is instantiable and CPU snapshot tests freeze inlet semantics before GPU kernel implementation.
- Exports to later PRs/phases:
  - Frozen CPU inlet reference outputs consumed by `P6-07` invariance and parity tests.
  - Runtime type contract consumed unchanged by Phase 7 kernel replacement path.

## P6-05 Alpha boundary kernels

- Objective:
  - Implement device-side alpha boundary execution for the admitted nozzle tuple (`fixedValue`, `zeroGradient`, `inletOutlet`) with no host patch evaluation in the hot loop.
- Exact citations:
  - Authority:
    - `docs/authority/support_matrix.md` -> `## Phase 6 Nozzle-Specific Envelope`
    - `docs/authority/support_matrix.json` -> `phase6_nozzle_specific_envelope`
    - `docs/authority/support_matrix.md` -> `## Global Policy` (contact-angle out of milestone-1 scope unless authority changes)
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (contact-angle scope and support-matrix ownership)
    - `docs/tasks/boundary_matrix.md` -> `### Phase 5 -> Phase 6`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `##### 7.3 Alpha patch policies`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### 5. Detailed algorithm for alpha BCs`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `##### 5.1 \`fixedValue\``
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `##### 5.2 \`zeroGradient\``
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `##### 5.3 \`inletOutlet\``
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### Step 6 — Implement alpha boundary kernels`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `##### 2.4 Alpha boundary snapshot tests`
  - Backlog scope:
    - "Implement the accepted alpha BC kernels (`fixedValue`, `zeroGradient`, `inletOutlet`) for the nozzle envelope."
  - Backlog done_when:
    - "Frozen-field boundary tests pass for the accepted alpha patch kinds without host patch evaluation in the hot loop."
- Depends on (backlog IDs):
  - `P6-02`
- Prerequisites:
  - `P6-02` flat boundary spans and uploaded metadata views.
  - `P6-01` support report ensures only admitted alpha patch kinds are dispatched.
- Concrete task slices:
  1. Implement `applyAlphaBoundaryKernel` dispatch for admitted kinds and roles only, using shared `phi` sign convention for inflow/outflow handling.
  2. Bind alpha boundary writes directly to authoritative device boundary storage and enforce no host callback fallback path.
  3. Add mixed-sign snapshot tests for ambient/open `inletOutlet`, plus inlet `fixedValue` and wall `zeroGradient` cases.
  4. Add boundedness assertions for debug mode (`[-1e-12, 1 + 1e-12]` diagnostic envelope for component tests).
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - alpha boundary kernel dispatch contract
    - alpha boundary snapshot fixtures for nozzle tuple
  - Consumed:
    - `P6-02` boundary spans and manifest views
    - support-matrix alpha policy for nozzle roles
- Validation:
  - Frozen-field GPU results match CPU reference snapshots for admitted alpha kinds.
  - No host patch execution events are observed in hot-loop traces for alpha boundary stages.
  - `R1-core` remains excluded from Phase 6 nozzle alpha semantics.
- Done criteria:
  - Accepted nozzle alpha BC kernels run on device and pass frozen-field tests with no host patch fallback.
- Exports to later PRs/phases:
  - Device alpha BC stage contract consumed by `P6-10` solver-stage integration and Phase 7 optimization scope.

## P6-06 Ambient/open velocity boundary kernels

- Objective:
  - Implement device-side ambient/open velocity boundary execution for the admitted nozzle tuple using `pressureInletOutletVelocity` semantics only.
- Exact citations:
  - Authority:
    - `docs/authority/support_matrix.md` -> `## Phase 6 Nozzle-Specific Envelope`
    - `docs/authority/support_matrix.json` -> `accepted_nozzle_case_family_boundary_tuple.ambient_open`
    - `docs/authority/reference_case_contract.md` -> `## Phase-Gate Mapping` (Phase 6 uses `R1` and `R0`)
    - `docs/tasks/boundary_matrix.md` -> `### Phase 5 -> Phase 6`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `##### 7.2A Accepted ambient/open velocity configuration`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### 4. Detailed algorithm for ambient/open-boundary velocity BCs`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `##### 4.2 \`pressureInletOutletVelocity\``
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### Step 7 — Implement ambient/open velocity kernels`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `##### 2.2 Ambient/open boundary snapshot tests`
  - Backlog scope:
    - "Implement the standard ambient/open velocity kernels admitted by the support matrix for nozzle cases."
  - Backlog done_when:
    - "The accepted open-boundary velocity subset runs correctly on device and passes frozen snapshot tests."
- Depends on (backlog IDs):
  - `P6-02`
- Prerequisites:
  - `P6-02` boundary spans and metadata upload.
  - `P6-01` support report confirming admitted ambient/open velocity kind for the active case.
- Concrete task slices:
  1. Implement ambient/open velocity kernel path for `pressureInletOutletVelocity` with correct inflow/outflow branch behavior.
  2. Enforce role-gated dispatch so ambient/open kernels do not execute on nozzle swirl inlet roles.
  3. Add CPU/GPU frozen snapshot tests for accepted ambient/open tuple with volumetric and mass-flux guard checks.
  4. Add diagnostics that flag any accidental use of non-admitted ambient/open velocity variants in accepted runs.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - ambient/open velocity kernel contract for admitted nozzle tuple
    - snapshot fixture set for ambient/open velocity behavior
  - Consumed:
    - `P6-02` flat-span/manifest upload contract
    - support-matrix ambient/open tuple policy
- Validation:
  - GPU ambient/open boundary outputs match CPU reference snapshot outputs on frozen fields.
  - Admitted ambient/open subset executes device-side without host patch evaluation events.
- Done criteria:
  - Ambient/open velocity subset admitted by support matrix is implemented on device and passes frozen snapshot tests.
- Exports to later PRs/phases:
  - Ambient/open velocity boundary stage consumed by `P6-10` integration and exported unchanged to Phase 7 custom-kernel backend work.

## P6-07 Swirl inlet device kernel and invariance tests

- Objective:
  - Implement custom swirl inlet device kernel with basis regularization and integrated-flux invariance checks against frozen CPU inlet semantics.
- Exact citations:
  - Authority:
    - `docs/authority/support_matrix.md` -> `## Phase 6 Nozzle-Specific Envelope`
    - `docs/authority/support_matrix.json` -> `accepted_nozzle_case_family_boundary_tuple.swirl_inlet`
    - `docs/authority/acceptance_manifest.md` -> `## Exact Threshold Classes` (`TC_R1_NOZZLE` flux/pressure/QoI guidance consumed by later acceptance)
    - `docs/tasks/boundary_matrix.md` -> `### Phase 5 -> Phase 6`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `### Design decisions` (`D6`, `D10`)
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### 3. Detailed algorithm for \`gpuPressureSwirlInletVelocity\``
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### Step 8 — Implement the custom swirl inlet kernel`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `##### 2.1 Swirl inlet patch snapshot test`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `##### 1.2 Cylindrical basis construction`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `##### 1.3 Flux sign convention`
  - Backlog scope:
    - "Implement the custom swirl inlet device kernel, including basis regularization and integrated-flux invariance checks."
  - Backlog done_when:
    - "Synthetic annular-patch tests and inlet-flux invariance checks pass before full solver integration."
- Depends on (backlog IDs):
  - `P6-04`
  - `P6-02`
- Prerequisites:
  - `P6-04` custom BC type and frozen CPU snapshot outputs.
  - `P6-02` flat boundary spans and cylindrical basis metadata.
- Concrete task slices:
  1. Implement device swirl inlet kernel using flux-derived normal component plus orthogonalized tangential/radial superposition.
  2. Add axis regularization and right-handed basis sanity checks for annular/inlet patch geometry edge cases.
  3. Add integrated normal-flux invariance checks and compare against CPU snapshot/reference outputs.
  4. Add debug counters (`nBackflowFaces`, `nAxisRegularizedFaces`, `nNaNFaces`) with host copy only at safe diagnostics boundaries.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - swirl inlet device kernel contract
    - invariance and basis-validation test suite
  - Consumed:
    - `P6-04` CPU inlet semantics and compiled profile IR
    - `P6-02` boundary manifests and geometric basis metadata
- Validation:
  - Annular synthetic tests pass basis orthogonality/right-handedness and axis-regularization criteria.
  - Integrated normal flux matches prescribed `phi` within frozen tolerance in snapshot tests.
  - Device and CPU boundary outputs match on frozen inlet fixtures.
- Done criteria:
  - Swirl inlet device kernel and invariance tests pass before solver integration.
- Exports to later PRs/phases:
  - Swirl kernel contract and evidence consumed by `P6-10`.
  - Frozen semantics exported to Phase 7 as immutable correctness baseline for hotspot replacement.

## P6-08 Pressure boundary integration

- Objective:
  - Implement admitted `prgh*` pressure boundary kernels and coupled `fixedFluxPressure` `snGradp` update path integrated with pressure assembly on the named Phase 5 boundary-state contract.
- Exact citations:
  - Authority:
    - `docs/authority/support_matrix.md` -> `## Phase 6 Nozzle-Specific Envelope`
    - `docs/authority/support_matrix.json` -> `accepted_nozzle_case_family_boundary_tuple`
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (native pressure baseline and no hidden host pressure staging in accepted paths)
    - `docs/authority/graph_capture_support_matrix.md` -> `## Canonical Stage IDs` (`pressure_assembly`, `pressure_post`, `nozzle_bc_update`)
    - `docs/tasks/boundary_matrix.md` -> `### Phase 5 -> Phase 6`
    - `docs/tasks/07_phase5_generic_vof_core.md` -> `## P5-09 Generic momentum and pressure stage integration`
    - `docs/tasks/07_phase5_generic_vof_core.md` -> `## exports_to_next` (pressure boundary-state handoff)
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### 3. Phase 6 dependencies on previous phases`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### 4. Hidden coupling points that must be called out explicitly`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### 6. Detailed algorithm for \`prghPressure\``
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### 7. Detailed algorithm for \`prghTotalHydrostaticPressure\``
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### 8. Detailed algorithm for \`fixedFluxPressure\` gradient update`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### Step 9 — Implement \`prghPressure\` and \`prghTotalHydrostaticPressure\` kernels`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### Step 10 — Implement fixed-flux pressure gradient kernel and assembly hook`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### Step 13 — Wire stage calls into pressure corrector`
  - Backlog scope:
    - "Implement the accepted `prgh*` pressure kernels plus the `fixedFluxPressure` gradient update and the associated pressure-assembly hook as one coupled change."
  - Backlog done_when:
    - "`snGradp` matches the CPU reference on frozen tests and a short R1 transient passes the numerically risky pressure/BC coupling point."
- Depends on (backlog IDs):
  - `P6-02`
  - `P5-09`
- Prerequisites:
  - `P6-02` manifest and flat-span field views for pressure-boundary writes.
  - `P5-09` pressure stage integration points and canonical pressure stage boundaries.
- Concrete task slices:
  1. Implement device kernels for admitted pressure-value kinds (`prghPressure`, `prghTotalHydrostaticPressure`) using one stable sign convention.
  2. Implement `fixedFluxPressure` gradient kernel writing device-resident `snGradp` every pressure corrector iteration before pressure assembly.
  3. Bind `snGradp` and pressure boundary outputs into the exported Phase 5 pressure-boundary-state interface without introducing secondary state ownership.
  4. Add frozen snapshot tests and short transient checks focused on pressure/BC coupling stability.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - pressure-value boundary kernel contract for admitted nozzle tuple
    - fixed-flux `snGradp` update contract and test fixtures
  - Consumed:
    - `P5-09` pressure assembly and pressure stage sequencing contracts
    - Phase 5 pressure-boundary-state handoff (`PressureBoundaryStateView` semantics)
- Validation:
  - `snGradp` device outputs match CPU reference/constrained-pressure snapshots on frozen fixtures.
  - Short `R1` transient crosses pressure/BC coupling region without nozzle-path-attributed divergence.
  - No host patch execution occurs in pressure boundary stage in accepted configuration.
- Done criteria:
  - Coupled pressure boundary integration is implemented, `snGradp` matches reference on frozen tests, and short transient risk test passes.
- Exports to later PRs/phases:
  - Device-resident pressure boundary state (`PressureBoundaryState` + `snGradp`) consumed by `P6-10` and Phase 7 hotspot replacement scope.
  - Pressure boundary evidence consumed by Phase 8 tuple-level acceptance evaluation.

## P6-09 Startup seeding subsystem

- Objective:
  - Implement canonical startup seeding DSL/parser, device seed kernel, and post-seed refresh flow for accepted nozzle startup behavior without host `setFields` in accepted path.
- Exact citations:
  - Authority:
    - `docs/authority/support_matrix.md` -> `## Canonical Startup-Seed DSL`
    - `docs/authority/support_matrix.json` -> `startup_seed_dsl`
    - `docs/authority/support_matrix.json` -> `startup_seed_dsl.canonical_owner`
    - `docs/authority/support_matrix.json` -> `startup_seed_dsl.application_policy`
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (startup scope + normalized `gpuRuntime.*` ownership)
    - `docs/authority/acceptance_manifest.md` -> `## Hard Gates` (`host_setFields_startup_events == 0` in accepted startup path)
    - `docs/authority/reference_case_contract.md` -> `## Frozen Cases` (`R1`/`R0` startup conditioning context)
    - `docs/tasks/boundary_matrix.md` -> `### Phase 5 -> Phase 6`
    - `docs/tasks/07_phase5_generic_vof_core.md` -> `## P5-11 Write-time commit, validation artifacts, and Phase 5 baseline freeze`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `### Design decisions` (`D8`)
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### 6. Dictionary sources and ownership`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `##### 7.5 \`gpuStartupSeedDict\``
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### 9. Detailed algorithm for startup seeding`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### Step 14 — Implement startup seeding grammar and host parser`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### Step 15 — Implement device seed kernel`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### Step 16 — Add post-seeding refresh sequence`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### 3. Startup seeding regression`
  - Backlog scope:
    - "Implement the canonical startup-seed DSL/parser, the seed-region kernel, and the post-seed refresh sequence used by the accepted nozzle path."
  - Backlog done_when:
    - "Seed masks and initialized fields match the frozen reference exactly on the accepted startup cases."
- Depends on (backlog IDs):
  - `P6-01`
  - `P5-11`
- Prerequisites:
  - `P6-01` support report and manifest substrate for startup admissibility checks.
  - `P5-11` baseline freeze packet and restart/reload parity evidence, including history-field expectations.
- Concrete task slices:
  1. Implement canonical `gpuRuntime.startupSeed` parser with compatibility shim ingestion from legacy `gpuStartupSeedDict` and strict unknown-key rejection.
  2. Implement device seed kernel for admitted region types (`cylinderToCell`, `frustumToCell`, `boxToCell`, `sphereToCell`, `halfSpaceToCell`) and admitted field-value entries.
  3. Implement deterministic precedence (`lastWins`) and enforce application policy: seed once pre-first-step; restart reseeding forbidden unless `forceReseed yes`.
  4. Implement post-seed refresh pipeline (boundary refresh, mixture/derived updates, history-field propagation) before first transient step.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - startup seed parser/IR contract for canonical DSL
    - device seeding kernel and region-manifest contract
    - post-seed refresh/handoff evidence artifact
  - Consumed:
    - `P5-11` history-field/restart parity expectations
    - support-matrix canonical startup DSL and policy keys
- Validation:
  - Parse-only tests reject unsupported keys/field types/region types and accept frozen DSL grammar.
  - Seed-mask and seeded-field outputs match frozen CPU reference exactly for accepted startup cases.
  - Accepted startup path produces zero `host_setFields_startup_events`; restart reseed occurs only when `forceReseed yes`.
- Done criteria:
  - Canonical startup seeding subsystem is implemented and deterministic; seed masks/fields match frozen reference on accepted startup cases.
- Exports to later PRs/phases:
  - Startup seeding contract/evidence consumed by `P6-10` and Phase 8 acceptance tuples on `R1`/`R0`.
  - Restart reseed policy guard exported to Phase 7/8 so optimization/profiling does not alter startup semantics.

## P6-10 Solver-stage integration, graph-safety hardening, and Phase 6 acceptance

- Objective:
  - Integrate all Phase 6 nozzle BC/startup components into alpha/momentum/pressure stage sequencing, harden graph-safety constraints, and deliver formal Phase 6 acceptance package for `R1` with `R0` regression readiness.
- Exact citations:
  - Authority:
    - `docs/authority/graph_capture_support_matrix.md` -> `## Run Modes`
    - `docs/authority/graph_capture_support_matrix.md` -> `## Canonical Stage IDs`
    - `docs/authority/graph_capture_support_matrix.md` -> `## Global Capture Rules`
    - `docs/authority/graph_capture_support_matrix.json` -> `stages[stage_id=nozzle_bc_update]`
    - `docs/authority/acceptance_manifest.md` -> `## Coverage Rules`
    - `docs/authority/acceptance_manifest.md` -> `## Hard Gates`
    - `docs/authority/acceptance_manifest.md` -> `## Exact Threshold Classes`
    - `docs/authority/acceptance_manifest.json` -> `accepted_tuples[*].required_stage_ids`
    - `docs/authority/acceptance_manifest.json` -> `hard_gates`
    - `docs/authority/acceptance_manifest.json` -> `threshold_classes.field_qoi.TC_R1_NOZZLE`
    - `docs/authority/acceptance_manifest.json` -> `threshold_classes.field_qoi.TC_R0_NOZZLE`
    - `docs/authority/acceptance_manifest.json` -> `threshold_classes.parity_replay.RP_STRICT`
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (GPU-only operational contract; no silent fallback)
    - `docs/tasks/boundary_matrix.md` -> `### Phase 5 -> Phase 6`
    - `docs/tasks/boundary_matrix.md` -> `### Phase 6 -> Phase 7`
    - `docs/tasks/10_phase8_profiling_performance_acceptance.md` -> `## P8-03 Solver-stage instrumentation coverage`
    - `docs/tasks/10_phase8_profiling_performance_acceptance.md` -> `## P8-05 Nsight Systems capture scripts and artifact layout`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### 1. High-level orchestration`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### 2. Stage ordering relative to PIMPLE and alpha subcycling`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### 10. Graph-capture-safe control flow`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### Step 11 — Wire stage calls into alpha predictor`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### Step 12 — Wire stage calls into momentum predictor`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### Step 13 — Wire stage calls into pressure corrector`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### Step 17 — Add NVTX, counters, and debug controls`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### Step 18 — Add component tests`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### Step 19 — Add end-to-end reference-case regression`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### Step 20 — Profile and remove remaining fallback paths`
    - `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `### Acceptance checklist`
  - Backlog scope:
    - "Wire the nozzle boundary and startup subsystem into alpha/momentum/pressure stages, add runtime-parameter update handling, remove transient allocations, and run the Phase 6 acceptance pass."
  - Backlog done_when:
    - "Full R1 runs complete with device-side nozzle BC/startup behavior, graph-safety audits are clean, and the R0 regression package is ready for review."
- Depends on (backlog IDs):
  - `P6-05`
  - `P6-06`
  - `P6-07`
  - `P6-08`
  - `P6-09`
- Prerequisites:
  - All Phase 6 component kernels and startup subsystem from `P6-05..P6-09`.
  - Canonical stage scaffolding and run-mode policy from Phase 3 (`async_no_graph`, `graph_fixed`) and Foundation stage registry.
  - Pass-A stage-coverage and profiler artifact conventions from Phase 8 (`P8-03`, `P8-05`) for acceptance evidence formatting.
- Concrete task slices:
  1. Wire alpha/momentum/pressure boundary stage calls into solver loop ordering exactly once per required loop scope (alpha subcycle, pre-momentum, pressure-corrector iteration).
  2. Add runtime parameter refresh path that updates small device parameter buffers without stage-time dictionary access or allocation.
  3. Harden graph safety: no post-warmup allocation, no hidden host reads, no CPU patch execution in capture-safe stages; explicit downgrade to `async_no_graph` on graph failure.
  4. Run Phase 6 acceptance pipeline: component suite, `R1` reduced nozzle pass, `R0` regression package, and required profiling/trace artifacts.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `phase6_acceptance_bundle` (support report, component tests, `R1`/`R0` regression summary, graph-safety audit, profiler traces)
    - stage-sequencing and runtime-param update evidence report keyed to canonical stage IDs
  - Consumed:
    - `P6-05..P6-09` boundary/startup kernel contracts
    - canonical stage IDs and accepted run-mode policy from graph/acceptance authorities
    - Phase 8 pass-A instrumentation/profiler artifacts for consistent evidence formatting
- Validation:
  - `R1` device-side nozzle path runs end-to-end with accepted stage sequencing and no nozzle-path-attributed divergence in the defined observation window.
  - Hard-gate metrics remain compliant (`cpu_boundary_fallback_events`, `host_patch_execution_events`, `post_warmup_alloc_calls`, `mandatory_nvtx_ranges_present`, etc.).
  - `R0` regression artifact package is complete and mapped to acceptance tuple vocabulary without introducing Phase 7 custom-kernel claims.
- Done criteria:
  - Phase 6 integration and graph-safety hardening are complete, `R1` acceptance run succeeds on device-side nozzle path, and `R0` regression package is review-ready.
- Exports to later PRs/phases:
  - Frozen Phase 6 semantic baseline consumed by Phase 7 as non-widening contracts for hotspot replacement backends.
  - Acceptance evidence packet consumed by Phase 8 tuple-level profiling and disposition workflows.
  - Explicit `nozzle_bc_update` stage ownership evidence for cross-phase seam auditing.

## imports_from_prev

- `FND-04` centralized support scanner/fail-fast taxonomy and unsupported-tuple diagnostics.
- `P2-08` mesh mirror/startup registration and boundary-index stability substrate.
- Phase 3 execution contracts:
  - canonical run modes (`sync_debug`, `async_no_graph`, `graph_fixed`)
  - canonical stage IDs and fallback semantics
  - graph-external write/restart boundaries
- Phase 4 pressure bridge contracts consumed through Phase 5:
  - `PressureMatrixCache`
  - canonical pressure stage boundaries (`pressure_assembly`, `pressure_solve_native`, `pressure_post`)
  - native baseline pressure policy
- Phase 5 generic-core seam packet:
  - explicit `R1-core` generic-only guard
  - `PressureBoundaryStateView` handoff for nozzle boundary pressure integration
  - write/restart parity expectations and baseline freeze evidence (`P5-11`)
- Phase 8 pass-A instrumentation/profiling contracts used as evidence format inputs:
  - `P8-03` stage instrumentation coverage
  - `P8-05` standardized Nsight Systems artifact layout

## exports_to_next

- Boundary support-report and patch-role classification artifacts for frozen nozzle tuple.
- `DeviceBoundaryManifestView` and flat boundary-span contracts for all admitted Phase 6 patch roles.
- Constrained profile parser/compiler outputs and CPU snapshot fixtures for `gpuPressureSwirlInletVelocity`.
- Device kernel contracts for:
  - alpha boundary kinds (`fixedValue`, `zeroGradient`, `inletOutlet`)
  - ambient/open velocity (`pressureInletOutletVelocity`)
  - swirl inlet velocity (`gpuPressureSwirlInletVelocity`)
  - pressure-value + `fixedFluxPressure` gradient (`snGradp`) updates
- Canonical startup seeding subsystem (`gpuRuntime.startupSeed`) with deterministic restart reseed policy.
- Phase 6 integration and acceptance bundle (`R1` pass + `R0` regression package) mapped to canonical stage IDs and manifest vocabulary.
- Phase 6 to Phase 7 seam packet:
  - semantic ownership freeze for nozzle BC/startup behavior
  - explicit non-widening rule for hotspot replacement kernels
  - unchanged support-matrix tuple boundaries for `R1`/`R0`

## shared_terms

- `BoundarySupportReport`: deterministic startup classification artifact proving admitted vs rejected nozzle patch tuples.
- `DeviceBoundaryManifestView`: immutable device-resident boundary metadata and flat-span mapping used by all Phase 6 kernels.
- `PressureBoundaryState`: single boundary-state contract consumed by pressure assembly; Phase 6 updates but does not fork ownership.
- `snGradp`: device-resident fixed-flux pressure gradient state refreshed each pressure-corrector iteration before pressure assembly.
- `nozzle_bc_update`: canonical Phase 6 stage ID for nozzle boundary work; fallback mode is `async_no_graph`.
- `startupSeed`: canonical runtime subtree `gpuRuntime.startupSeed`; legacy `gpuStartupSeedDict` is compatibility input only.
- `R1-core`: reduced generic case that remains outside nozzle-specific Phase 6 semantics.
- `baseline_gpu`: Phase 5/6 semantic path without Phase 7 custom hotspot kernels.

## open_discontinuities

- `[tracked] phase5_phase6_pressure_boundary_state_shape`: Phase 5 exports `PressureBoundaryStateView`, but exact member-level freeze remains distributed across phase prose and seam docs. Impacted PR IDs: `P6-08`, `P6-10`. Citations: `docs/tasks/07_phase5_generic_vof_core.md` -> `## open_discontinuities`, `docs/tasks/boundary_matrix.md` -> `### Phase 5 -> Phase 6`, `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### 4. Hidden coupling points that must be called out explicitly`. Preferred reading: consume Phase 5 boundary-state contract unchanged; do not create a second `snGradp` ownership path.
- `[tracked] startup_seed_compatibility_shim_precedence`: Canonical startup-seed ownership is `gpuRuntime.startupSeed` while legacy `gpuStartupSeedDict` remains compatibility input; parser precedence and conflict resolution must stay deterministic. Impacted PR IDs: `P6-09`, `P6-10`. Citations: `docs/authority/support_matrix.json` -> `startup_seed_dsl.canonical_owner`, `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `##### 7.5 \`gpuStartupSeedDict\``, `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (normalized `gpuRuntime.*` ownership). Preferred reading: canonical tree wins; shim maps into canonical schema and cannot add new semantics.
- `[tracked] phase6_phase7_hotspot_scope_lock`: Phase 7 optimization work may replace kernels but cannot change admitted BC kinds, contact-angle scope, scheme tuple, or graph semantics. Impacted PR IDs: `P6-10`, `P7-01`, `P7-02`, `P7-03`, `P7-04`. Citations: `docs/tasks/boundary_matrix.md` -> `### Phase 6 -> Phase 7`, `docs/authority/support_matrix.md` -> `## Phase 6 Nozzle-Specific Envelope`, `docs/authority/support_matrix.md` -> `## Exact Audited Scheme Tuple`, `docs/authority/acceptance_manifest.md` -> `## Coverage Rules`. Preferred reading: treat Phase 6 semantics as frozen contracts and Phase 7 as replaceable backend implementation only.
- `[tracked] contact_angle_conditionality_guard`: Support matrix freezes contact-angle out of milestone-1 scope unless explicitly required; Phase 6 wall alpha handling is therefore `zeroGradient` by default and must not be widened ad hoc. Impacted PR IDs: `P6-05`, `P6-10`. Citations: `docs/authority/support_matrix.md` -> `## Global Policy`, `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions`, `docs/specs/phase6_pressure_swirl_nozzle_bc_spec.md` -> `#### Supported boundary-condition matrix` (wall alpha conditional note). Preferred reading: keep contact-angle path out unless authority docs are revised.

## validation_checks

- All Phase 6 cards preserve canonical dependency edges exactly:
  - `P6-01 <- FND-04,P5-11`
  - `P6-02 <- P6-01,P2-08`
  - `P6-03 <- P6-01`
  - `P6-04 <- P6-03,P5-01`
  - `P6-05 <- P6-02`
  - `P6-06 <- P6-02`
  - `P6-07 <- P6-04,P6-02`
  - `P6-08 <- P6-02,P5-09`
  - `P6-09 <- P6-01,P5-11`
  - `P6-10 <- P6-05,P6-06,P6-07,P6-08,P6-09`
- Every Phase 6 card anchors to authority docs/json + exact Phase 6 subsection citations + backlog `scope` + backlog `done_when`.
- Phase 6 scope remains nozzle-specific only:
  - no redefinition of generic VOF internals from Phase 5
  - no kernel specialization scope from Phase 7
  - no widened support-matrix BC/scheme/contact-angle policy
- Seam rules are preserved:
  - `R1-core` remains generic-only
  - `PressureBoundaryState` plus device-resident `snGradp` stays the single pressure-boundary-state contract
  - restart reseeding remains forbidden unless `forceReseed yes`
- Graph and acceptance language stays authority-aligned:
  - canonical stage IDs are used without aliases
  - capture failures downgrade explicitly to `async_no_graph`
  - hard-gate and threshold vocabulary is consumed from `acceptance_manifest(.md/.json)` without local policy widening
