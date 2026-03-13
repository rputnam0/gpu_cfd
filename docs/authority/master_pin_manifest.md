# Master Pin Manifest

This file is the authoritative package-level source for source/version/toolchain/profiler pins. Later phase documents consume it; they do not restate looser minima or select alternative dependency tuples locally.

## Frozen Defaults

| Area | Frozen value | Notes |
|---|---|---|
| Reviewed source tuple ID | `SRC_SPUMA_V2412_AD2A_FES_MAIN_AMGX_2_5_0` | Exact archive-approved external-source tuple; its concrete repo refs are resolved below and changing any component requires a manifest revision before coding starts. |
| Runtime base | `exaFOAM/SPUMA 0.1-v2412` mapped to OpenCFD `v2412` | Do not implement Phases 3-8 against OpenFOAM 12. |
| SPUMA support-policy snapshot | `GPU-support` wiki `ad2a385e44f2c01b7d1df44c5bc51d7996c95554` (2025-06-10) | Exact reviewed public support snapshot consumed by this archive. |
| External solver bridge | `exaFOAM/foamExternalSolvers` Phase-4 bridge line paired to `SRC_SPUMA_V2412_AD2A_FES_MAIN_AMGX_2_5_0` | Do not introduce a parallel external-solver stack in Phases 4-8. |
| AmgX reference line | `v2.5.0` | Exact external-solver baseline admitted by this archive. |
| Primary toolkit lane | CUDA 12.9.1 | Required build and validation lane. |
| Experimental toolkit lane | CUDA 13.2 | Verification-only lane until explicitly promoted. |
| Driver floor | `>=595.45.04` | Applies to the frozen production lane. |
| GPU target | `NVARCH=120`, native `sm_120` plus PTX | PTX/JIT validation remains mandatory. |
| Instrumentation | NVTX3 | NVTX2-era integration is forbidden. |
| Nsight Systems | `2025.2` | Exact timeline/profiling baseline for acceptance and Blackwell graph tracing references. |
| Nsight Compute | `2025.3` | Exact kernel deep-dive baseline with Blackwell node-level graph support improvements. |
| Compute Sanitizer | `2025.1` | Exact sanitizer baseline with documented additional Blackwell support. |
| Workstation target | Single RTX 5080, compute capability 12.0, 16 GB GDDR7, no NVLink assumption | The plan is single-GPU by default. |

## Resolved Frozen Source Tuple

The reviewed tuple ID above resolves to the exact upstream refs below. Later phases may mirror these trees locally, but they may not float to a newer branch head or substitute a different tag while still claiming compliance with `SRC_SPUMA_V2412_AD2A_FES_MAIN_AMGX_2_5_0`.

| Component | Upstream object | Frozen ref kind | Frozen ref / version | Exact resolved commit / snapshot |
|---|---|---|---|---|
| SPUMA runtime base | `exaFOAM/SPUMA` | tag | `0.1-v2412` | `3d1d7bf598ec8a66e099d8688b8597422c361960` |
| SPUMA support-policy snapshot | `exaFOAM/SPUMA` wiki `GPU-support` | wiki version | `2025-06-10` | `ad2a385e44f2c01b7d1df44c5bc51d7996c95554` |
| External solver bridge | `exaFOAM/foamExternalSolvers` | branch freeze | `main` | `4c764d027f8f124a1cc0b6df0520eb63593c2a2b` |
| AmgX backend | `NVIDIA/AMGX` | tag | `v2.5.0` | `cc1cebdbb32b14d33762d4ddabcb2e23c1669f47` |

## Consumption Rules

- Historical references to CUDA 12.8 or 570.xx drivers are compatibility background only; they are not the active project pin set.
- Native pressure is the frozen default backend. AmgX is a supported secondary lane and may count as production-eligible only for tuples explicitly admitted by [acceptance_manifest.md](acceptance_manifest.md).
- The reviewed source tuple ID above is a frozen archive identifier, not shorthand for “current `develop`” or “current `main`”. Any different branch tip, local fork head, or repinned mirror is a different dependency tuple and requires a manifest revision first.
- The exact local mirror SHAs used by an implementation branch must realize the resolved refs above; record those mirror SHAs plus the exact toolkit/profiler binaries actually invoked in `host_env.json` and `manifest_refs.json`.
- `host_env.json` and `manifest_refs.json` are execution-trace artifacts for the already-frozen tuple above, not a mechanism for selecting a different SPUMA / `foamExternalSolvers` / AmgX / profiler combination per run.

## Required Revalidation If This Manifest Changes

1. Phase 1 smoke/build lane on the primary toolkit.
2. Phase 3 `async_no_graph` and `graph_fixed` smoke coverage.
3. Phase 5 `R1-core` native baseline.
4. Phase 8 baseline timeline acceptance on `R1` and `R0`.
