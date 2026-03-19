# Phase 1 Blackwell Bring-Up

This note is the human-readable companion to the Phase 1 workstation artifact bundle.
It captures how to run the final `P1-07` checks and what must be present before the
Phase 1 acceptance report can be treated as reviewable.

## Required Inputs

- Canonical discovery artifacts from `P1-01`: `host_env.json`, `manifest_refs.json`, `cuda_probe.json`
- Discovery must have passed the runtime-health checks recorded in `cuda_probe.json`:
  `native_kernel_ok=true` and `managed_memory_probe_ok=true`
- `cuda_probe.json` must also carry the same reviewed tuple, runtime base, and primary-lane
  toolkit selection as `host_env.json` and `manifest_refs.json`
- Primary-lane build metadata and fatbinary evidence from `P1-02` and `P1-03`
- `build_metadata.json` and `fatbinary_report.json` must stay traceable to the same reviewed
  tuple, runtime base, and primary-lane toolkit selection as the canonical manifests
- Smoke-case results for `cubeLinear`, `channelSteady`, and `channelTransient` from `P1-04`
- Smoke, memcheck, and Nsight result JSONs must stay traceable to the same reviewed tuple,
  runtime base, and primary-lane toolkit selection as the canonical manifests
- Compute Sanitizer memcheck result from `P1-05`
- Nsight Systems `basic` and `um_fault` result JSONs from `P1-06`

## WSL Driver Guard

On WSL, the checked-in discovery path now rejects mixed driver stacks before the
CUDA probe runs. If `tools/bringup/env/run_cuda_probe.sh` prints a message like:

```text
WSL host should not expose Linux display driver libraries at /usr/lib/x86_64-linux-gnu/libcuda.so.1.
Remove the Linux display driver packages from WSL and rely on /usr/lib/wsl/lib.
Conflicting Linux-side driver libraries: /usr/lib/x86_64-linux-gnu/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so.535.288.01 /usr/lib/x86_64-linux-gnu/libnvidia-ml.so /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.288.01 /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.535.288.01
Installed Linux-side driver owner packages: libnvidia-compute-535
Example cleanup command: sudo apt remove --purge libnvidia-compute-535 libnvidia-compute-535-server
Simulated apt fallout: libcuinj64-12.0, libnvidia-ml-dev, nsight-systems, nsight-systems-target, nvidia-cuda-dev, nvidia-cuda-toolkit, nvidia-profiler, nvidia-visual-profiler
Manual toolkit package anchor: nvidia-cuda-toolkit -> nvidia-cuda-dev -> libnvidia-compute-535
If you need to restore the CUDA toolkit in WSL afterward, use NVIDIA's WSL-Ubuntu installer path or the cuda-toolkit-12-x meta-package only; do not install cuda, cuda-12-x, or cuda-drivers under WSL.
Installed related CUDA toolkit packages: libcudart12:amd64, nvidia-cuda-dev:amd64, nvidia-cuda-toolkit
```

then the distro is exposing native Linux NVIDIA driver libraries alongside the WSL
driver shim, including the PTX JIT compiler and NVML libraries that the `P1-07`
acceptance lane depends on. This is a host-configuration problem, not a Phase 1
artifact-parser problem.

Expected remediation:

- remove the Linux display-driver packages from the WSL distro
- if the host looks like this workstation, check for conflicting packages such as
  `libnvidia-compute-525`, `libnvidia-compute-525-server`,
  `libnvidia-compute-535`, `libnvidia-compute-535-server`, `libcudart12`,
  `nvidia-cuda-dev`, and `nvidia-cuda-toolkit`
- on Ubuntu 24.04, purging `libnvidia-compute-535` alone may cause `apt` to install
  `libnvidia-compute-535-server` as a replacement, so include both in the purge command
- on this workstation, simulating `apt-get -s remove --purge libnvidia-compute-535
  libnvidia-compute-535-server` confirms that naming both packages explicitly avoids
  that replacement loop; the remaining direct blast radius is the expected toolkit
  and profiler package set already shown in the guard output
- on this workstation, `apt-mark showmanual` currently lists `nvidia-cuda-toolkit`,
  and `apt-cache depends` shows the relevant chain as
  `nvidia-cuda-toolkit -> nvidia-cuda-dev -> libnvidia-compute-535`, so the manual
  distro toolkit install is the current package anchor keeping the conflicting native
  Linux driver library owner in the WSL distro
- expect `apt` to propose removing dependent toolkit packages when the native
  `libcuda` owner package is purged; the guard now prints a `Simulated apt fallout`
  line so the operator can review that blast radius before making host changes
- treat `apt autoremove` as a separate follow-up choice, not as part of the minimum
  cleanup step needed to remove the conflicting native Linux driver package
- if the toolkit must be restored after cleanup, follow NVIDIA's WSL-specific
  toolkit path: use the WSL-Ubuntu installer or the `cuda-toolkit-12-x` meta-package
  only, and do not install `cuda`, `cuda-12-x`, or `cuda-drivers` inside WSL
- on this workstation, `command -v nvcc` currently resolves to
  `/home/rputn/.local/nvidia/cuda-12.9.1/bin/nvcc` and `nvcc --version` reports
  `12.9.86`, so the active primary-lane compiler is already outside the distro
  `nvidia-cuda-toolkit` package set that `apt` proposes removing
- on this workstation, `apt-cache search cuda-toolkit` currently shows only the
  distro `nvidia-cuda-toolkit` packages and does not expose a `cuda-toolkit-12-x`
  meta-package; if cleanup really does remove the active toolchain path, follow
  NVIDIA's WSL-Ubuntu installer guidance rather than assuming the Ubuntu archive
  can restore the reviewed CUDA 12.9.1 lane directly
- if cleanup removes the native Linux bundle but the probe still fails, try one
  last diagnostic pass with the WSL-side PTX JIT and NVML libraries forced ahead
  of distro copies; on this workstation that experiment still ended at
  `cudaFree(0): OS call failed or operation not supported on this OS`, which
  means the remaining fault is deeper in the host CUDA/WSL runtime path
- if the failure remains after cleanup, move to the Phase 1 spec's host-level
  HMM/KASLR path: this workstation's `/proc/cmdline` does not currently include
  `nokaslr`, the WSL-visible worker view does not expose `nvidia_uvm` tunables,
  and `dmesg` shows repeated early `dxgkio_query_adapter_info` ioctl failures
- keep using the Windows-side NVIDIA WSL driver and the `/usr/lib/wsl/lib` shim
- rerun the canonical host discovery entrypoint first:
  `tools/bringup/env/check_host_env.sh <discovery-dir>`
- if that wrapper fails, inspect the emitted `nvidia_runtime_snapshot.txt`; it now
  includes `command -v nvcc`, the full `nvcc --version` block, a compact
  environment section with `CUDA_HOME`, `LD_LIBRARY_PATH`, and `PATH`,
  `ldconfig -p` CUDA-driver resolution, native-vs-WSL driver library paths,
  canonical native-driver `realpath` output, `dpkg -S` owner lookups, installed
  NVIDIA/CUDA package rows, relevant `apt-cache policy` output, relevant
  `apt-mark showmanual` rows, the current `apt-cache depends` toolkit-anchor
  chain, `nvidia-smi` state, `/proc/cmdline`, and the recent `dxg` kernel
  messages in one place
- the same discovery directory now also keeps `check_host_env.log`, which preserves
  the exact `run_cuda_probe.sh` / discovery wrapper terminal output from the failed
  run so the cleanup hint and any later discovery-side failure text survive handoff;
  the log now also records the wrapper UTC time, repo commit, lane, output directory,
  and invocation line for quick provenance checks
- on Ubuntu/WSL usrmerged systems, `ldconfig -p` may show the conflicting native
  libraries under `/lib/x86_64-linux-gnu` even though the canonical paths resolve
  back into `/usr/lib/x86_64-linux-gnu`; use the snapshot's `realpath` section to
  confirm that they are the same native Linux bundle rather than a second bundle
- use the snapshot's `apt-cache policy` and `apt-mark showmanual` sections to see
  which native driver/toolkit packages are installed, which replacement candidate
  `apt` prefers, and whether toolkit packages were installed manually before making
  host cleanup or restore decisions

NVIDIA's CUDA on WSL guide says the Windows display driver is the only driver
needed, warns that the default CUDA installation can overwrite the WSL driver mapping,
recommends the WSL-Ubuntu toolkit path, and says not to install a Linux display
driver in WSL:
https://docs.nvidia.com/cuda/archive/13.0.0/wsl-user-guide/index.html

Do not continue to build, smoke, memcheck, Nsight, PTX-JIT, or final acceptance
until the CUDA probe returns real RTX 5080 metadata and `managed_memory_probe_ok=true`.

## PTX-JIT Proof

Run the Phase 1 compatibility proof with the checked-in wrapper:

```bash
tools/bringup/run/check_ptx_jit.sh \
  --artifact-root <artifact-root> \
  --scratch-root <scratch-root> \
  --fatbinary-report <fatbinary-report.json>
```

The wrapper forces `CUDA_FORCE_PTX_JIT=1`, reuses the audited `cubeLinear` smoke case,
and writes `ptx_jit/` artifacts under the chosen artifact root.
The emitted `ptx_jit_result.json` must stay traceable to the same reviewed tuple,
runtime base, and primary toolkit lane as the canonical discovery manifests.

## Acceptance Bundle

Generate the final machine-readable and human-readable reports with:

```bash
tools/bringup/run/run_phase1_acceptance.sh \
  --output-dir <acceptance-dir> \
  --host-env-json <host_env.json> \
  --manifest-refs-json <manifest_refs.json> \
  --cuda-probe-json <cuda_probe.json> \
  --build-metadata-json <build_metadata.json> \
  --fatbinary-report-json <fatbinary_report.json> \
  --smoke-result <cubeLinear smoke_result.json> \
  --smoke-result <channelSteady smoke_result.json> \
  --smoke-result <channelTransient smoke_result.json> \
  --memcheck-result-json <memcheck_result.json> \
  --nsys-result <basic nsys_profile_result.json> \
  --nsys-result <um_fault nsys_profile_result.json> \
  --ptx-jit-result-json <ptx_jit_result.json> \
  --bringup-doc docs/bringup/phase1_blackwell.md
```

For compatibility with the Phase 1 spec's original layout, the repo also keeps
`tools/bringup/python/acceptance_gate.py` as a thin shim to the same checked-in
acceptance implementation and CLI.

Expected outputs:

- `phase1_acceptance_report.json`
- `phase1_acceptance_report.md`
- `phase1_acceptance_bundle_index.json`

The JSON report is the deterministic gate source. The Markdown report is the review-facing
summary that points back to each artifact path, including the build log, PTX-JIT command
logs, raw fatbinary `ptx` / `sass` dumps, Compute Sanitizer memcheck log, raw Nsight trace
artifacts, and the smoke-result and smoke-log / Nsight-result JSON inputs used by the hard
gates. The bundle index is the archive-facing directory of the final reports plus the
supporting artifact inputs used to build them.
It explicitly records the reviewed tuple, selected lane, workstation summary, and the
canonical workstation-manifest paths (`host_env.json`, `manifest_refs.json`,
`cuda_probe.json`) plus the supporting smoke/build/fatbinary/PTX-JIT/memcheck/Nsight
trace artifacts, so downstream consumers can trace the acceptance packet back to the same
frozen Phase 1 host evidence without reparsing the full report body.

## PASS Criteria

Phase 1 acceptance is `PASS` only when all hard checks in the `P1-07` card succeed:

- frozen workstation/toolchain identity is present in the canonical manifests
- `cuda_probe.json` confirms native-kernel and managed-memory readiness
- `cuda_probe.json` is traceable to the same reviewed tuple, runtime base, and primary lane
- build metadata records `have_cuda=true` and `NVARCH=120`
- `build_metadata.json` is traceable to the same reviewed tuple, runtime base, and primary lane
- fatbinary inspection proves native `sm_120` and retained PTX
- `fatbinary_report.json` is traceable to the same reviewed tuple, runtime base, and primary lane
- the PTX-JIT run succeeds under `CUDA_FORCE_PTX_JIT=1`
- `ptx_jit_result.json` is traceable to the same reviewed tuple, runtime base, and primary lane
- all three Phase 1 smoke cases passed
- smoke, memcheck, and Nsight result JSONs are traceable to the same reviewed tuple, runtime base,
  and primary lane
- Nsight Systems shows required NVTX ranges and GPU kernels
- the UVM diagnostic trace is present and its activity is either clean or documented
- memcheck reports no actionable errors

If any hard check fails, the acceptance bundle must remain `FAIL`.
