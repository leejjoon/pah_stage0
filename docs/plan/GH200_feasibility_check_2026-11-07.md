# GH200 feasibility check, 2026-11-07

Empirical check of `T0_T0b_execution_plan_GH200.md` §3 (environment), §3.4 (smoke tests) and the GH200-specific claims in §4.3, run on the project's own GH200 node. It is a check of the *software stack*, run in plain `uv` venvs that mirror containers A, B and C. Apptainer was then installed, and all four images were built and tested the same afternoon; see **Container builds** below. Recipes: `containers/*.def`. In-image package lists: `containers/requirements-*-container.txt`. Hashes: `containers/images-sha256-2026-11-07.txt`.

## Node

| Property | Value | Plan assumption |
| --- | --- | --- |
| Host | `gh200`, Ubuntu 24.04.4, kernel `7.0.0-1019-nvidia-64k` | — |
| Arch | aarch64 | aarch64 ✔ |
| Superchips | **2 × GH200 144G HBM3e** (GPU0 ↔ CPUs 0–71, NUMA 0; GPU1 ↔ CPUs 72–143, NUMA 1) | one GH200, `nproc` = 72 |
| CPU cores | 144 (72 per Grace) | 72 |
| Host memory | 1.24 TB total, ~480 GB per Grace (NUMA 0 = 490 GB) | up to 480 GB ✔ |
| NUMA nodes | 18 (GPU memory exposed as NUMA 2 and 10; Addressing Mode ATS, C2C enabled) | — |
| Page size | **65,536 bytes** (64 KiB kernel) | not mentioned |
| Driver / CUDA | **595.45.04 / CUDA 13.2** | — |
| Host CUDA toolkits | `/usr/local/cuda-13.1`, `-13.2`, `-13.3` | — |
| Container runtimes | **none** (no apptainer, singularity, docker, podman, enroot); `mksquashfs` present | Apptainer available |
| User namespaces | `kernel.apparmor_restrict_unprivileged_userns = 1` | — |
| MPS binaries | `nvidia-cuda-mps-control`, `-server` present | ✔ |
| Storage | `/home` is NFS (56 TB); `/` local NVMe, 1.6 TB free | "scratch ≥ 200 GB" |
| Network | nvcr.io, pypi.org, github.com, julialang-s3 reachable | — |
| `uv` | 0.12.12, CPython 3.12.14 aarch64 | Python 3.12 ✔ |

The synced repository needed no machine fixes. `.venv` points to an aarch64 uv CPython 3.12.14, `papertrail` runs, and the absolute paths in `.mcp.json` match this machine.

## Results

| # | Item | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Container runtime | **Closed later the same day:** Apptainer 1.5.5 (setuid) installed; `--fakeroot` builds work with the user's subuid range. Before that: none installed. AppArmor restricts unprivileged user namespaces, so a non-setuid Apptainer needs an AppArmor profile; `apptainer-suid` avoids that. | `which` finds nothing; sysctl above |
| 2 | JAX on the H100 (container A's stack) | **PASS.** jax 0.11.2 with `jax-cuda12-plugin` 0.11.2 → `[CudaDevice(id=0)]`; `jax_enable_x64` False | §A below |
| 3 | Container A recipe order | **FAIL as written.** `mace-jax` depends on `e3nn==0.4.4` (PyTorch e3nn), which pulls PyPI's default aarch64 torch, `2.15.0+cu132`, with the CUDA 13 libraries. The plan installs mace-jax *first* and CPU torch *second*, so the second step is a no-op: "already satisfied". Fixed by installing the CPU torch *before* mace-jax. That gives `torch 2.15.0+cpu` and `uv pip check` clean. | `logA.txt`; rebuilt venv A |
| 4 | cuEquivariance-JAX import | **PASS with caveat.** `cuequivariance_ops` is imported before JAX has loaded its pip CUDA 12 libraries, so `libcue_ops.so` fails to load (`libnvrtc.so.12` not found), and `mace-jax-from-torch` crashes at import. It works once `jax.devices()` has run first, or with `LD_LIBRARY_PATH` pointing at `site-packages/nvidia/{cuda_nvrtc,cublas}/lib`. The plan's `cuda:*-cudnn-devel` base ships these libraries system-wide, which masks the problem; a `-runtime`/`-base` image or a bare venv does not. | §A |
| 5 | Smoke test 1 (jax-md on GPU) | **PASS.** `nvt_langevin` 1,000 steps compiled and ran on `CudaDevice(id=0)` (run-only check: random LJ start, energy not meaningful) | — |
| 6 | Smoke test 2 (MACE-OFF small, PyTorch, float32, GPU) | **PASS.** PyPI `torch 2.15.0+cu132` + `mace-torch 0.3.17` + cuEq 0.10.0 (cu13). Benzene E = −6324.1147 eV. With `enable_cueq=True` the model holds 88 `cuequivariance_torch` modules, so the backend really is active | §B |
| 7 | Smoke test 3 (convert + mace-jax) | **PASS via `--foundation off --model-name small`.** **`--torch-model <file>` FAILS** under PyTorch ≥ 2.6: `torch.load` defaults to `weights_only=True` and rejects the pickled `ScaleShiftMACE`. The foundation route converted the same cached file the PyTorch side loaded (sha256 `165cce4c…c46f`) | §A |
| 8 | **Force gate**, torch vs jax, float32, no TF32 | **PASS by ~500×.** max \|ΔF\| on benzene: jax(no cuEq) vs torch(no cuEq) 2.1e-6, vs torch(cuEq) 2.0e-6; jax(cuEq) vs torch(cuEq) 1.3e-6 eV/Å. ΔE = 0 at float32 resolution. Tolerance is 1e-3 | §C |
| 9 | mace-jax cuEquivariance switch | **Found.** Set `"equivariance_config": {"backend": "cueq"}` in the bundle JSON (`cueq_config` is the deprecated alias). Forces change at the 1e-7 level, showing a different code path | §C |
| 10 | mace-jax dtype | The *converter* has no dtype flag (plan is right). The *loader* has one: `load_model_bundle(path, "float32")` / `mace-jax-predict --dtype float32`. It only toggles `jax_enable_x64` and leaves some float64 leaves in the parameter tree, so the plan's explicit cast is still needed | §A |
| 11 | TorchSim on aarch64 + H100 | **PASS (install + batched forces).** `torch-sim-atomistic` 0.6.2, `nvalchemi-toolkit-ops` 0.5.0, `warp-lang` 1.19.0 install without replacing torch. `mace.calculators.mace_torchsim.MaceTorchSimModel(dtype=float32, compute_stress=False)` on a 3-molecule batch matches per-molecule ASE to 6.5e-7 eV/Å. *Not yet run:* the pilot's 100-molecule 1 ps Langevin run and the edge check | §D |
| 12 | PySR / Julia on the 64 KiB-page kernel | **PASS.** PySR 2.7.2, juliacall 0.9.36, Julia 1.13.3 (aarch64), downloaded and precompiled at first import (124 s on this node). The 100-point quick-start recovers `x0² + 2.5 cos(x1) − 0.5` with `deterministic=True, parallelism="serial"` and with `parallelism="multiprocessing", procs=4`. The Julia binary is installed *inside* `PYTHON_JULIAPKG_PROJECT` (`pyjuliapkg/install/`), so the writable copy of `/opt/julia_env` carries it. Project + depot = 1.5 GB | §E |
| 13 | Smoke test 4 prerequisites (PAHFIT) | **PASS.** The pinned commit `ca599f9` installs as `pahfit 2.6.dev93`; `packs/science/classic.yaml` and `packs/instrument/jwst.yaml` exist; `Model.from_yaml('classic.yaml')` builds 53 features; `jwst.yaml` has `nirspec` and `miri`. *Not yet run:* the synthetic-spectrum fit itself | — |
| 14 | AmesPAHdbPythonSuite | **PASS (import).** Pinned commit `e8961b2` installs as `0.5.0.post96+ge8961b2`. The 2 GB database was not downloaded | — |
| 15 | MPS client limit | **Measured: 60 on driver 595.45.** 70 clients on GPU0: 60 attached, and 10 failed `cuInit` with error 808 (`CUDA_ERROR_MPS_MAX_CLIENTS_REACHED`); the server log says "Maximum number of clients (60) reached". The plan attributes 60 to "Release 620"; it already applies on R595 | §F |
| 16 | MPS pipe-directory length | **Trap found.** With `CUDA_MPS_PIPE_DIRECTORY` on a ~130-character path, the control daemon could not bind its Unix socket (108-byte limit) and exited. All 70 clients then succeeded **without MPS** and reported no error. With `/dev/shm/mps-$USER` it works. This is the plan's "silently runs without MPS" warning, with a concrete cause | §F |
| 17 | NGC `pytorch:26.10-py3` vs this driver | **PASS in the container** (see below). Before the build: From the registry config (arm64 manifest `sha256:c24a66e7…`): `CUDA_VERSION=13.4.3.001`, **`CUDA_DRIVER_VERSION=615.71.09`**, `/usr/local/cuda/compat/lib` on `LD_LIBRARY_PATH`. The node's 595.45 is older, so the image depends on CUDA forward compatibility. Test: `torch.zeros(1).cuda()` under `apptainer exec --nv`. Fallbacks: (a) NGC **`26.03-py3`**, whose `CUDA_DRIVER_VERSION` is exactly **595.45.04** (CUDA 13.2.0, PyTorch 2.11.0a0, still ≥ 2.10 for nvalchemi); (b) PyPI `torch 2.15.0+cu132` (item 6), tested here. Driver requirements of the 2026 tags: 26.01–02 → 590.48; **26.03 → 595.45**; 26.04–05 → 595.58; 26.06–07 → 610.43; 26.08 → 615.65; 26.09–10 → 615.71 | registry query |
| 18 | `uv pip check` warning | Harmless: `nvidia-cusparselt-cu13 0.8.1` carries the non-standard tag `manylinux2014_sbsa`, which uv flags as "built for a different platform". It loads and runs | `logB.txt` |

## Details

**§A, container A stack (venv A).** CPU torch first, then `mace-jax[jax-cuda12] @ git+…@e8a6f2d` (2026-10-29), `jax-md flax optax ase numpy scipy h5py mace-torch`. Resolved: jax/jaxlib 0.11.2, cuequivariance(-jax, -ops-cu12, -ops-jax-cu12) 0.10.0, e3nn-jax 0.21.0, jax-md 0.2.29, flax 0.12.12, mace-jax 0.2.0, mace-torch 0.3.17, torch 2.15.0+cpu, nvidia-*-cu12 12.9. mace-jax also has a `jax-cuda13` extra. Conversion: `mace-jax-from-torch --foundation off --model-name small --output off_small_jax.npz` writes `.npz` + `.json`. Prediction: `mace-jax-predict off_small_jax.npz benzene.xyz --dtype float32`, with `JAX_DEFAULT_MATMUL_PRECISION=highest`, `CUDA_VISIBLE_DEVICES=0`.

**§B, container B stack (venv B).** `torch "mace-torch[cueq,cueq-cuda-13]" cuequivariance==0.10.0 cuequivariance-torch==0.10.0 cuequivariance-ops-torch-cu13==0.10.0 ase h5py` from PyPI → torch 2.15.0+cu132 (CUDA 13.2), triton 3.9.0. Settings: `torch.set_float32_matmul_precision("highest")`, `cudnn.allow_tf32=False`, `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` unset; `mace_off(model="small", default_dtype="float32", device="cuda", enable_cueq=…)`. MACE logs "Default dtype float32 does not match model dtype float64, converting models to float32", which confirms that the stored checkpoint is float64.

**§C, force comparison.** Molecule: ASE `molecule("C6H6")`, the same geometry on both sides. This is a single small molecule at its reference geometry, enough for the feasibility gate. The plan's 20-species × 6-geometry equivalence study (§4.2) is still to do.

**§D, TorchSim.** `ts.io.atoms_to_state([C6H6, C6H6, CH4])`, then `MaceTorchSimModel(model=mace_off(..., return_raw_model=True), dtype=torch.float32, compute_forces=True, compute_stress=False)`. Compared with ASE `MACECalculator` per molecule.

**§E, PySR.** `JULIA_DEPOT_PATH` and `PYTHON_JULIAPKG_PROJECT` were set to local-NVMe paths. Run time 2 min 29 s wall time, including the Julia download and precompilation.

**§F, MPS.** `CUDA_VISIBLE_DEVICES=0 CUDA_MPS_PIPE_DIRECTORY=/dev/shm/mps-$USER/pipe CUDA_MPS_LOG_DIRECTORY=/dev/shm/mps-$USER/log nvidia-cuda-mps-control -d`, run as an ordinary user. Each client is a ctypes program that calls `cuInit`, retains the primary context and runs `cuMemAlloc`. `nvidia-smi` showed 60 `python3` clients plus `nvidia-cuda-mps-server`. The daemon was shut down with `echo quit | nvidia-cuda-mps-control`.

## Container builds (Apptainer 1.5.5, `apptainer build --fakeroot`, same day)

The images live on local NVMe in `/var/tmp/jjlee-pah/sif/`, with the Apptainer cache and tmp under `/var/tmp/jjlee-pah/`. Every test was run as an ordinary user, pinned with `CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0`.

| Image | Build | Size | First-build problem → fix | In-container result |
| --- | --- | --- | --- | --- |
| `A_jax.sif` | 169 s | 7.1 GB | **Built, but JAX silently ran on the CPU.** With the plan's `cuda:12.8.1-cudnn-devel` base, the image held CUDA 12.8 (toolkit) *and* CUDA 12.9 (jax pip wheels). Importing `cuequivariance_ops` before `jax`, as `mace-jax-from-torch` and `mace-jax-predict` do, loads the 12.8 libraries first. JAX's plugin then fails with "Unable to load cuSPARSE" and falls back to `CpuDevice`, still writing predictions. → `ubuntu:24.04` base, pip CUDA libraries on `LD_LIBRARY_PATH` via `$APPTAINER_ENVIRONMENT`, `JAX_PLATFORMS=cuda` | `[CudaDevice(id=0)]` with any import order; torch `2.15.0+cpu`; converter and predictor on the GPU; jax-md Langevin on the GPU; without `--nv` it now raises instead of using the CPU |
| `B_torch.sif` (NGC 26.10) | 263 s | 12 GB | `pip check` fails on the **untouched** NGC image: `cupti-python`, `triton-kernels`, `quack-kernels`, `flash-attn-4`. → record the base's `pip check` and fail only on new lines (there were none) | **Forward compatibility works on driver 595.45:** the compat `libcuda.so.615.71.09` is first on `LD_LIBRARY_PATH`. torch `2.15.0a0+875d815502.nv26.10`, CUDA 13.4; `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` unset by `%environment`; MACE-OFF small float32, E = −6324.114746 eV; cuEquivariance on (88 modules) and off; NGC torch and triton unchanged |
| `B_torchsim.sif` | 128 s | 13 GB | none | torch-sim 0.6.2, warp 1.19.0, nvalchemi 0.5.0 under the NGC constraints; batched forces against ASE 9.7e-7 eV/Å; a 100-molecule `nvt_langevin` run of 200 steps executed in 5.0 s, including warm-up (units and physics not checked) |
| `C_cpu.sif` | 211 s | 1.3 GB | (1) Ubuntu's `python3.12` has no shared `libpython`, so PythonCall precompilation failed. (2) With `libpython3.12-dev` added, the build passed, but juliacall pinned `OpenSSL_jll ~3.0` (Ubuntu Python's OpenSSL), which forced **Julia 1.11.9, and PySR segfaulted** in type inference (`jl_decode_value_array`) on the first fit. → uv-managed CPython 3.12.14 (OpenSSL 3.5.8), which gets Julia 1.13.3 | Run per the plan's run-time setup (`--containall --env SCRATCH=… --bind $SCRATCH --bind $SCRATCH/julia_env:/opt/julia_env`): `import pysr` 4.6 s, serial-deterministic and multiprocessing fits both recover the formula; PAHFIT packs present and `classic.yaml` gives 53 features; suite imports. Without the `julia_env` bind, `import pysr` fails on juliapkg's lock file, which bears out the plan's rule |

**Cross-container force gate** (benzene, float32, no TF32), JAX in A against NGC PyTorch in B, max |ΔF| in eV/Å: jax/torch 2.04e-6; jax/torch+cuEq 1.79e-6; jax+cuEq/torch 1.70e-6; jax+cuEq/torch+cuEq 1.58e-6. The tolerance is 1e-3. Both sides loaded the same checkpoint, sha256 `165cce4c…c46f`.

## Still open

1. TorchSim pilot run (100 molecules, 1 ps, with its unit and physics checks) and the PAHFIT synthetic-spectrum fit: weeks 1 and 4 work, not feasibility.
2. The 20-species × 6-geometry equivalence study (§4.2); only benzene has been checked so far.
3. Forward compatibility depends on the driver branch. Re-run the B check after any driver update and on the production cluster.
