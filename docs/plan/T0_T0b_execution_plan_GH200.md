# T0 and T0b Execution Plan on GH200

Stack benchmark and vertical slice for the symbolic-regression PAH diagnostics project · Oct 30, 2026 · @이재준

*Revised 2026-11-06 with the fixes from `T0_T0b_plan_review.md` and then `T0_T0b_committee_review.md`. GH200 facts in Sections 3, 4.3, 5.6 and 7 updated 2026-11-07 from measurements on the project node (`GH200_feasibility_check_2026-11-07.md`).*

## 1. Purpose and what success looks like

This document is a self-contained work plan for the first two months of the project. It is written for a graduate student who is comfortable with Python and Linux but may be new to astrochemistry, machine-learned force fields and symbolic regression. Read Section 2 first if any of those three terms is unfamiliar.

**The two tasks**

- **T0, the stack benchmark**, answers one question: for small molecules like PAHs, is a JAX-based molecular-dynamics stack (jax-md plus mace-jax) much faster than the standard PyTorch stack (ASE plus MACE) when many molecules are simulated at once? The answer decides which software the whole project is built on. "Much faster" is defined in advance as more than 5× in wall-clock time per nanosecond of simulation per molecule, at a batch of 100 molecules.
- **T0b, the vertical slice**, runs every stage of the project pipeline once on a deliberately tiny input, so that plumbing problems appear now rather than after a large compute allocation has been spent. It also includes a scientific sanity check, run as a *controlled closure test*: the symbolic-regression search must rediscover a band ratio we already know is a good ionisation tracer. Because the training data come from our own forward model, passing this test shows that the stages fit together; on its own it does not validate the physics. Section 5.9 adds the controls that make the test informative.

**Hardware.** Both tasks run on an NVIDIA GH200 node: a 72-core Grace CPU (ARM architecture) and an H100 GPU sharing one memory space. The ARM architecture is the main practical complication; Section 3 handles it.

**Success criteria**

| Task | Success means | Fails if |
| --- | --- | --- |
| T0 | A written decision memo with measured timings for both stacks at batch sizes 1, 10 and 100, at matched numerical precision; a verified equivalence between the two force-field implementations (same energies and forces within tolerance); a batched PyTorch baseline; and a clear recommendation | Timings are not reproducible; the two implementations give different forces, so the comparison is meaningless; the measurement is confounded by a missing optimisation (cuEquivariance) or by unmatched precision (float64 or TF32 on one side, Section 4.4) |
| T0b | The pipeline runs from database to real JWST spectra without manual fixes; the closure-test controls of Section 5.9 are run and reported, and the shuffled-label control does no better than the constant predictor; PySR finds a formula of complexity ≤ 7 that either rediscovers a known ionisation tracer (monotonic in 6.2/11.2 or 7.7/11.2, test error ≤ 1.1× the one-ratio baseline) or combines ratios to come within 2× of the random-forest reference; the formula is applied to the five Orion Bar template spectra and the results are reported with their domain coverage and uncertainties (Section 5.10) | Any stage needs hand intervention each run; a control fails; PySR cannot recover the known relation |

**What you will learn.** How PAH emission spectra are modelled from quantum-chemical data; how to run and time GPU molecular dynamics in two frameworks; how PAHFIT decomposes an infrared spectrum; how symbolic regression works in practice; and how to write a reproducible pipeline with provenance. These are the skills the rest of the project needs from you.

**Time.** Eight weeks, with weekly checkpoints (Section 6). T0 and T0b can be interleaved: T0b does not depend on the T0 result.

## 2. Background for a newcomer

**PAHs and the aromatic infrared bands.** Polycyclic aromatic hydrocarbons are flat molecules of fused benzene rings, 20 to a few hundred carbon atoms, with hydrogen on the edges. In space they absorb an ultraviolet photon, heat up to several hundred kelvin, and cool by emitting infrared light at the frequencies of their vibrations. The result is a set of broad emission bands at 3.3, 6.2, 7.7, 8.6, 11.2 and 12.7 µm seen almost everywhere in the universe. Because thousands of different PAHs contribute, the bands tell us about the *population*, not about one molecule: roughly, the 6–9 µm bands are strong in ionised PAHs, the 11.2 µm band is strong in neutral ones, and the 3.3 µm band is strong in small, hot ones. The project asks how to read those population properties from the bands as accurately as the data allow.

**Where the spectra come from.** We cannot measure most PAHs in the laboratory, so their spectra are computed with density functional theory (DFT) and collected in the NASA Ames PAH IR Spectroscopic Database (PAHdb). A computed spectrum is a list of vibrational frequencies and intensities for a cold molecule. To compare with space, it must be turned into an *emission* spectrum of a hot molecule (the "cascade model"), blurred to the resolution of a telescope, and then measured the way astronomers measure real data (PAHFIT). T0b does exactly this chain.

**Machine-learned force fields and MLMD.** A force field is a function that returns the energy and forces for a given set of atomic positions. Classical force fields use simple formulas; a machine-learned force field (also called an ML interatomic potential, MLIP) is a neural network trained on DFT data that reaches DFT accuracy thousands of times faster. MACE is the leading model family. Molecular dynamics (MD) integrates Newton's equations with those forces; if you also predict the molecule's dipole moment along the trajectory, the Fourier transform of the dipole autocorrelation is the infrared spectrum *including* temperature effects that DFT frequencies omit. This is "MLMD" and is the GPU-heavy part of the project. T0 benchmarks it.

**Why two software stacks.** The standard way to run MLMD is ASE (a Python MD driver on the CPU) calling a PyTorch MACE model on the GPU, one molecule at a time. PAHs are small, so the GPU is mostly idle and the Python loop dominates. JAX is a framework in which the whole MD loop can be compiled into one GPU program and *vectorised* over many molecules at once. The hypothesis is that this is much faster for our small, uniform molecules. T0 tests it.

**Symbolic regression.** Ordinary regression fits numbers inside a formula you chose. Symbolic regression searches for the formula itself, using an evolutionary algorithm over expression trees, and returns a set of candidate formulas trading accuracy against complexity (a Pareto front). PySR is the standard tool. The attraction for this project is that the result is a readable formula like the band ratios astronomers already use, but derived from tens of thousands of spectra instead of intuition.

**Glossary**

| Term | Meaning here |
| --- | --- |
| AIB | Aromatic infrared band, one of the PAH emission features |
| PAHdb | NASA Ames PAH IR Spectroscopic Database, our source of computed spectra |
| Cascade model | Converts a cold computed spectrum to a hot emission spectrum after UV absorption |
| PAHFIT | Python package that decomposes an infrared spectrum into PAH bands, dust continuum and lines |
| MIRI MRS | JWST's mid-infrared integral-field spectrograph, 5–28 µm; our target instrument mode |
| PDRs4All | The JWST early-release programme on the Orion Bar; provides five public template spectra |
| MLIP, MACE | Machine-learned interatomic potential; MACE is the model family we use |
| ASE | Atomic Simulation Environment, Python toolkit for setting up and running atomistic simulations |
| jax-md, mace-jax, e3nn-jax | JAX implementations of MD, MACE and the equivariant layers it is built from |
| Batch size | Number of molecules simulated simultaneously in one GPU call |
| ns/day | Nanoseconds of simulated time per day of wall-clock; the standard MD speed unit |
| PySR | Symbolic regression package, Python front end over a Julia search engine |
| Pareto front | The set of formulas where no other is both simpler and more accurate |
| Apptainer | Container system for HPC (formerly Singularity); packages our software so it runs identically everywhere |
| aarch64 / ARM | The CPU architecture of the Grace chip; software must be built for it, not for x86 |

## 3. Prerequisites and environment on the GH200

### 3.1 Before you start

- An account on the GH200 node with a scratch allocation of at least 200 GB and the ability to run Apptainer (or Docker, if the node allows it). On the project node (checked 2026-11-07) no container runtime was installed, and Ubuntu 24.04's `kernel.apparmor_restrict_unprivileged_userns=1` blocks unprivileged Apptainer. Install the `apptainer-suid` package, or add an AppArmor profile for Apptainer. Put scratch on local NVMe, not the NFS `/home`: the Julia lock files, the Apptainer cache and the MPS pipes all live there.
- A GitHub account; the project repository will hold every script you write. Commit daily.
- Read: the PDRs4All Orion Bar paper (Chown et al. 2024) for what the data look like, the PAHdb v4.00 paper (Ricca et al. 2025) or the website "about" page for how computed spectra are produced, the MACE and PySR documentation front pages. Two days of reading is enough to start; deeper reading comes with each stage.

### 3.2 Hardware notes

The GH200 is one Grace CPU (72 ARM Neoverse V2 cores, up to 480 GB LPDDR5X) connected to one H100 GPU (96 GB HBM3 or 144 GB HBM3e, depending on the variant; read the actual figure from `nvidia-smi`) by NVLink-C2C, which lets the GPU address CPU memory directly. Two consequences matter for us. First, every binary must be built for aarch64; an x86 container or wheel will not run. Second, moving data between CPU and GPU is much cheaper than on a conventional node, so the PyTorch-plus-ASE baseline will look better here than on the production cluster. Record this in the T0 memo and plan to repeat the key timing on one node of the target cluster later.

**The project node is a dual GH200** (checked 2026-11-07): two superchips, each a 72-core Grace with about 480 GB LPDDR5X plus a GH200 144G HBM3e GPU. That makes 144 cores, 2 GPUs and 18 NUMA nodes (each GPU's memory also appears as a NUMA node: ATS addressing, C2C on). The unit of T0 is **one superchip**: one GPU and its own Grace cores and memory. Pin every T0 run with `CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0` (GPU0 ↔ CPUs 0–71). Otherwise PyTorch workers or MPS clients can straddle sockets and cross the slower inter-socket link. Keep the other superchip idle during timed runs, and record the pinning in the CSV. The kernel uses **64 KiB pages**; every component in Section 3.3, including Julia, was checked to work with that.

Check the node first:

```
uname -m            # must print aarch64
nvidia-smi          # shows the GPU(s) and the driver version; note the CUDA version it supports
nvidia-smi topo -m  # which CPUs and NUMA node belong to each GPU
nproc               # 72 per Grace; 144 on a dual-GH200 node
getconf PAGESIZE    # 65536 on the project node (64 KiB-page kernel)
apptainer --version
```

### 3.3 Build the container

The four recipes below were built with `apptainer build --fakeroot` and tested on the project node on 2026-11-07. They are kept verbatim in `containers/*.def`, with the in-image `pip list` in `containers/requirements-*-container.txt` and the image hashes in `containers/images-sha256-2026-11-07.txt` (results: `GH200_feasibility_check_2026-11-07.md`). Use aarch64 base images and install everything with pinned versions. Build in two containers, one per stack, so that a problem in one does not block the other; a third small container holds PySR and PAHFIT (CPU only).

Two rules apply to all three recipes. First, use Python 3.12: current JAX needs ≥ 3.12 and current mace-torch needs ≥ 3.11, so a Python 3.10 base (Ubuntu 22.04, or NGC PyTorch images before the Ubuntu 24.04 switch) silently resolves to old releases or fails. Second, install into one virtual environment per container and always call `python -m pip`, so that `pip` and `python` cannot point at different interpreters (Ubuntu has no bare `python` command outside a venv).

**Container A: JAX stack**

```
Bootstrap: docker
From: ubuntu:24.04          # NOT a CUDA-toolkit base: jax[cuda12] brings its own CUDA 12.9 wheels; see below
%post
  set -e
  apt-get update && apt-get install -y python3.12 python3.12-venv git ca-certificates
  python3.12 -m venv /opt/venv && . /opt/venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # FIRST: CPU torch, so no second CUDA runtime beside JAX's
  python -m pip install "mace-jax[jax-cuda12] @ git+https://github.com/ACEsuit/mace-jax@e8a6f2d3e793e2020439df1736a32703c9cac170"   # 2026-10-29; JAX 0.11.2, cuEquivariance-JAX 0.10.0
  python -m pip install jax-md flax optax ase numpy scipy h5py matplotlib
  python -m pip install mace-torch      # needed only by the weight converter (it imports mace.calculators)
  python -c "import torch; assert torch.__version__.endswith('+cpu'), torch.__version__"
  python -m pip check
  python -m pip list --format=freeze > /opt/requirements-frozen.txt
  # cuequivariance_ops loads libcue_ops.so before JAX has loaded its CUDA libraries: put them on the loader path
  echo "export LD_LIBRARY_PATH=$(ls -d /opt/venv/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')\${LD_LIBRARY_PATH}" >> $APPTAINER_ENVIRONMENT
%environment
  export PATH=/opt/venv/bin:$PATH
  export JAX_DEFAULT_MATMUL_PRECISION=highest   # no TF32; see Section 4.4
  export JAX_PLATFORMS=cuda                     # fail loudly instead of silently falling back to CPU (run with --nv)
```

Let mace-jax choose the JAX version: it caps JAX below 0.12 and pins cuEquivariance to 0.10.x for its backend, and a separately installed newer JAX will conflict. PyTorch is used here only for the conversion. The CPU wheel must be installed *before* mace-jax. mace-jax depends on `e3nn==0.4.4`, the PyTorch e3nn, so installing it first pulls PyPI's default aarch64 torch (`2.15.0+cu132`, with the CUDA 13 libraries). After that, the CPU-index install is a silent no-op ("already satisfied"). Checked on the project node 2026-11-07; the assert above catches it. **Keep exactly one CUDA in this image** (checked 2026-11-07). The first build used the `cuda:12.8.1-cudnn-devel` base, so its CUDA 12.8 libraries sat beside the CUDA 12.9 wheels that `jax[cuda12]` installs. When `cuequivariance_ops` was imported before JAX, which mace-jax's own CLIs do, the 12.8 libraries loaded first. JAX's cuSPARSE then failed to load, and JAX **silently fell back to the CPU**, still writing predictions. Hence the plain `ubuntu:24.04` base. `cuequivariance_ops` still needs `libnvrtc.so.12` and `libcublas.so.12` before JAX has loaded them, hence the `LD_LIBRARY_PATH` line. Without it, the import, and with it `mace-jax-from-torch`, fails with `libnvrtc.so.12: cannot open shared object file`. `JAX_PLATFORMS=cuda` turns any CPU fallback into an error. Every MD driver should still assert that `jax.devices()[0].platform == 'gpu'`. The image is 7.1 GB. mace-jax also offers a `jax-cuda13` extra. Neither jax-md nor mace-jax pins `e3nn-jax`, so after the build run the jax-md Lennard-Jones example and smoke test 3, to confirm that both work on the release pip resolved. Pin the exact versions once the first build works (`python -m pip freeze > requirements-jax.txt`) and rebuild from the frozen list. Verify that JAX sees the GPU:

```
python -c "import jax; print(jax.devices())"   # expect [CudaDevice(id=0)]
```

**Container B: PyTorch stack**

```
Bootstrap: docker
From: nvcr.io/nvidia/pytorch:26.10-py3        # verified 2026-11-07: tag has an arm64 image; PyTorch 2.15.0a0, CUDA 13.4, built for driver 615.71 (see below)
%post
  set -e
  python -m pip list --format=freeze | grep -i -E '^(torch|triton|nvidia-)' > /opt/ngc-constraints.txt
  python -m pip check > /opt/pip-check-ngc-base.txt 2>&1 || true     # 26.10 already fails pip check; see below
  python -m pip install -c /opt/ngc-constraints.txt \
      "mace-torch[cueq,cueq-cuda-13]" "cuequivariance==0.10.0" "cuequivariance-torch==0.10.0" \
      "cuequivariance-ops-torch-cu13==0.10.0" ase h5py
  python -m pip check > /opt/pip-check-after.txt 2>&1 || true
  if grep -vxF -f /opt/pip-check-ngc-base.txt /opt/pip-check-after.txt; then exit 1; fi   # fail only on NEW problems
  python -c "import torch; print(torch.__version__, torch.version.cuda)"   # must match the NGC release notes
  python -m pip list --format=freeze > /opt/requirements-frozen.txt
%environment
  unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE        # 26.10 sets this to 1 (checked in the image config); see Section 4.4
```

Use `pip list --format=freeze`, not `pip freeze`: NGC's locally built PyTorch can appear as `torch @ file://…`, and pip rejects URLs in a constraints file. A plain `pip check` fails on the untouched 26.10 image. It reports `cupti-python` (missing `nvidia-cuda-cupti`), `triton-kernels` (missing `pytest`) and `quack-kernels`/`flash-attn-4` (`nvidia-cutlass-dsl` 4.8.0 against 4.6.x), all NVIDIA's own packages. So the recipe records the base's complaints and fails only on new ones. The install added none. NGC's torch and triton were unchanged, and the image is 12 GB.

**Build TorchSim as a separate candidate image**, so that a failure cannot break the baseline. First build and verify the ASE + MACE image above. Then build a second image from it that adds `torch-sim-atomistic` under the same constraints, and re-run `pip check` and the PyTorch-version check. If the candidate fails, keep the verified baseline image and pilot the concatenated-`Atoms` fallback (Section 4.3).

**cuEquivariance 0.10.0 and the NGC PyTorch.** The 0.10.0 ops wheels (April 2026) declare no PyTorch version range. That is not an ABI risk: the cp312 aarch64 extension is a nanobind module that links only `libcue_ops.so`, `libcuda.so.1` and system libraries, with no `libtorch`/`libc10` (checked with `readelf` on 2026-11-07). The kernels should therefore load against the NGC 26.10 PyTorch (2.15.0a0). Still, the build must check that MACE reports the cuEquivariance backend at model load. If it does not, step back to an earlier 2026 NGC tag rather than forward to cuEquivariance 0.13.

**Driver match (checked 2026-11-07).** The project node's driver is 595.45.04 (CUDA 13.2). The 26.10 image is built for driver 615.71.09 and runs on 595 only through CUDA forward compatibility: it puts `/usr/local/cuda/compat/lib` on `LD_LIBRARY_PATH`. So the first check after building is `apptainer exec --nv pytorch.sif python -c "import torch; torch.zeros(1).cuda()"`. **Result on the project node: it works.** Inside the image, the compat `libcuda.so.615.71.09` comes ahead of the host driver on `LD_LIBRARY_PATH`. Smoke test 2 passed with cuEquivariance active (88 modules) and `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` unset. Forward compatibility applies only to data-centre GPUs and only to supported driver branches, so re-check it after any driver change and on the production cluster. If it fails, use **`26.03-py3`**. Its `CUDA_DRIVER_VERSION` is exactly 595.45.04 (CUDA 13.2.0, PyTorch 2.11.0a0, still ≥ 2.10 for `nvalchemi-toolkit-ops[torch]`). Driver requirements of the 2026 tags, read from the registry: 26.01–02 → 590.48; 26.03 → 595.45; 26.04–05 → 595.58; 26.06–07 → 610.43; 26.08 → 615.65; 26.09–10 → 615.71. A second tested fallback is PyPI `torch` on a CUDA 13 base: on aarch64 the default wheel is `2.15.0+cu132`. With mace-torch 0.3.17 and cuEquivariance 0.10.0 (cu13), it passed smoke test 2 and the force gate on the project node, and the cuEquivariance backend was confirmed active (88 `cuequivariance_torch` modules in the loaded model).

This container deliberately has no venv: the NGC image's own Python is the one with the CUDA-enabled PyTorch, and the constraints file stops pip from replacing that PyTorch with a PyPI build. NGC PyTorch has shipped CUDA 13 since release 25.08, so use the `-cu13` cuEquivariance packages, not `-cu12`. If you choose an older CUDA 12 tag instead, switch all of them to `-cu12` together.

`cuequivariance` provides the fused kernels that make MACE fast. Pin **0.10.x in both stacks**. mace-torch 0.3.17 caps the ops packages below 0.11 and `cuequivariance-torch` below 0.12, and mace-jax requires the 0.10 family. The newer 0.13 wheels install on aarch64 but fall outside both stacks' declared ranges. Because mace-jax uses the same backend, cuEquivariance is a variable for *both* stacks, not a handicap for one (Section 4.4). TorchSim (`torch-sim-atomistic`) is the candidate batched PyTorch driver (Section 4.3). Its dependencies resolve on aarch64 on paper (checked 2026-11-07): `torch-sim-atomistic` 0.6.2 and `nvalchemi-toolkit-ops` 0.5.0 are pure-Python wheels; `nvalchemi-toolkit-ops[torch]` needs torch ≥ 2.10, and 26.10 ships 2.15.0a0; and `warp-lang` 1.19.0 has a `manylinux_2_34_aarch64` wheel. On the project node (2026-11-07, PyPI torch 2.15.0+cu132), it installed without replacing torch. `MaceTorchSimModel(..., dtype=torch.float32, compute_stress=False)` on a mixed three-molecule batch matched per-molecule ASE + MACE to 6.5e-7 eV/Å. Inside the NGC 26.10 image (`B_torchsim.def`, built on top of B), it installed under the constraints without changing torch. Batched forces matched ASE to 9.7e-7 eV/Å, and a 100-molecule batched Langevin run of 200 steps executed on the GPU. That run's units and physics were not checked. The 1 ps pilot run of Section 4.3 is still to be done. It keeps its own image. Verify:

```
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Container C: CPU tools**

```
Bootstrap: docker
From: ubuntu:24.04
%post
  set -e
  apt-get update && apt-get install -y git ca-certificates curl
  curl -LsSf https://astral.sh/uv/0.12.12/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
  export UV_PYTHON_INSTALL_DIR=/opt/python
  uv python install 3.12                     # NOT Ubuntu's python3.12: see "Julia version" below
  uv venv --seed -p 3.12 /opt/venv && . /opt/venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install pysr astropy specutils scikit-learn pandas h5py matplotlib jupyter
  python -m pip install "git+https://github.com/PAHFIT/pahfit.git@ca599f9a1eb57a9ed3305651e77fcc9d8e039333"   # master HEAD, 2024-11-01; has packs/science/classic.yaml and packs/instrument/jwst.yaml. NOT `pip install pahfit`; see below
  python -m pip install "git+https://github.com/PAHdb/AmesPAHdbPythonSuite.git@e8961b2bf8b0ae663663a6790d452e1fe9fd1bd3"   # HEAD, 2026-07-30; the latest tag (v0.5.0) is from 2022
  export JULIA_DEPOT_PATH=/opt/julia_depot PYTHON_JULIAPKG_PROJECT=/opt/julia_env
  python -c "import pysr"        # first import downloads Julia and compiles the backend via juliacall
  chmod -R a+rX /opt/julia_depot /opt/julia_env /opt/python
  python -c "import json; print('julia', json.load(open('/opt/julia_env/pyjuliapkg/meta.json'))['version'])"   # expect 1.13.x
  python -m pip check
%environment
  export PATH=/opt/venv/bin:$PATH
  export PYTHON_JULIAPKG_PROJECT=/opt/julia_env     # must be writable at run time: bind a copy over it (see below)
  export PYTHON_JULIAPKG_OFFLINE=yes                 # do not install Julia or packages at run time
  export JULIA_DEPOT_PATH=${SCRATCH:-$HOME}/julia_depot:/opt/julia_depot   # writable depot first, baked one second
```

**Julia version (checked 2026-11-07).** juliacall pins `OpenSSL_jll` to the OpenSSL that the *Python interpreter* links. Ubuntu 24.04's `python3.12` links OpenSSL 3.0, which forces Julia 1.11 (1.11.9 was installed). On the project node, PySR then segfaulted inside Julia's type inference (`jl_decode_value_array`) on the first fit. The root cause was not investigated further. Ubuntu's Python also lacks the shared `libpython3.12.so` that PythonCall needs, so the build failed until `libpython3.12-dev` was added. uv's python-build-standalone CPython 3.12.14 links OpenSSL 3.5 and ships `libpython`. With it, juliapkg installs Julia 1.13.3, and PySR works in both serial-deterministic and multiprocessing mode. The image is 1.3 GB, and the writable `/opt/julia_env` copy is 1.1 GB, because it contains the Julia installation.

**Run-time Julia setup.** Both the juliapkg project and the first Julia depot must be writable at run time. juliapkg takes a lock file inside the project directory before it checks anything, even in offline mode. So, per job:
1. Copy the baked project out of the image once: `apptainer exec pysr.sif cp -r /opt/julia_env $SCRATCH/julia_env`.
2. Bind the copy over the baked path, keeping that path: `apptainer run --containall --env SCRATCH=$SCRATCH --bind $SCRATCH --bind $SCRATCH/julia_env:/opt/julia_env pysr.sif …`. `--containall` implies `--cleanenv`, so pass `SCRATCH` explicitly, or the depot path falls back to the container's empty `$HOME`. Without the `julia_env` bind, `import pysr` fails on juliapkg's lock file (checked).
3. Under `--containall`, bind the writable depot explicitly, as above, and keep `/opt/julia_depot` second in `JULIA_DEPOT_PATH`.

**PAHFIT must come from git.** PyPI holds a single release, 2.0 from February 2022. It has the old API and `scipack_classic.yml`, and lacks `Model.from_yaml`, the `classic.yaml` science pack and the `instrument/jwst.yaml` pack that Sections 5.5–5.6 rely on. Those exist only from tag v2.5 onward and on the trunk, which the PAHFIT install page recommends. The import name is the same, so a plain `pip install pahfit` fails only at Section 5.6. Pin a commit that contains `pahfit/packs/science/classic.yaml` and `pahfit/packs/instrument/jwst.yaml`, and record its hash: it is "the PAHFIT version" in the Section 5.11 provenance, because no release carries the packs.

PySR no longer has an install step: `pysr.install()` was removed and now only prints a warning. Julia and its packages are fetched through `juliacall` on the first `import pysr`, which takes several minutes, so do it during the build with the depot set to a path inside the image. At run time the image is read-only, so follow the run-time setup above. `PYTHON_JULIAPKG_OFFLINE=yes` is juliapkg's documented offline mode. It stops package installation, but it does not remove the need for a writable project directory. PySR's repository ships an `Apptainer.def` that is a useful starting point. Julia supports Linux aarch64 as a full platform. On the project node's 64 KiB-page kernel (2026-11-07), PySR 2.7.2 with juliacall 0.9.36 installed Julia 1.13.3 at first import (about 2 minutes). The quick-start recovered its formula both with `deterministic=True, parallelism="serial"` and with `parallelism="multiprocessing", procs=4`. juliapkg puts the Julia binary itself inside the project directory (`/opt/julia_env/pyjuliapkg/install/`), so the writable copy of step 1 contains it. Project plus depot came to about 1.5 GB. Verify as a non-root user with `apptainer run --containall`, using the PySR quick-start example on a 100-point synthetic dataset.

### 3.4 Smoke tests (do all four before any real work)

1. *JAX MD:* run the jax-md Lennard-Jones example on the GPU for 1,000 steps; confirm it runs and that `jax.devices()` shows the GPU.
2. *MACE on PyTorch:* load the small MACE-OFF23 model with `mace_off(model="small", default_dtype="float32", device="cuda")`. The explicit dtype matters: `mace_off` defaults to float64, while `mace_mp` defaults to float32. Compute energy and forces for coronene, and print the force on the first atom. Coronene is not in ASE's built-in molecule list, so write an XYZ file from the PAHdb geometry and read that.
3. *Same molecule in mace-jax:* convert *the exact checkpoint used in smoke test 2*, using `--foundation off --model-name small`, and record the checksum of the cached `MACE-OFF23_small.model` that both sides load. `--torch-model <file>.model` fails under PyTorch ≥ 2.6: `torch.load` defaults to `weights_only=True` and rejects the pickled MACE model (checked 2026-11-07). If you need it for a non-foundation checkpoint, load the file yourself with `weights_only=False`, and only for a trusted file. If you leave out `--model-name`, the converter falls back to `mace_off()`'s default, which is *medium*, and the two numbers will not agree. The converter has no dtype option. The loader has one (`load_model_bundle(path, "float32")`, `mace-jax-predict --dtype float32`), but it only switches `jax_enable_x64` off and leaves float64 leaves in the parameter tree. So still cast the parameters to float32 and assert that `jax.config.jax_enable_x64` is off. Then compute energy and forces and print the same number. mace-jax uses cuEquivariance only when the bundle JSON says so: `"equivariance_config": {"backend": "cueq"}`. The default is the plain-JAX path. On the project node, benzene agreed to 2e-6 eV/Å in every torch/jax × cuEq on/off pairing (2026-11-07). The two numbers should agree to better than 1e-3 eV/Å, the same float32 tolerance as Section 4.2. If they do not, stop and resolve it: T0 depends on this.
4. *PAHFIT:* build a model with `Model.from_yaml('classic.yaml')` and fit one short synthetic spectrum whose `meta['instrument']` is a `jwst.miri.mrs.*` segment name from `jwst.yaml`. Plot the result. This exercises exactly the packs used in Section 5.6; the generic documentation example does not.

When all four pass, tag the repository `env-ok` and record the container image hashes in the README.

### 3.5 Data to download

| Data | Where | Size | Used in |
| --- | --- | --- | --- |
| PAHdb computational library v4.00 (XML or the Python suite's cached format) | astrochemistry.org/pahdb, download page | \~2 GB | T0 geometries, T0b spectra |
| MACE foundation-model weights (MACE-MP-0 small and medium, or MACE-OFF23) | MACE GitHub releases | \~100 MB | T0 |
| PDRs4All Orion Bar template spectra (five zones, NIRSpec + MIRI MRS) | MIRI: CDS catalogue J/A+A/685/A75 (Chown et al. 2024); NIRSpec: CDS J/A+A/685/A74 (Peeters et al. 2024); apertures and cubes: MAST PDRs4All HLSP, doi:10.17909/wqwy-p406 | \~50 MB | T0b |
| JWST MIRI MRS resolving-power curves | The per-band polynomials in PAHFIT's instrument pack `pahfit/packs/instrument/jwst.yaml` (use these so that simulation and fitting share one instrument model). Note that the pack's MIRI coefficients are pre-flight estimates (Labiano et al. 2021); the flight-derived trend is Argyriou et al. 2023, Eq. 1, R = 4603 − 128λ. Compare a few values and note the difference in the report: sharing the pack gives consistency, not independent validation. | small | T0b instrument model |
| AmesPAHdbPythonSuite | GitHub, PAHdb organisation | pip | T0b cascade reference |

## 4. T0: the stack benchmark, step by step

The goal is one number per stack: wall-clock seconds per picosecond of simulation per molecule, at batch sizes 1, 10 and 100, with the two stacks running the *same* model on the *same* molecules at the *same* numerical settings. Everything below serves that fairness.

### 4.1 Choose the 20 species

Take neutral PAHs from PAHdb spanning sizes evenly: about five each in the ranges 20–30, 30–50, 50–80 and 80–130 carbon atoms (for example naphthalene is too small; pyrene C16H10 can be added as an extra small anchor outside the four ranges; then coronene C24H12, ovalene C32H14, circumcoronene C54H18, up to circumcircumcoronene C96H24 and a couple of irregular shapes). Record for each: PAHdb UID, formula, number of atoms, and whether it is compact or irregular. Export the optimised geometry as an XYZ file. The spread in atom count matters: JAX pads all molecules in a batch to the largest, so a wide spread tests the padding cost.

### 4.2 Get one model into both frameworks

Use a published MACE foundation model, not one you train: MACE-MP-0 (small) is the simplest; MACE-OFF23 is trained on organic molecules and is more appropriate for PAHs. Load it in PyTorch with `mace.calculators.mace_off(..., default_dtype="float32")` or `mace_mp`. Always pass the dtype: `mace_off` defaults to float64. Convert the weights with mace-jax's `mace-jax-from-torch` tool. Use `--torch-model` on the same `.model` file the PyTorch side loads, or `--foundation off --model-name small`; without `--model-name`, the medium model is converted. Both frameworks must use the identical checkpoint, so record its checksum. The tool takes only `--torch-model`, `--foundation`, `--model-name` and `--output`; there is no dtype flag (`--torch-param-dtype` belongs to `mace-jax-train`). So cast the converted parameters to float32 when the JAX driver loads them. Do one single-molecule conversion in week 1 for smoke test 3, and the full equivalence study in week 3. Note that MACE-OFF is trained only on neutral, closed-shell molecules (Kovács et al. 2025): fine for this timing benchmark, which uses neutral PAHs, but it cannot be used for cation trajectories later in the project.

**Equivalence check (mandatory).** For all 20 molecules, compute energy and forces in both frameworks at the DFT geometry and at five randomly perturbed geometries (displace every atom by a Gaussian of 0.05 Å). Report the maximum absolute force difference and the energy difference. Target: forces agree to better than 1e-3 eV/Å and energies to better than 1e-4 eV per atom (but see `GH200_feasibility_check_2026-11-07.md` §I: at float32 the energy target sits at the rounding floor of the total energy for molecules above ~50 atoms, so it needs a decision before week 3), with both stacks at the precision settings of Section 4.4 (float32, no TF32). Repeat the check once per cuEquivariance setting, because custom kernels need their own verification. If the agreement is worse, the implementations differ and the benchmark is not comparing like with like. Investigate before going on: cutoff radius, neighbour-list convention, dtype, and TF32.

### 4.3 Write the two MD drivers

Both drivers must use: the Langevin thermostat at 600 K with friction 0.01 fs⁻¹, a 0.5 fs timestep, 20,000 steps (10 ps) after 2,000 steps of equilibration, positions and velocities initialised identically from a fixed seed, and no constraints. Save a per-step dipole proxy so that you also exercise the output path: MACE does not output charges, so use fixed charges (for example −0.1 e on each C bonded to H and +0.1 e on each H) times the positions; the real dipole model comes later. Mai et al. 2025, the MLMD workflow the project builds on, used a different force field (NeuralIL, in JAX) and ran 200 ps Nosé–Hoover NVT followed by 200 ps NVE at 0.5 fs. If time allows, also time an NVE segment, since that is the production ensemble.

**PyTorch baseline (`md_torch.py`).** MACE on `cuda` at the precision settings of Section 4.4. Run it three ways. In every mode, "batch *b*" means *b* trajectories of total work, not *b* simultaneous processes.

- *Sequential*, which is how the standard workflow behaves: ASE `Atoms` for each molecule, `ase.md.langevin.Langevin`, the *b* molecules one after another in one process. This scales linearly, so measure it at batch 1 and 10 and extrapolate to 100, rather than spending about 10 GPU-hours per repeat on it.
- *Concurrent (MPS pool)*: P worker processes share the GPU through CUDA MPS (`nvidia-cuda-mps-control -d`) and take the *b* trajectory jobs from a shared queue. Each worker handles at most ⌈b/P⌉ jobs, so the allocation can be uneven. Assert that exactly *b* jobs completed, and set a CPU thread limit per worker (`OMP_NUM_THREADS`, `torch.set_num_threads`) so the workers do not oversubscribe the cores.
  - MPS accepts at most 60 client contexts per device on current drivers (default `CUDA_DEVICE_MAX_CONNECTIONS=2`), and 48 on older ones. Measured on the project node's driver 595.45 (2026-11-07): the 61st client fails `cuInit` with error 808 (`CUDA_ERROR_MPS_MAX_CLIENTS_REACHED`).
  - Keep P ≤ 48 and below the number of cores; try two or three values and record P.
  - A client that cannot reach the MPS daemon silently runs *without* MPS. Confirm in the MPS server log that all P clients attached. One concrete cause: `CUDA_MPS_PIPE_DIRECTORY` holds a Unix socket, and socket paths are limited to 108 bytes. On a long path the control daemon exits at start-up, and every client then runs without MPS and reports success. Use a short path such as `/dev/shm/mps-$USER/pipe` and the matching `CUDA_MPS_LOG_DIRECTORY`, so that a per-user daemon needs no root. Start it inside the same `CUDA_VISIBLE_DEVICES=0`/`numactl` pinning as the workers.
- *Batched*: many molecules in one GPU call, which is the PyTorch analogue of the JAX batch and the real answer to "you did not batch PyTorch". There are two candidates. A **week-4 pilot on the GH200** decides which one is used, and the memo must say which.
  - **Primary: TorchSim** (`torch-sim-atomistic`) with MACE 0.3.17's first-party adapter, `MaceTorchSimModel(..., dtype=torch.float32, compute_forces=True, compute_stress=False)`. Its default dtype is float64, and its default computes stress, which is extra work the JAX driver does not do. TorchSim runs batched Langevin NVT natively. It uses metal units: `timestep` in ps (0.0005 for 0.5 fs) and γ in ps⁻¹ (10 for 0.01 fs⁻¹). Pass γ explicitly: the default, 1/(100 dt), is 20 ps⁻¹. A 1 ps, 100-molecule run already passes on the project node; there the MACE force call is host-bound (synchronisations in the wrapper, GPU busy ~5% with cuEquivariance), so profile it before treating its timing as the PyTorch ceiling (`GH200_feasibility_check_2026-11-07.md` §H).
  - **Fallback: concatenated ASE `Atoms`**: all *b* molecules in one `Atoms` object, at least 30 Å apart (more than twice the model cutoff), so that one MACE call evaluates the whole batch. Assert at the end of the run that the minimum distance between molecules stayed above twice the cutoff.
  - **Equal work:** every timed driver computes the same outputs (energy, forces and the dipole proxy) and nothing more. Disable stress and virial evaluation wherever possible, and disclose any extra work that cannot be avoided.
  - **Centre-of-mass handling** differs between the libraries' defaults. ASE's Langevin has `fixcm=True`, applied per `Atoms` object, so separate molecules and one concatenated `Atoms` are not treated the same. jax-md's and TorchSim's behaviour depends on their integrator options and version. Choose one policy for every mode: identical initial momenta and unconstrained COM motion during production (`fixcm=False` in ASE, no COM constraint in jax-md or TorchSim). Set it explicitly, count temperature degrees of freedom consistently (excluding padding), and verify the policy in the pilot.
  - **The pilot passes a candidate if** it installs on aarch64; its batched forces match isolated ASE + MACE to within 1e-3 eV/Å; no neighbour edges appear between molecules during the run; and a 100-molecule, 1 ps run completes. If TorchSim fails, the fallback becomes primary. If neither passes, report the narrower comparison (JAX against ASE sequential and MPS only), and say in the memo that the general framework decision is deferred.

**JAX driver (`md_jax.py`).** mace-jax provides no ASE calculator, no jax-md hook and no documented public energy function. Its batching is a padded `jraph` graph with caps on `n_node`, `n_edge` and `n_graph`, not a `(b, N_max, 3)` array under `vmap`. So build the energy function yourself from `mace_jax.tools.bundle.load_model_bundle` and the model's predictor, and design the driver around **one padded batched graph holding all *b* molecules**. Budget week 3 for this; it is the main schedule risk in T0.

- **Integrator:** drive the graph with `jax_md.simulate.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma)` over `space.free()`. Use `jax.lax.scan` over steps in chunks of 1,000 so the trajectory stays on the GPU, and `jax.jit` the whole chunk function.
- **Units:** jax-md has no fixed units, and `init_fn` defaults to `mass=1.0`. Pass the atomic masses (amu) explicitly. In eV, Å and amu, one time unit is about 10.18 fs, so dt = 0.5 fs is 0.0491 and γ = 0.01 fs⁻¹ is about 0.102. This is the same convention as ASE's `units.fs`. Pass kT in eV, not a temperature in kelvin.
- **Padding:** give padded atoms zero force, pin them far away, and exclude them from the temperature and every other observable. Check that padded and unpadded forces agree.
- **Overflow:** detect neighbour-list or edge-cap overflow and recover from it (reallocate and re-run the chunk); never let it truncate silently.
- **Compile time:** exclude the first call's compile time from the timing, but *record it separately*. It is a real cost when molecule sizes vary. If `N_max` would be far above most molecules, bucket the species by atom count (Section 7) and record the bucketing in the CSV.

### 4.4 Benchmark protocol

| Setting | Values |
| --- | --- |
| Batch size | 1, 10, 100 trajectories (for 100, cycle through the 20 species five times). PyTorch sequential at 100 is extrapolated; PyTorch MPS pool and PyTorch batched are measured |
| Precision parity | Matched float32 with no TF32 on both sides, set explicitly, because the library defaults silently differ. **PyTorch:** pass `default_dtype="float32"` (`dtype=torch.float32` for TorchSim); unset `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE`; call `torch.set_float32_matmul_precision("highest")`; set `torch.backends.cudnn.allow_tf32 = False`; no autocast. **JAX:** set `JAX_DEFAULT_MATMUL_PRECISION=highest`; cast the converted parameters to float32; assert `jax_enable_x64` is off. Print the effective settings at start-up and store them in the CSV. JAX uses TF32 for float32 matrix multiplies on the H100 by default, which alone could move the 5× decision. Reduced-precision (TF32) runs are optional and go in a separate column, never in the decision |
| cuEquivariance | Run each stack with and without cuEquivariance (a 2 × 2 grid) at batch 1 and 100; record the cuEquivariance version and the backend MACE reports; state which combination the decision rests on |
| Repeats | 3 independent runs per setting; report median and spread |
| Timing | `time.perf_counter()` around the production steps only. JAX dispatches asynchronously, so before starting the clock, block on the warm-up call's returned state with `jax.block_until_ready(state)`. Synchronise the GPU before stopping the clock: `torch.cuda.synchronize()` in PyTorch, `jax.block_until_ready(result)` on the production call's returned state and outputs in JAX. Make output buffering and flush boundaries the same in both stacks |
| Metric | Aggregate cost: seconds per molecule-picosecond = wall time / (10 ps × b); its reciprocal is aggregate throughput. Also report per-trajectory speed in ns/day, which is what one molecule experiences. The two differ when *b* > 1 |
| Also record | GPU utilisation (`nvidia-smi dmon`), peak GPU memory, compile time (JAX), CPU utilisation (`top`), MPS worker count P, driver MPS client limit |
| Correctness | For one molecule and one seed: the two trajectories will not be identical (different RNG), but the means and variances of temperature and potential energy should agree. That alone cannot catch wrong masses or time units (by equipartition, the temperature statistics do not depend on mass), so also: compare the vibrational density of states from the velocity autocorrelation (the C–H stretch near 3050 cm⁻¹ and the C–C modes should line up in both stacks); run a 1 ps NVE segment per stack and report the energy drift; and check padded-versus-unpadded forces in JAX |
| Node state | Run with nothing else on the node; note the driver and CUDA versions and the container hashes |

Put the results in one CSV with these columns: stack, mode (sequential, mps\_pool, batched), batched\_impl (torchsim, concat, jax), cueq (yes/no), cueq\_version, dtype, matmul\_precision, batch, mps\_workers, repeat, species\_set, wall\_s, s\_per\_mol\_ps, ns\_per\_day\_per\_traj, gpu\_util, gpu\_mem\_gb and compile\_s. Make one plot: seconds per molecule-picosecond against batch size, both stacks, all PyTorch modes, log–log.

### 4.5 The decision rule

Apply the project-wide rule below only after a batched-PyTorch candidate has passed the pilot (Section 4.3) and been measured. Then compare at batch 100 and matched precision against the *fastest validated* PyTorch mode, and state which PyTorch stack that was: ASE + MACE or TorchSim + MACE. If neither batched candidate passed, report the JAX-versus-ASE sequential and MPS-pool timings, but defer the general framework recommendation and say so in the memo. If the JAX stack is more than 5× faster per molecule, recommend JAX for the project. If it is between 2× and 5×, recommend JAX for the MD loop but keep the PyTorch model as the reference implementation. If it is below 2×, recommend the PyTorch stack and confine JAX to the differentiable forward model. Add two caveats to every outcome: the GH200's fast CPU–GPU link favours the PyTorch baseline relative to the production cluster, and the result depends on whether cuEquivariance was active in each stack; state which applied.

### 4.6 The memo (two pages)

1. Hardware, driver, container hashes, model and version.
2. Equivalence-check table and the precision settings actually in effect.
3. The timing plot and the CSV summary table, including the cuEquivariance 2 × 2 grid, the three PyTorch modes, the batched-PyTorch pilot outcome and the MPS worker count.
4. Compile-time and memory notes.
5. Recommendation with the caveats.
6. What you would do differently with another week.

## 5. T0b: the vertical slice, step by step

The slice runs the whole project pipeline on an input small enough to finish in hours and simple enough that you know what the right answer is. The "day" labels below are effort estimates, not calendar days; Section 6 gives the calendar. Every stage below becomes a script in the repository, each reading from and writing to a single HDF5 file `slice.h5`, so that the full chain can be re-run with one command (`make slice` or a Snakemake rule). All computation is on the Grace CPU; the GPU is not needed.

### 5.1 Choose 50 species (day 1)

From PAHdb v4.00 select 25 molecules in both neutral and cation charge states (50 spectra). Choose a size ladder with roughly five molecules in each of 20–30, 30–45, 45–60, 60–80 and 80–110 carbons; make about 18 compact (pericondensed, coronene family, no trio or quartet CH) and 7 irregular (with trio CH, or with bays). Note that compact PAHs below about 50 carbons are dominated by *duo* CH (coronene has only duo CH); solo CH becomes important only for the larger compact species such as circumcoronene. Exclude anything with heteroatoms, pentagons or aliphatic groups: the slice should contain nothing but size and charge variation.

Select neutral/cation *pairs* first: cations are scarce at the small end (PAHdb v4.00 has 786 neutral but only 30 cationic pure PAHs with 21–30 carbons; Ricca et al. 2025, Table 1). Save a table `species.csv` with UID, formula, Nc, charge, compactness class and the solo/duo/trio hydrogen counts. The counts are not stored as fields: derive them from the boundary-edge code, which the AmesPAHdbPythonSuite computes from the geometry (`Geometry.bec()`; Ricca et al. 2025, Sections 2.4 and 3.2), and check a few by drawing the molecule. Check that the neutral/cation pairs span the same Nc range in each compactness class, so that size and charge are not correlated through the species list.

**Why the hydrogen counts matter.** The out-of-plane C–H bands sit at 11.2, 12.0, 12.7 and 13.5 µm for solo, duo, trio and quartet H respectively (Bauschlicher et al. 2018, Section 4). The 11.2 µm band is commonly attributed to neutral solo C–H, but charge and structure both affect it. In Ricca et al.'s particular v4.00 fit of the atomic PDR, the cation contribution is concentrated in the 11.0 µm satellite (Table 3 notes). That is not a universal exclusion: Chown et al. 2024 (Section 4.3) allow appreciable cation emission at 11.2 µm. Even so, duo-only compact species are likely to have little intrinsic 11.2 µm emission, and small-size populations may have a weak 11.2 µm denominator. Do *not* fix this by selecting only solo-bearing species: that repeats the selection effect Wang 2026 (Fig. 1) warns about. Instead, report the fraction of species without solo H in each size bin, and check the denominator at population level (Section 5.4).

### 5.2 Load the spectra (day 1–2)

Use the AmesPAHdbPythonSuite to read the database and extract, for each of the 50 species, the list of band frequencies (cm⁻¹) and integrated intensities (km/mol). These are harmonic, scaled, zero-temperature values. Plot three of them as stick spectra to see what you have.

### 5.3 Emission cascade (day 2–3)

Convert each stick spectrum to an emission spectrum for a molecule that has absorbed a 6 eV photon, using the suite's cascade model (`transitions.cascade(6*1.603e-12, ...)` in its units). Apply no band redshift, following Ricca et al. 2025 (Section 6, Table 2), and convolve with a Gaussian of 15 cm⁻¹ FWHM. Ricca et al. use Gaussians of 20/15/10 cm⁻¹ by wavelength region; Bauschlicher et al. 2018 used a 15 cm⁻¹ redshift with a 15 cm⁻¹ Gaussian. A Lorentzian's wide wings would inflate the continuum that PAHFIT fits. Output: a spectrum on a common wavelength grid from 2.5 to 20 µm at R = 5,000. Keep the cascade on the fixed 6 eV photon for the slice; variable photon energy comes later. The 6 eV choice lies between the about 3 eV that Wang 2026 (Appendix B) infers for JWST MIRI spectra and the 8 eV that Ricca et al. 2025 use for the Orion atomic PDR. Record it as a known systematic for Section 5.10. Check: the 3.3 µm band should be weak for the largest molecules and strong for the smallest; the 6–9 µm bands should be far stronger in the cations than in the neutrals. If not, something is wrong in units or in the charge selection.

### 5.4 Build 2,000 populations (day 3)

Sample 2,000 points with a Latin hypercube over two parameters: mean log Nc from log 25 to log 100, with a log-normal size distribution of fixed 0.2 dex width; and ionisation fraction f from 0 to 1, the same fraction for all sizes in the slice.

**Weights.** For each population, the weight of a molecular identity is the log-normal density times that species' bin width on the discrete size ladder, renormalised after truncation to the 20–110 C range. Near the top of the range, much of a log-normal's weight falls beyond the largest species, so the renormalisation matters. Neutral and cation partners then get (1 − f) and f of that weight. f is the cation number fraction and is defined *before* any absorption weighting. Each species' emission spectrum is the cascade output per absorbed photon. State in the report whether UV absorption is taken to scale with Nc (the default here: emitted weight × Nc) or is equal per molecule. That choice changes the emitted weights, not the definition of f.

**What to store.** Sum the weighted emission spectra, and store the spectrum with:
- **two ionisation labels:** f, the cation *number* fraction, which is the regression target; and the cation share of integrated PAH *emission* in 6–15 µm, the range of Ricca et al. 2025 (Table 2);
- the realised size moments (mean and width of Nc after truncation);
- the "mass-weighted solo fraction", defined as Σ wᵢ Ncᵢ n_solo,ᵢ / Σ wᵢ Ncᵢ n_H,ᵢ. We will not regress on it yet.

**Denominator check (week 3).** Plot the population-level 11.2 µm strength relative to total PAH emission across the size × charge grid, including the pure-neutral (f = 0) and pure-cation (f = 1) endpoints. Do not rely on S/N alone: the noise in Section 5.5 is set at the 11.2 µm peak, which hides intrinsic weakness. Before any PAHFIT run, write down the domain in which 11.2-normalised ratios are valid. Choose it from the endpoint plot; for example, the 11.2 µm strength must be at least 20% of its pure-neutral value for the same size distribution. Avoid a fixed share of total emission: the 11.2 µm band is intrinsically weak in cations, so a fixed share would declare the high-f end of the label range invalid by construction. If weak denominators turn out to be widespread, make the total-PAH-normalised features (Section 5.7) primary, and report classical ratios only inside the declared domain. Never drop populations silently.

### 5.5 Instrument model (day 4)

Convolve each population spectrum to the MIRI MRS resolving power as a function of wavelength (use the per-band polynomials from PAHFIT's `jwst.yaml` instrument pack, so that the simulation and the fit use the same instrument model), resample onto the wavelength grid of the Orion MIRI templates (CDS J/A+A/685/A75), so that synthetic and real spectra share sampling and segment labels, then add a dust continuum and Gaussian noise. For the continuum, use a sum of modified blackbodies (ν² emissivity) at warm temperatures such as PAHFIT's fixed set (Smith et al. 2007, Section 4.1.2: 35–300 K), with weights chosen so that the shape resembles the Orion template continua, scaled so that the continuum under 11.2 µm is 30% of the band peak, with a random factor 0.5–2. Be aware that this continuum lies exactly in PAHFIT's own fitting basis, so continuum errors cannot appear in the main library. That is a known optimism, and the out-of-basis control in Section 5.9 tests it. Do *not* use a single cold component: a 60 K modified blackbody normalised at 11.2 µm is about 680 times larger at 20 µm (some 200× the band peak) and essentially zero below 8 µm, which would swamp 12–20 µm and wreck the fits. Add the noise at S/N 50 per pixel at the 11.2 µm peak. Restrict to 5–20 µm so that only MIRI is used in the slice (the 3.3 µm band is dropped on purpose; adding NIRSpec is a later extension).

### 5.6 Measure bands with PAHFIT (day 4–6)

Run PAHFIT, from the pinned git commit of Section 3.3, on all 2,000 spectra. Use the `classic.yaml` science pack (the Smith et al. 2007 feature set, currently PAHFIT's only full science pack) and the JWST instrument pack, and assign each wavelength range its MRS segment name from `jwst.yaml`. This is the slowest stage. Time 20 fits on day 4 before planning the full run; if a fit takes about a minute on one core, 64 of the 72 Grace cores through `joblib` finish in under an hour. Set `OMP_NUM_THREADS=1` (and the OpenBLAS/MKL equivalents) in each `joblib` worker so that 64 workers do not oversubscribe the cores. On the project node a 5–20 µm synthetic fit took 1–7 s single-threaded (`GH200_feasibility_check_2026-11-07.md` §G). **Pass `fit(..., maxiter=10000)`:** at the default `maxiter=1000`, 4 of 5 test fits stopped unconverged ("maxfev = 1000") without raising, so record `model.fitter.message` in `features.csv` and use it for the fit-converged flag. Sum complexes from `model.features` by group key; `pahfit.feature_strengths.featcombine` is legacy code for the old table format and does not apply.

**Band strengths.** PAHFIT reports individual features, so sum the 7.7 complex, the 11.3 complex (the "11.2" band) and the 12.6 complex (the "12.7" band) by their group keys in `classic.yaml`, checking the keys once. Extract the 6.2, 7.7, 8.6, 11.2 and 12.7 strengths. The classic pack has no 11.0 µm feature (only 10.68, 11.23 and 11.33 µm), so 11.0/11.2 is not available in the slice; adding it would need a custom science pack and is a later extension.

**PAHFIT returns no fit uncertainties.** Its documentation says they "will be supported … in the future"; issue #149 has been open since 2021, and PR #313 (covariance output) was closed unmerged in 2026. The parameter bounds in the output are not errors. So estimate uncertainties by **Monte Carlo refitting**, as Ricca et al. 2025 do (Appendix, "Monte Carlo Fitting Approach"):
- Perturb the spectrum with its noise, preserving the spectral covariance where it is available, and refit completely.
- Compute complex sums, ratios and formula predictions *inside* each realisation, so that the correlations between a ratio and its denominator are kept.
- Do this for all five Orion templates and for a representative subset of 100 synthetic validation spectra. Start with 30–50 realisations as a pilot, increase until the reported intervals are stable, and budget the measured cost. Elsewhere, mark uncertainties as unavailable.

**Systematic and fit quality.** PAHFIT's component strength is the full Drude integral (Smith et al. 2007, Section 4.1.4), whereas a finite window of the blended input also picks up truncation and neighbouring bands. So fix the integration conventions *before* comparing:
- For window-based checks, integrate the input PAH-only spectrum and PAHFIT's reconstructed PAH-only spectrum over the same windows, in the same spectral-density units, with the same wavelength-to-frequency Jacobian.
- Verify the extraction of component strengths separately, by injecting known PAHFIT-profile features.
- Report the cross-profile recovery bias per band as a combined *decomposition* bias (Section 5.9, control 1), not as an isolated or complete systematic.

Compute the reduced chi-square from the residuals yourself.

Store `features.csv` with one row per population: labels, band strengths, Monte Carlo uncertainties where computed, chi-square and a fit-converged flag. Define in advance what counts as a failed fit and what to do with weak or zero denominators. Inspect ten fits by eye, including the worst three by chi-square.

### 5.7 Feature table, the random-forest reference and the baselines (day 6)

Normalise strengths by the 11.2 µm strength, giving 6.2/11.2, 7.7/11.2, 8.6/11.2 and 12.7/11.2, and also by the total PAH flux. Split 70/15/15 into train, validation and test. No choice is made on the test set. All test-set numbers, for the forest, the baselines and the chosen formula, are produced in one final pass after every selection has been made on validation (end of Section 5.9).

**Freeze one feature schema before fitting anything.** Fix the wavelength interval and the component list that define "total PAH flux", the column names, and whether the primary features are 11.2-normalised or total-PAH-normalised (Section 5.4). Use the same schema for the forest, PySR and the Orion application. If total-PAH normalisation is primary, evaluate the classical-ratio baselines only inside their declared valid domain, and compare errors on identical rows. Outside that domain, judge the primary formula against the forest, the constant predictor and the absolute target; there, classical-ratio rediscovery is a restricted-domain diagnostic.

- **Reference:** fit a scikit-learn `RandomForestRegressor` (500 trees) to predict ionisation fraction from the frozen features, and record its validation RMS error now and its test RMS error in the final pass. This is the *random-forest reference*, a strong comparator rather than a proven ceiling.
- **One-ratio baselines:** fit ionisation fraction as a monotone function of the single classical ratio 6.2/11.2, and of 7.7/11.2. Pre-specify the method: isotonic regression, or a monotone spline with a fixed number of knots chosen on validation. A fitted smoother is a comparator, not an unbeatable optimum.
- **Constant predictor:** also record the RMS error of predicting the training mean.

You now have the numbers the formula is judged against. Note that this random-forest *regression* on band ratios is our own construction: Wang 2026 used a random-forest *classifier* on full binned spectra (Section 8).

### 5.8 Symbolic regression with PySR (day 7)

Configuration for the slice:

```
from pysr import PySRRegressor
model = PySRRegressor(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["log", "sqrt", "square"],
    maxsize=12,                 # complexity cap, generous for the slice
    niterations=200,
    populations=32, population_size=50,
    model_selection="best",
    constraints={"/": (-1, 4)}, # keep denominators simple
    nested_constraints={"log": {"log": 0, "sqrt": 0}, "sqrt": {"sqrt": 0}},
    random_state=0, deterministic=True, parallelism="serial"  # for the first reproducibility run
)
model.fit(X_train, y_train, variable_names=["r62", "r77", "r86", "r127"])   # unweighted; names follow the frozen schema
```

If the frozen schema of Section 5.7 makes total-PAH-normalised features primary, use those columns and their names instead. The example shows the 11.2-normalised case.

Run PySR **unweighted**. The label is exact (f comes from the Latin hypercube), and the noise is in the features, which the S/N 50 noise of Section 5.5 already models. A `1/sigma_y**2` weight would need the derivative of the very formula being searched for, and PAHFIT supplies no sigma anyway. If you add weights in a later stage, they go to `fit()`: passed to the constructor, PySR ignores them with only a warning. Run once deterministically to make sure it reproduces, then run three times with different seeds using `deterministic=False, parallelism="multiprocessing", procs=16` (16 Grace cores; `procs` has no effect in serial mode). Also run one **shuffled-label** fit, with `y_train` randomly permuted, as the negative control of Section 5.9. Save the full Pareto front (`model.equations_`) as CSV for each run.

### 5.9 The rediscovery check: a controlled closure test (day 7–8)

**What to expect.** At fixed size weights, the population spectrum is *linear* in f, so any band ratio has the form r(f) = (A₀ + f·ΔA)/(B₀ + f·ΔB) and is monotone in f wherever the denominator stays positive. Inverting it gives the exact single-ratio answer, f = c1 + c2/(c3 − r). That expression has PySR complexity 7: it is the algebraic rewrite of k·(r − a)/(b − r). The constants follow from the pure-neutral and pure-cation endpoints. When the size distribution varies, one common inverse need not exist, so expect the front to approximate it. Note that `a*r62/(b + r62)` silently assumes that neutrals emit no 6.2 µm band, so it is not the expected form.

Because monotonicity is built in, a pass by itself rules out only gross errors (swapped charges, cascade unit errors, mislabelled MRS segments). Run these controls; rediscovery counts only if all of them are run and reported:

1. **Injection–recovery.** On noise-free, continuum-free populations, compare PAHFIT's band strengths and ratios with the input, using the integration conventions fixed in Section 5.6. Record the decomposition bias per band, which includes the Gaussian-versus-Drude profile mismatch.
2. **Analytic endpoints (positive control).** Run this separately, at a few *fixed* size distributions.
   - **Endpoint strengths.** For each distribution, record the endpoint band *strengths* on a common normalisation: A₀ and A₁ (6.2 µm) and B₀ and B₁ (11.2 µm) at f = 0 and f = 1. Two endpoint ratios alone leave one constant free.
   - **The exact inverse.** f = (A₀ − r·B₀) / (r·(B₁ − B₀) − (A₁ − A₀)), which is c1 + c2/(c3 − r) with c3 = (A₁ − A₀)/(B₁ − B₀), c1 = −B₀/(B₁ − B₀) and c2 = (B₀/(B₁ − B₀))·(c3 − A₀/B₀).
   - **Edge cases.** Handle B₁ = B₀ (the inverse becomes linear in r), and check that the ratio actually varies with f.
   - **Pass condition.** Mix known intermediate fractions, run them through PAHFIT, and check that the recovered f matches. Judge this by recovering f, not by equality of fitted coefficients.
3. **Out-of-basis continuum.** Regenerate a 200-spectrum subset with a continuum PAHFIT cannot represent exactly: a power law, or the continuum shape of the atomic-PDR template. Report how much the formula's error grows.
4. **Shuffled-label negative control.** The PySR run on permuted labels (Section 5.8) must do no better than the constant predictor.
5. **Validation-only selection.** Choose the front member, the seed and the baseline settings on the validation 15%. Evaluate the locked choice *once* on the test 15%.

**Judging the chosen formula.** Look at the front up to complexity 9, but the pass target is ≤ 7 (`maxsize=12` is unchanged).

- *Single-ratio formula* (rediscovery): it must be monotone in 6.2/11.2 or 7.7/11.2 (evaluate on a grid), have test error ≤ 1.1× the one-ratio baseline of Section 5.7, beat the constant predictor, and meet an absolute target fixed before the run, for example test RMS ≤ 0.1.
- *Multi-ratio formula*: test error ≤ 2× the random-forest reference and below the one-ratio baseline.

**12.7/11.2 is a legitimate predictor.** "The 12.7 AIB complex is due to either duo H's in neutral PAHs or trio H's in cations" (Chown et al. 2024, Section 3.6; also Tielens 2008, Section 3.5). So a formula that uses r127 is not evidence of a leak. If it does, refit without r127 and report how much the error grows. The real leak risk is in the species list (Section 5.1).

Write down the formula, its complexity and its error. This is the sentence the whole slice exists to produce. A failure is a diagnostic outcome, not automatic proof that the plumbing is broken: use the controls to locate it. State in the report that the result is optimistic. Wang 2026 (Fig. 1) shows that a band ratio which works well on a hand-picked set (I₁₁.₂/I₃.₃ against size, R² = 0.82 on 81 molecules) can fail on the full library (R² = 0.23 on 15,022), and the slice uses only 25 hand-picked species.

### 5.10 Apply to the Orion Bar templates (day 8–9)

Take the five PDRs4All template spectra (atomic PDR, dissociation front 1, 2, 3 and the H II region). Run the *same* PAHFIT configuration on each, build the same normalised features, and evaluate the chosen formula inside each Monte Carlo realisation of Section 5.6, so that the uncertainties carry through.

Do *not* expect a clean ionisation gradient. Chown et al. 2024 (Section 4.3) find that the 6.2/11.2 ratio varies by only 12% across the templates and is *highest* in DF3, despite a factor of about 40 in ionising flux. They also note that the AIB emission in the H II-region template comes from the background PDR, and that DF1 is affected by foreground extinction (Peeters et al. 2024 give the extinction per template). A formula that increases with r62 will therefore put DF3 slightly *more* ionised, and the differences are close to the slice's systematics.

This is an application of the formula, not a gate. There is no reason to require the five templates to agree with each other. Chown et al. 2024 call the limited variation "somewhat surprising" and leave its interpretation to further modelling. And the observational-noise uncertainties on these spectra are tiny: Ricca et al. 2025 (Table 3) give the v4.00 cation share of the atomic PDR as 64.86 ± 0.04%, while unquantified database systematics dominate. So an agreement test would either fail on every real difference or be widened by hand until it passed. Report:

1. **Domain coverage.** For each template, whether its features lie inside the joint distribution of the training features, not just inside each feature's 1st–99th percentile. A nearest-neighbour distance or a density estimate on the training features will do. Anything outside is flagged as *extrapolation*.
2. **Validity.** The unclipped predictions, and whether each lies in [0, 1].
3. **Spread next to the error scales.** Write it as "template spread Δf = X; synthetic held-out RMSE = Y; observational Monte Carlo intervals = Z". Y is the error on synthetic data. It is not a calibrated observational resolution and not a bound on the transfer error, so do not call it one.
4. **Ricca comparison.** Ricca et al. 2025 obtain a breakdown for the atomic PDR by fitting PAHdb v4.00 spectra to the same template: 64.86% cation and 35.14% neutral (Table 3; Fig. 8 is in their Appendix). That is a share of *emission* in 6–15 µm, while the formula predicts the *number* fraction f, so the two are not directly comparable.
   - **Numerical comparison:** train and validate an auxiliary predictor of the emission label (Section 5.4) on the same splits and feature schema, apply it to the template inside the Monte Carlo realisations, and compare.
   - **Without the auxiliary predictor:** report Ricca's value as context only, mark the like-for-like comparison as unavailable, and do *not* substitute the number-fraction prediction.
   - **Their caveat:** unquantified database systematics dominate, and the values are "likely no more precise than up to the decimal point".
   - **Known differences, listed as offsets:**
     - photon energy: 8 eV versus 6 eV;
     - band profile: Gaussian 20/15/10 cm⁻¹ versus Gaussian 15 cm⁻¹;
     - molecular pool: Nc 21–384, including nitrogenated PAHs;
     - gas lines in the templates that the library lacks. Take the line list from the PDRs4All data release; the strongest near the 12.7 µm complex is [Ne II] 12.81 µm.
5. **Order.** The order of the five templates, as an observation.

Out-of-domain or unphysical values are a *transfer failure* to investigate, not automatically a broken pipeline. First check that the template continuum and extinction are handled the same way as in the library (the usual culprit), then revisit the cascade photon energy.

### 5.11 Pass criteria and the report

| Criterion | Pass |
| --- | --- |
| End-to-end run | `make slice` regenerates everything from `species.csv` without manual edits, in under four hours on the node. Cached stages are allowed; time the full chain, including the 2,000 PAHFIT fits and the PySR runs, once in week 8 |
| Reference | Random forest recovers ionisation fraction with test RMS error below 0.1 |
| Controls | Injection–recovery bias, analytic endpoints, out-of-basis continuum and shuffled labels (Section 5.9) are all run and reported; the shuffled-label run does no better than the constant predictor |
| Formula | Selected on validation, scored once on test. Some front member with complexity ≤ 7 either uses one ratio, monotone in 6.2/11.2 or 7.7/11.2, with test error ≤ 1.1× the one-ratio baseline, better than the constant predictor and within the pre-set absolute target; or uses several ratios with test error ≤ 2× the random-forest reference and below the one-ratio baseline |
| Rediscovery | The positive control recovers f for the known mixtures at fixed size distributions (Section 5.9, control 2). The chosen single-ratio formula's constants lie within the range of the endpoint constants across the sampled size distributions; report where in that range they fall and the deviation at the median size. The formula is interpretable in one sentence |
| Orion application | Domain coverage, validity in [0, 1], the spread next to the synthetic RMSE and the Monte Carlo intervals, and the Ricca et al. 2025 comparison (numerical only with the auxiliary emission-label predictor) are reported (Section 5.10); none of these is a gate on agreement between templates |
| Provenance | `slice.h5` records the PAHdb version, the suite commit, the PAHFIT commit hash, seeds and the container hash |

The report is three pages:
- the species table, with the solo-H fraction per size bin;
- one figure of three example emission spectra and one of three PAHFIT fits;
- the reference, one-ratio and constant-predictor numbers;
- the outcomes of the controls;
- the Pareto front table, with the chosen formula highlighted;
- the Orion Bar plot, with domain flags;
- a list of everything that broke and how you fixed it. The last item is the most useful part for the people who will scale this up.

## 6. Eight-week schedule and checkpoints

The two tasks are interleaved because they use different resources: T0 waits on GPU runs and weight conversion, the slice waits on PAHFIT. When one is blocked, work on the other. Each week ends with a short checkpoint meeting; the table says what to bring.

Weeks 1–2 are shared setup; T0 finishes with the memo in week 5, the slice with its report in week 8.

| Week | T0 | T0b | Show at the checkpoint |
| --- | --- | --- | --- |
| 1 | Build containers A, B, C; `uname -m`, GPU checks; one-molecule weight conversion for smoke test 3 | Download PAHdb, PDRs4All templates, MRS resolution curves | All four smoke tests passing; repository tagged `env-ok` |
| 2 | Select 20 species; export XYZ; load MACE in PyTorch at float32 | Select 50 species; `species.csv` with H-adjacency counts; load spectra; stick-spectrum plots | Species tables for both tasks; one MACE force vector printed from PyTorch |
| 3 | Convert weights for all 20; equivalence check at matched precision; build the mace-jax batched energy function | Cascade model; 2,000 populations; sanity plots (3.3 µm against size, 6–9 µm against charge); 11.2 µm denominator check and declared valid domain | Equivalence table; three emission spectra; denominator plot |
| 4 | Write `md_torch.py` (sequential, MPS pool) and `md_jax.py`; **batched-PyTorch pilot (TorchSim versus concatenated `Atoms`)**; single-molecule runs; trajectory checks (temperature, VDOS, NVE drift) | Instrument model; time 20 PAHFIT fits, then start the full run on 64 cores; noise-free injection–recovery run | One trajectory from each stack with matching statistics; pilot outcome; first ten PAHFIT fits inspected |
| 5 | Batch 1/10/100 runs, three repeats, all PyTorch modes, cuEquivariance on/off; plot; write memo | PAHFIT finished; `features.csv`; Monte Carlo refits for the validation subset | T0 memo and recommendation |
| 6 | Buffer: repeat any unstable timing; optional TF32 column (appendix to the memo) | Random-forest reference; one-ratio and constant baselines; out-of-basis continuum subset | Reference and baseline numbers |
| 7 | — | PySR runs (deterministic, three seeds, shuffled labels); Pareto fronts; closure-test controls; rediscovery check | The formula, its error, the controls and its interpretation |
| 8 | — | Orion Bar templates with Monte Carlo refits; `make slice` end-to-end timing; report | Slice report; go/no-go for T1 |

**If you fall behind.** Protect, in order:
1. the equivalence check at matched precision (without it T0 is meaningless);
2. the end-to-end `make slice` run (the point of T0b);
3. the rediscovery check with its endpoint and shuffled-label controls;
4. everything else.

Three repeats of the T0 timing can drop to two. The first things to cut are the cuEquivariance-off runs, the optional TF32 column, the TorchSim candidate (if the concatenated fallback already passed the pilot) and the three PySR seeds. **Never cut** the selected batched-PyTorch mode or the MPS pool: Section 4.5 compares against the fastest PyTorch mode measured, and without them the comparison is open to the "you did not batch PyTorch" objection.

## 7. Troubleshooting and common pitfalls

**ARM and containers**

- *"exec format error" or "no matching manifest":* the image or wheel is x86. Always pull `linux/arm64` images and check `uname -m` inside the container.
- *pip finds no wheel and tries to compile:* the package has no aarch64 build, or no build for your Python version. Check the Python version first (Section 3.3), then try conda-forge (very good aarch64 coverage) before building from source.
- *pip installs a surprisingly old release:* the package's newest version needs a newer Python than the container has (current JAX needs 3.12, mace-torch 3.11). Fix the base image rather than pinning old versions.
- *JAX silently runs on the CPU (predictions appear, no error) or logs `Unable to load cuSPARSE`:* two CUDA copies in container A (a CUDA-toolkit base plus the pip wheels), with cuEquivariance imported before JAX. Use the single-CUDA recipe (Section 3.3); `JAX_PLATFORMS=cuda` makes this an error.
- *PySR segfaults inside Julia (`signal 11`, `jl_decode_value_array`) or the build says `Could not find Python library`:* container C uses Ubuntu's Python, which means OpenSSL 3.0, Julia 1.11 and no shared libpython. Use the uv-managed Python (Section 3.3).
- *`pip check` fails in container B before you installed anything:* NGC 26.10 ships with conflicts of its own; compare against the recorded base (Section 3.3).
- *JAX does not see the GPU:* mismatch between the jaxlib CUDA build and the driver's supported CUDA version shown by `nvidia-smi`. Reinstall mace-jax with the matching extra (`[jax-cuda12]` or `[jax-cuda13]`), or use NVIDIA's JAX container for Grace Hopper.
- *MACE is slow in either stack:* check whether cuEquivariance is actually in use (mace prints the backend at model load; mace-jax selects it in its configuration). Record it in the CSV. If pip pulled cuEquivariance 0.13, you are outside both stacks' pins; reinstall 0.10.x.
- *`pip install pahfit` worked but `Model.from_yaml` or `classic.yaml` is missing:* you have the 2022 PyPI release. Install from git (Section 3.3).
- *PySR fails at run time with a read-only file system or lock-file error:* either the juliapkg project or the first Julia depot is not writable. Bind a writable copy of `/opt/julia_env` over it, and put a writable depot first in `JULIA_DEPOT_PATH` (Section 3.3, run-time Julia setup).
- *PyTorch version changed after installing mace-torch in container B:* pip replaced the NGC build. Rebuild with the constraints file.
- *Container A has a CUDA torch (`+cu13x`) instead of `+cpu`:* mace-jax was installed before the CPU torch; its `e3nn==0.4.4` dependency pulled the default wheel. Install the CPU torch first (Section 3.3).
- *`libcue_ops.so` / `libnvrtc.so.12: cannot open shared object file` when importing mace-jax or running `mace-jax-from-torch`:* the CUDA 12 libraries are not on the system path (non-`devel` base or bare venv). Use the `-cudnn-devel` base, call `jax.devices()` before importing cuEquivariance, or add `site-packages/nvidia/{cuda_nvrtc,cublas}/lib` to `LD_LIBRARY_PATH`.
- *`Weights only load failed … ScaleShiftMACE` from `mace-jax-from-torch --torch-model`:* PyTorch ≥ 2.6 default. Use `--foundation off --model-name small` (Section 3.4).
- *`CUDA_ERROR_SYSTEM_DRIVER_MISMATCH` or "forward compatibility was attempted on non supported HW" in the NGC image:* the image needs a newer driver than the node has. Use the NGC tag whose `CUDA_DRIVER_VERSION` matches the node's driver (`26.03-py3` for 595.45; Section 3.3).
- *`uv pip check` says `nvidia-cusparselt-cu13` "was built for a different platform":* harmless. The wheel carries the non-standard `manylinux2014_sbsa` tag; it loads and runs.
- *Timings vary between runs on a dual-GH200 node:* the run is not pinned to one superchip. Use `numactl --cpunodebind=0 --membind=0` with `CUDA_VISIBLE_DEVICES=0` (Section 3.2).

**MACE and mace-jax**

- *Forces disagree between frameworks:* compare cutoff radius, the neighbour-list convention (whether pairs are counted once or twice), float32 versus float64 (`mace_off` defaults to float64), TF32 (JAX uses it by default on the H100; set `JAX_DEFAULT_MATMUL_PRECISION=highest`), and whether the JAX model applies the same per-atom energy shifts. Check a single two-atom system first.
- *JAX is suspiciously fast, or the force gate fails by about 1e-3:* TF32 is on. Check the precision settings printed at start-up (Section 4.4).
- *MPS workers fail to start, or run no faster than sequential:* the MPS client limit (60 per device on current drivers, including 595.45; 48 on older ones) was exceeded, or the clients silently ran without MPS. Lower P and check the MPS server log. If the control log shows the daemon starting and nothing more, the pipe-directory path is probably too long for a Unix socket (Section 4.3).
- *Temperature looks right but the VDOS is shifted:* masses or time units are wrong in jax-md. Masses default to 1.0, and one time unit is about 10.18 fs in eV/Å/amu (Section 4.3).
- *JAX recompiles every step or every molecule:* shapes are changing. Pad every batch to fixed node and edge caps; bucket species by atom count if `N_max` would otherwise be far above most molecules.
- *JAX out of memory at batch 100:* reduce the scan chunk length (fewer steps per compiled call) before reducing the batch; the memory cost is in the trajectory buffer, not the model.
- *Energy drifts or molecule explodes:* timestep too long for the C–H stretch (0.5 fs is the maximum; use 0.25 fs if in doubt), or the thermostat friction is wrong by a factor of fs versus ps.

**PAHdb and the cascade**

- *Intensities look wrong by orders of magnitude:* units. PAHdb intensities are km/mol; the suite's cascade expects energy in erg (6 eV = 9.6 × 10⁻¹² erg). Print one spectrum's maximum before and after each step.
- *Cations look like neutrals:* the UID filter picked the wrong charge. The suite exposes charge as a field; assert on it when loading.
- *3.3 µm band does not fall with size:* the cascade was not applied (you are still looking at zero-temperature sticks).

**PAHFIT**

- *Fits fail on some spectra:* almost always the continuum. First check that the synthetic continuum is realistic (Section 5.5: no single cold component); in the out-of-basis subset, some extra failures are expected and informative. Then supply a starting value from the median of the 9.5–10.5 µm region, and clip spectra to 5–20 µm. Record the converged flag; never silently drop failures.
- *PAHFIT rejects the spectrum or ignores the resolution:* each wavelength range needs its instrument segment name from `jwst.yaml` (for example the MRS channel and band); synthetic spectra must be labelled the same way as real ones.
- *Band strengths differ from the input:* the profiles differ (PAHFIT's fixed-width Drude components versus the Gaussian used in the cascade), and overlapping wings shift the apparent continuum (Smith et al. 2007, Section 4.1.4). The bias can differ from band to band, so it is *not* automatically harmless for ratios. Measure it with the injection–recovery control (Section 5.9) and carry it as a systematic.
- *Uncertainties are missing or equal to the bounds:* PAHFIT returns no fit uncertainties; the bracketed values are parameter bounds. Use the Monte Carlo refits (Section 5.6).
- *Very slow:* run one process per spectrum with `joblib`, not threads; PAHFIT is not thread-safe in all versions.

**PySR**

- *First import takes ten minutes:* normal; juliacall downloads Julia and precompiles. Do it in the container build, not in a batch job. `pysr.install()` no longer exists; ignore old instructions that use it.
- *Results differ between runs:* expected unless `deterministic=True` with `parallelism="serial"`. Use that for the reproducibility run only.
- *A "data-dependent parameter" warning at construction:* `weights` or `variable_names` was passed to `PySRRegressor(...)` and was ignored. Pass them to `fit()`.
- *Front is full of formulas using all four variables with tiny coefficients:* the complexity cap is too high. Lower `maxsize`, add a parsimony penalty, and check the feature scaling.
- *Formula uses 12.7/11.2 for ionisation:* this is not by itself a problem; the 12.7 µm complex carries charge information (Chown et al. 2024, Section 3.6). Refit without r127 and report how much the error grows. Check for a real leak in the species list: do the neutral/cation pairs span the same Nc range in each compactness class? (Plotting f against mean Nc cannot reveal a leak, because the Latin hypercube makes the two independent by construction.)
- *Ratios have heavy tails or PySR is unstable:* weak 11.2 µm denominators. Check the declared valid domain (Section 5.4) and switch to total-PAH-normalised features if needed.

**Where to ask**

- Container and ARM problems: the HPC centre's help desk, with `uname -m`, `nvidia-smi` output and the recipe attached.
- MACE, mace-jax, jax-md, PySR: each project's GitHub discussions; search closed issues first, and include a minimal reproducer.
- PAHdb and the suite: the PAHdb team's contact on the database site; they respond to specific questions.
- Anything about which result is "right": your supervisor, at the weekly checkpoint, with a plot.

## 8. Deliverables checklist and reading list

**Deliverables**

- [ ] Repository with `containers/` (three recipes and frozen requirements), `t0/` (species list, equivalence script, `md_torch.py`, `md_jax.py`, benchmark CSV and plot), `slice/` (every stage script, `Makefile` or `Snakefile`, `slice.h5`), `reports/`
- [ ] T0 decision memo (two pages, Section 4.6 outline)
- [ ] T0b slice report (three pages, Section 5.11 outline)
- [ ] A README that lets someone else rebuild the containers and run `make slice` without asking you anything
- [ ] A one-paragraph note on what should change for the medium-scale run (500 species, 10,000 populations): what was slow, what was fragile, what you would automate

**Reading list, in order**

1. Tielens 2008, *Interstellar polycyclic aromatic hydrocarbon molecules*, Annual Review of Astronomy and Astrophysics: the standard introduction to PAHs and the bands. Read Sections 1–4.
2. Chown et al. 2024, *PDRs4All IV. An embarrassment of riches: Aromatic infrared bands in the Orion Bar*, A&A 685, A75: what the JWST data look like, how the MIRI template spectra were made, and (Section 4.3) how little the band ratios vary between them. Skim Peeters et al. 2024, *PDRs4All III*, A&A 685, A74, for the template apertures and the extinction toward each one.
3. Ricca et al. 2025, *The NASA Ames PAH IR Spectroscopic Database: Computational Version 4.00, Software Tools, Website, and Documentation*, ApJS 282, 7: the library and Python suite we use (Sections 2–3, and Section 6 for the Orion Bar fit). Bauschlicher et al. 2018 (the version 3.00 paper) for background on how the spectra are computed and scaled, and the PAHdb website "cascade model" help page.
4. Smith et al. 2007, *The mid-infrared spectrum of star-forming galaxies* (the PAHFIT paper), Sections 3–4 on the decomposition method; then the PAHFIT documentation. Argyriou et al. 2023, *JWST MIRI flight performance: The Medium-Resolution Spectrometer*, A&A, Section 2.3.2, for the MRS resolving power.
5. Batatia et al. 2022, *MACE: higher order equivariant message passing neural networks* (NeurIPS): skim the architecture, read the benchmarks; then Kovács et al. 2025, *MACE-OFF: Short-Range Transferable Machine Learning Force Fields for Organic Molecules*, JACS 147, 17598 (the model T0 runs; note its neutral-only scope); then the MACE README. Batatia et al. 2025, J. Chem. Phys. (MACE-MP-0), is the reference for the alternative foundation model.
6. Mai et al. 2025, MNRAS 541, 3073, *Computing anharmonic infrared spectra of polycyclic aromatic hydrocarbons using machine learning molecular dynamics*: the MLMD workflow the project builds on (ASE-driven MD with a NeuralIL force field in JAX, NVT then NVE); T0 benchmarks MACE-based stacks of the same kind.
7. Schoenholz and Cubuk 2020, *JAX MD: a framework for differentiable physics* (NeurIPS), and the jax-md documentation notebooks.
8. Cranmer 2023, *Interpretable machine learning for science with PySR and SymbolicRegression.jl*: Sections 1–3 and the examples; then the PySR tutorial.
9. Wang 2026, *Full-spectrum infrared fingerprinting: A transformative AI paradigm for interstellar polycyclic aromatic hydrocarbons*, A&A 710, L17: a random-forest classifier on full binned spectra that sorts PAHs into size and charge classes, and the motivation for asking how much band ratios miss; Fig. 1 shows how ratio diagnostics depend on sample selection. Read it before week 6.
10. The parent proposal, *Interpretable PAH Diagnostics by Symbolic Regression*, for how T0 and T0b feed everything that follows.
