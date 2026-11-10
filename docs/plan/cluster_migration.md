# Moving the MLMD (D1) GPU work to the Slurm GH200 cluster

Written 2026-11-10 on the project GH200 (`gh200`), for whoever (person or Claude) picks the work up on the
cluster. `CLAUDE.md` describes the project node; this file records what differs on the cluster and what to do
first.

## Cluster facts (checked 2026-11-10)

| Item | Value | Consequence |
| --- | --- | --- |
| Architecture | aarch64, NVIDIA GH200 120GB (96 GB HBM visible) | Same ARM images as the project node; no x86 rebuild |
| Driver | 570.124.06 (CUDA 12.8) | Container A carries CUDA 12.9 pip wheels: relies on CUDA minor-version compatibility, untested until the smoke test passes |
| Apptainer | `module load apptainer` gives 1.4.5; `--fakeroot` is offered | Load the module in every job; images can be built here |
| `$TMPDIR` | `/tmp`, tmpfs (RAM), about 200 GB free | Small working files and the JAX compile cache only; results go to persistent storage |
| Nodes | partitions or nodes `exclusive01`–`09` (to confirm), 4 GH200 per node, no time limit | One node = 4 GPUs; checkpoint for safety, not for wall-time limits |
| Internet | Available from compute nodes | Zenodo models and the MACE-OFF checkpoint download directly |

Not yet known: the persistent (non-tmpfs) storage path, and whether `exclusive01`–`09` are partition or node names.

## What runs here and what does not

Only container A (JAX + mace-jax + jax-md) is needed for MLMD. Containers B (PyTorch baseline, MPS) and C (PySR,
PAHFIT) stay on the project node or on CPU machines. Nothing in `CLAUDE.md` about the MPS daemon, forward-compatible
libcuda or the host driver 595.45 applies here.

## Step 1: build container A, locked to the GH200 versions

`containers/A_jax.def` pins only the mace-jax commit, so a fresh build pulls newer JAX, cuEquivariance and CUDA
wheels than T0 validated. Build `containers/A_jax_locked.def` instead: it constrains every install to
`containers/requirements-A_jax-container.txt` (the freeze of the GH200 image) and fails if the result differs.

```bash
# on a compute node, from the repository root
module load apptainer
export APPTAINER_TMPDIR=/tmp/$USER-abuild APPTAINER_CACHEDIR=/tmp/$USER-acache   # tmpfs: ~20 GB of RAM while building
mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR
apptainer build --fakeroot <persistent-dir>/sif/A_jax_locked.sif containers/A_jax_locked.def
sha256sum <persistent-dir>/sif/A_jax_locked.sif
```

**Preferred: pull the exact T0 image from Harbor instead of building.** On 2026-11-10 the GH200 image was pushed to
the KSC Harbor registry as an ORAS artifact: `oras://myhub.ksc.re.kr/pah/a_jax:gh200-2026-11-07`. Its layer digest
is `sha256:7304b1becb903c165694c9cc2233335621ed65e39e73f8a2cc724a867a75adc2`, the same file the T0 runs used. The
project `pah` is public (since 2026-11-10; anonymous manifest and blob reads checked), so pulling needs no login.
Use `apptainer pull`, not podman: the artifact is a SIF, not a Docker image.

```bash
module load apptainer
export APPTAINER_CACHEDIR=<persistent-dir>/apptainer-cache       # not /tmp (RAM) and not a small home
apptainer pull <persistent-dir>/sif/A_jax.sif oras://myhub.ksc.re.kr/pah/a_jax:gh200-2026-11-07
sha256sum <persistent-dir>/sif/A_jax.sif                          # must be 7304b1becb903c16...
```

Pushing still needs credentials: `HARBOR_USER` and `HARBOR_SECRET` in `.env` (not in git), used as
`printf '%s' "$HARBOR_SECRET" | apptainer registry login --username "$HARBOR_USER" --password-stdin oras://myhub.ksc.re.kr`.
They can push and pull, but deleting a repository returned 401. A 20 MB test artifact, `pah/oras-test:t1`, is left
over from the first push test; remove it in the Harbor web UI.

## Step 2: smoke test on all four GPUs

```bash
sbatch -p <partition> containers/tests/slurm_a_smoke.sbatch <persistent-dir>/sif/A_jax_locked.sif
```

Pass criteria, from the log `a_smoke_<jobid>.log`:
- `jax.devices()` lists `CudaDevice` entries on every GPU, not `CpuDevice`;
- the x64 conversion check (`x64_verify.py`) passes;
- predict and the jax-md Langevin run finish;
- force-call medians are close to the project node's (feasibility check §J; `reports/t0_timing_summary.md`).

## Step 3 onwards (not yet written)

- `d1/mlmd_run.py`: the production driver. It runs the Mai et al. 2025 force field and dipole model (Zenodo
  10.5281/zenodo.14998197; neutral PAHs only) in batched jax-md, NVT then NVE, saving dipole time series and
  checkpoints. Group batches by atom count, because jax-md pads every molecule to the largest in the batch.
- The species input: 2,000 neutral PAHdb v4.00 species, built on the CPU side.
- A timing calibration at batch 100 on this cluster before production. The 0.170 s per molecule-ps figure is from
  the project node with MACE-OFF small, not from Mai's model.

Estimated production: about 200 GPU-hours as specified (200 ps NVT + 200 ps NVE, 2,000 species × 4 temperatures),
or about 110 with a 20 ps Langevin equilibration. On one 4-GPU node that is roughly 1–2 days.
