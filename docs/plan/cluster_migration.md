# Moving the MLMD (D1) GPU work to KISTI Hangang

Written 2026-11-10 on the project GH200 (`gh200`), for whoever (person or Claude) picks the work up on Hangang.
`CLAUDE.md` describes the project node; this file records what differs on Hangang and what to do first.

KISTI's documentation can be mirrored as Markdown under `docs/external/` by running
`docs/external/fetch_ksc_docs.sh` (the mirror is not in git; the facts below come from the 2026-11-10 fetch):
- `hangang-user-guide/`: the Hangang user guide (system, storage, Slurm queues, containers, software). Start here.
- `myksc-hangang/`: the MyKSC web portal (web terminal, file browser, web Slurm form, Jupyter/VS Code apps).
- `neuron-user-guide/pages/appendix/appendix-12-how-to-use-containers.md`: the myhub (Harbor) section the
  MyKSC guide links to.

Each directory has `llms-full.txt` (the whole guide in one file) and `pages/` (one Markdown file per page).

## Hangang facts

From the guide (`hangang-user-guide/pages/system/*.md`) and from our own checks on 2026-11-10.

| Item | Value | Consequence for us |
| --- | --- | --- |
| GPU nodes | 4 GH200 per node (Grace 72 cores + Hopper, 120 GB LPDDR5X + 96 GB HBM3 each); 288 Slurm cores and about 850 GiB per node | Same aarch64 image as the project node; a 96 GB GPU is ample (batch 100 used 2–4 GB) |
| CPU nodes | AMD Turin, **x86-64** (`normal_cpu`, `long_cpu`, `hugemem`) | Container C must be the x86 build (`containers/C_cpu_x86.def`) there |
| Driver | 570.124.06 (CUDA 12.8), measured | Container A's CUDA 12.9 pip wheels rely on minor-version compatibility: confirm with the smoke test |
| GPU queues | `normal` (beta users), `natStr`, `5thEx`; **24 h wall time; at most 5 active and 5 running jobs per user** | Production must checkpoint and resubmit; at most 5 nodes (20 GPUs) at once |
| Allocation | Whole nodes only (`--exclusive`) | One job = one node = 4 GPUs: run 4 processes, one per GPU |
| Mandatory options | `--time`, and `--comment="field=...;appl=..."` (values from `showappl`) | We use `field=aac;appl=jax` (astrophysics; JAX) |
| Submit directory | Jobs can be submitted **only from `/scratch/$USER` or `/appsdata/$USER`** | Keep the clone, images and results under `/appsdata/$USER` |
| `/home01/$USER` | 64 GB, 500k files, backed up | Configuration and caches only |
| `/scratch/$USER` | 80 TB per group, 5M files, flash+HDD; **files unread for 15 days are deleted** | Per-job working files and checkpoints during a run |
| `/appsdata/$USER` | 20 TB per group, 5M files, no purge, no backup | Images, inputs and results to keep; copy final results off the system too |
| `/tmp` on GPU nodes | tmpfs (RAM), about 200 GB free, measured | Not for build trees or results |
| Apptainer | `module load apptainer/1.4.5` (default); `--nv` required; `--fakeroot` offered | Pull images; build only if needed. KISTI recommends Enroot for its own images; Apptainer is supported for `.sif` |
| Internet | Reachable from compute nodes, measured (zenodo.org) | Models download inside jobs |

Earlier `sinfo` output showed nodes or partitions named `exclusive01`–`09` with no time limit. The guide lists only
the queues above. Check `sinfo -s` and use the queue your account is entitled to.

## What runs here and what does not

Only container A (JAX + mace-jax + jax-md) is needed for MLMD. Container B (PyTorch baseline, MPS) stays on the
project node. Container C (PySR, PAHFIT) can run on Hangang CPU nodes only as the x86 build. Nothing in `CLAUDE.md`
about the MPS daemon, forward-compatible libcuda or the host driver 595.45 applies here.

## Step 1: get container A

Pull the exact T0 image from myhub (KISTI's Harbor). It was pushed on 2026-11-10 as an ORAS artifact, and its layer
digest is `sha256:7304b1becb903c165694c9cc2233335621ed65e39e73f8a2cc724a867a75adc2`, the same file the T0 runs
used. The project `pah` is public (anonymous manifest and blob reads checked), so no login is needed. Use
`apptainer pull … oras://`, not `podman` or `docker://`: the artifact is a SIF, not a Docker image.

```bash
module load apptainer/1.4.5
mkdir -p /appsdata/$USER/sif
export APPTAINER_CACHEDIR=/appsdata/$USER/apptainer-cache     # not /tmp (RAM) and not the 64 GB home
apptainer pull /appsdata/$USER/sif/A_jax.sif oras://myhub.ksc.re.kr/pah/a_jax:gh200-2026-11-07
sha256sum /appsdata/$USER/sif/A_jax.sif                         # must be 7304b1becb903c16...
```

Pushing needs credentials: `HARBOR_USER` and `HARBOR_SECRET` in `.env` (not in git; the CLI secret from the
myhub user profile), used as
`printf '%s' "$HARBOR_SECRET" | apptainer registry login --username "$HARBOR_USER" --password-stdin oras://myhub.ksc.re.kr`.
They can push and pull, but deleting a repository returned 401. A 20 MB test artifact, `pah/oras-test:t1`, is left
over from the first push test; remove it in the myhub web UI.

Fallback, if the pull ever fails: build `containers/A_jax_locked.def` on a GPU login or compute node (aarch64). It
constrains every install to `containers/requirements-A_jax-container.txt`, the freeze of the GH200 image, and fails
if the result differs. Put `APPTAINER_TMPDIR` on `/scratch/$USER`, not `/tmp`.

## Step 2: smoke test on all four GPUs

```bash
cd /appsdata/$USER && git clone git@github.com:leejjoon/pah_stage0.git && cd pah_stage0 && git checkout gh200-containers
sbatch containers/tests/slurm_a_smoke.sbatch /appsdata/$USER/sif/A_jax.sif
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
- A timing calibration at batch 100 on Hangang before production. The 0.170 s per molecule-ps figure is from the
  project node with MACE-OFF small, not from Mai's model.
- Job layout: one job per node, four processes (one per GPU), checkpointing so that a job can stop at the 24 h limit
  and resume; at most 5 jobs running.

Estimated production: about 200 GPU-hours as specified (200 ps NVT + 200 ps NVE, 2,000 species × 4 temperatures),
or about 110 with a 20 ps Langevin equilibration. On one 4-GPU node that is roughly 1–2 days (2–3 jobs of 24 h);
with 5 nodes in parallel, well under a day.
