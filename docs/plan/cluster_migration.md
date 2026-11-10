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

**On Hangang itself, do not run the fetch script.** The cluster keeps the same Hangang guide in
`/etc/user-guide`, which also includes `local.md`, KISTI's site corrections. Where they differ from the guide
pages, `local.md` wins, and the fetch script does not mirror it. The script adds only the MyKSC guide and the
myhub page. Points from `local.md` (checked 2026-11-10) that affect this work:
- Large downloads (images, models) go on the data-transfer nodes `dm01`–`dm04`. Login-node processes are
  killed after 120 CPU-minutes (`ulimit -t 7200`) and limited to 32 GB of virtual memory.
- Every array task counts as one job, and arrays with more than 5 tasks are rejected at submission.
- A single-node GPU job gets all the CPUs and memory of its GPUs' NUMA nodes: use only `--ntasks-per-node`, not
  `--ntasks`, `--cpus-per-task` or `--mem`. Any `srun` line needs `--cpus-per-task 72` (4 tasks per node)
  or `288` (1 task); without it each task is pinned to a single core.
- Caches belong on `/scratch` or `/appsdata`, not `~/.cache`. Fetch models ahead of time (Mai's Zenodo model
  to `/appsdata`) instead of downloading them in every job.
- Interactive allocations (`salloc`, `srun --pty`) are limited to 8 h.

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
apptainer cache clean --force                                   # the cache keeps a second 7.5 GB copy
```

Done 2026-11-10 on `glogin03`: the image is 7.5 GB, the pull took about a minute and the hash matched. Run the
pull on a GPU login node: the data-transfer nodes `dm01`–`dm04` have neither `module` nor Apptainer.

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

The timing probe's input, `eq_frames.xyz`, came from container B on the project node. The job now rebuilds it in
container A from `t0/species_T0.xyz` (`containers/tests/eq_frames_t0.py`: same recipe and seed, 120 frames, 6,532
atoms in the first 100). The first attempt, job 1251632, failed for lack of that file.

**Result, 2026-11-10: passed.** Job 1251720 COMPLETED in 7 min 36 s on `node0872`. On all four GPUs JAX
listed `CudaDevice`, the x64 conversion and bundles, predict (cueq off and on) and 1,000 jax-md steps ran,
so the 570.124 driver (CUDA 12.8) runs the image's CUDA 12.9 wheels. Force-call medians, MACE-OFF small, float32,
30 calls, against the project node:

| Batch | Hangang, default | Hangang, `backend: cueq` | Project node, default / cueq |
| --- | --- | --- | --- |
| 1 molecule (coronene, 36 atoms) | 1.37 ms | 1.40 ms | 1.23 / 1.26 ms |
| 10 molecules (352 atoms) | 1.96 ms | 1.93 ms | not measured |
| 100 molecules (6,532 atoms, 112,580 edges) | 9.95 ms | 9.99 ms | 9.20 / 9.19 ms |

Hangang is 8–11% slower per call (96 GB HBM3 instead of 144 GB HBM3e). Scale the project node's 0.170 s per
molecule-ps up by about 10% until the production calibration replaces it. The `conv_fusion` bundle was not timed.

## Step 3: port Mai et al.'s models (in progress; develop on the project node)

Decision, 2026-11-10: develop and debug the port on the project node, where it can run directly, and come back to
Hangang only for production. Hangang has no debug queue open to us (`intTest` is for assigned projects only), and
every Hangang test is a whole 4-GPU node.

**What Mai et al. published** (Zenodo 15771437, the latest version of 14998197; `Supplementary.zip`, 102 MB,
md5 `99277e0844c6405c6d41a732fc674d49`):
- Not MACE. Forces come from NeuralIL (`lib/bessel-nn-potentials-velo`, `NeuralILwithMorse` from `model_older`,
  a 5-member `PlainEnsemblewithMorse`, `MAX_NEIGHBORS = 25`). Dipoles come from EPNN (`lib/epnn-main`). Both are
  Flax linen; the models are `model/model_enery.pkl` and `model/model_dipole.pkl`.
- Their environment: Python 3.10, JAX 0.4.19, Flax 0.7.4. Their ASE calculator (`ase_integration_gai.py`) forces
  JAX onto the CPU when imported, so they ran MD on CPUs, one molecule at a time.
- Protocol (`code/1_MD_calc_position.py`): 0.5 fs; ASE Langevin NVT, friction 0.02, 400,000 steps; then
  VelocityVerlet NVE, 400,000 steps; positions written as text every 2 steps. `2_calc_dipole.py` runs EPNN on
  centre-of-mass-centred frames; `3_calc_IR.py` Fourier-transforms the dipole autocorrelation (first 7.987%),
  then scales frequencies by 0.9578.
- Inputs and references: 1,704 PAHdb geometries (`inputs/XYZ_pahdb_1704/XYZ_pahdb_1704.zip`; 16–252 atoms, median
  52; C, H, and N in a few) and their spectra at 50, 300 and 600 K (`outputs/IR_txt_phdb_1704/`). Production starts
  with these 1,704 species, decided 2026-11-10, not with 2,000 from PAHdb.
- Licence: code Apache 2.0; data and models CC BY-NC-SA 4.0. Keep them out of git, and check whether the
  library's open release is compatible.
- The pickles reference only `datetime`, NumPy and JAX array reconstruction, `neuralil.model.NeuralILModelInfo`
  and `epnn.model.EPNNModelInfo` (checked with `pickletools`, nothing executed). `d1/port/mai_port_test.py`
  loads them through an allow-list unpickler.

**Port test** (`d1/port/mai_port_test.py`): the authors' classes in container A (JAX 0.11.2, Flax 0.12.12), patched
only for removed APIs (`jax.tree_map`, the `jax.config` module, the forced CPU platform). It checks:
1. the two models load;
2. naphthalene energy, forces and dipole agree on GPU and CPU;
3. 1 ps of NVE conserves energy;
4. a sweep over all 1,704 geometries (finite energies and forces, at most 25 neighbours).

On Hangang, `d1/port/slurm_mai_port_test.sbatch` runs it with one shard per GPU. Its first run, job 1251891,
failed at import (`No module named 'jax.config'`). With that and the fixes below, job 1253279 passed (see below).

**Passed on the project node, 2026-11-10** (GPU 0, one shard, 900 s sweep), after two more fixes: a `jnp.clip`
shim (`a_min`/`a_max` were removed; `spherical_bessel/functions.py` uses `a_max`), and `JAX_PLATFORMS=cuda,cpu`
(the image sets `cuda` only, and check 2 needs the CPU backend; pass it as `APPTAINERENV_JAX_PLATFORMS`, because
`--env` splits on commas). Results:
1. NNFF 190,085 parameters, EPNN 410,948, both float32.
2. Naphthalene GPU vs CPU: |dE| 0, max|dF| 1.2e-7 eV/Å, max|dμ| 3.7e-8. The dipole is zero by symmetry, so this
   does not exercise EPNN much; step 2 below must use polar species.
3. 1 ps NVE: E_tot drift +1.3e-3 eV, std 6.8e-4 eV; 44 ms per step through ASE (one molecule), so production
   must be batched jax-md.
4. All 1,704 geometries in 1,316 s (`... /xyz 2400`): no problems (all elements known, at most 25 neighbours, finite
   energies and forces). Per-atom max|F| at the PAHdb DFT geometry: median 0.28, 95th percentile 1.08, max 6.68
   eV/Å. The largest are planar giants (C96H22_699 6.7, C66H18_713/714 6.7 and 6.5, C54H18_836 5.0) and non-planar
   species with sp³ CH₂ or curvature (C24H24_357 5.1 with 12 CH₂, C60H24_812, C24H16_381, C9H8_495, C101H30_804).
   Not yet known whether these reflect a level-of-theory mismatch with the training data or model error; step 2
   should include some of them when comparing with Mai's own code and published spectra.

**Passed on Hangang, 2026-11-10:** job 1253279 COMPLETED in 19 min 34 s on `node0791`, one shard per GPU, 900 s
sweep limit. The results match the project node:
- Both models load.
- Naphthalene GPU vs CPU: |dE| 0, max|dF| 2.4e-7 eV/Å, max|dμ| 5.5e-8.
- 1 ps NVE: drift +1.26e-3 eV, std 7.4e-4 eV. Through ASE a step takes 62 ms instead of 44; for one small
  molecule that is CPU-side Python overhead, which batched jax-md avoids.
- The sweep found no problems in the 1,538 of 1,704 species it reached. The per-GPU median max|F| is 0.26–0.27 eV/Å,
  and the large forces fall on the same species classes, up to 5.1 eV/Å (C24H24_357, C60H24_812, C101H30_804,
  C54H18_836).

Splitting the sweep four ways did not make it 4× faster. Every shard still recompiles for nearly every atom count,
so each GPU stopped at the time limit with about 40 species left. The 166 skipped species are the last in each
shard's alphabetical order, among them C96H22_699 and C66H18_713/714, which the project node's full sweep covered.
For production: group molecules by atom count, so compiles are counted per size group, not per molecule. This
script and these options now have a COMPLETED job on Hangang.

Everything else the port uses still exists in JAX 0.11 (`pmap`, `ops.segment_sum`, `lax.map`, `lax.cond`,
`_reconstruct_array`, Flax `FrozenDict`).

On the project node:

```bash
D=/var/tmp/$USER-pah/data/mai2025; mkdir -p $D && cd $D
curl -fL -o Supplementary.zip "https://zenodo.org/records/15771437/files/Supplementary.zip?download=1"
md5sum Supplementary.zip                    # 99277e0844c6405c6d41a732fc674d49
unzip -q Supplementary.zip 'Supplementary_Data_Code/*' -x '*/outputs/*' -d src
unzip -q src/Supplementary_Data_Code/inputs/XYZ_pahdb_1704/XYZ_pahdb_1704.zip -d .
cd ~/work/pah_stage0                        # this repository
APPTAINERENV_JAX_PLATFORMS=cuda,cpu numactl --cpunodebind=0 --membind=0 \
  apptainer exec --nv --env CUDA_VISIBLE_DEVICES=0,XLA_PYTHON_CLIENT_PREALLOCATE=false \
  --bind $D/src/Supplementary_Data_Code:/mai:ro --bind $D/XYZ_pahdb_1704:/xyz:ro --bind $PWD/d1/port:/w:ro \
  /var/tmp/$USER-pah/sif/A_jax.sif python -I /w/mai_port_test.py /mai /xyz 900
```

Still to do on the project node, in order:
1. ~~Get `mai_port_test.py` passing.~~ Done (above).
2. Validate against Mai's own code. Compute energies, forces and dipoles for a few species in their pinned
   environment on CPU (JAX 0.4.19, from their `environment.yaml`) and compare them with the port. Then run one
   species end to end and compare its spectrum with their published 300 K spectrum.
3. Write `d1/mlmd_run.py`. Batch molecules in jax-md (group by atom count), with their protocol, unit conversions
   checked (ASE friction and time units). Compute the EPNN dipole on the GPU during NVE instead of writing
   trajectories, and checkpoint so jobs can resume.
4. Calibrate timing at batch 100. Then, back on Hangang: `git pull` in `/appsdata/$USER/pah_stage0`, and submit one
   short calibration job (a new job, so ask first) before production.

On Hangang the inputs are already in place: `/appsdata/t641a02/d1/mai2025/` (zip, `src/` with code, models and
inputs) and `/appsdata/t641a02/d1/XYZ_pahdb_1704/`.

## Step 4 onwards (not yet written)

- `d1/mlmd_run.py`: the production driver. It runs the Mai et al. 2025 force field and dipole model (Zenodo
  10.5281/zenodo.14998197; neutral PAHs only) in batched jax-md, NVT then NVE, saving dipole time series and
  checkpoints. Group batches by atom count, because jax-md pads every molecule to the largest in the batch.
- The species input: Mai et al.'s 1,704 PAHdb geometries first (see Step 3); 2,000 neutral PAHdb v4.00 species later.
- A timing calibration at batch 100 on Hangang before production. The 0.170 s per molecule-ps figure is from the
  project node with MACE-OFF small, not from Mai's model.
- Job layout: one job per node, four processes (one per GPU), checkpointing so that a job can stop at the 24 h limit
  and resume; at most 5 jobs running.

Estimated production: about 200 GPU-hours as specified (200 ps NVT + 200 ps NVE, 2,000 species × 4 temperatures),
or about 110 with a 20 ps Langevin equilibration. On one 4-GPU node that is roughly 1–2 days (2–3 jobs of 24 h);
with 5 nodes in parallel, well under a day.
