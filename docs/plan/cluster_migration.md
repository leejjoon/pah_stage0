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
   species end to end and compare its spectrum with their published 300 K spectrum. **Done, 2026-11-11; the
   intensity gap in 2b is accepted as a caveat (see below).**
3. **Done, 2026-11-11 (see "Step 3 driver" below).** Write `d1/mlmd_run.py`. Batch molecules in jax-md (group by atom count), with their protocol, unit conversions
   checked (ASE friction and time units). Compute the EPNN dipole on the GPU during NVE instead of writing
   trajectories, and checkpoint so jobs can resume. Run several replicas (seeds) per species and average their
   spectra: one trajectory leaves small-PAH band ratios uncertain by tens of percent (step 2b), and replicas
   batch almost for free. Pad the EPNN graphs to a fixed edge count, as `mai_spectrum.py` does.
4. **Done on the project node, 2026-11-11 (see "Step 4 calibration" below).** Calibrate timing at batch 100. Then,
   back on Hangang: `git pull` in `/appsdata/$USER/pah_stage0`, and submit one short calibration job (a new job, so
   ask first) before production.

**Step 2 so far (2026-11-10, project node).** `d1/port/mai_port.py` now holds the port (shims, models, dipole);
`mai_port_test.py` uses it.
- *2a, static (passed).* Reference environment: `/var/tmp/$USER-pah/venv/mai_ref`, Python 3.10 by uv, with
  their pins (jax/jaxlib 0.4.19, flax 0.7.4, numpy 1.26.1, scipy 1.11.3, ase 3.23.0, jraph 0.0.6.dev0; aarch64
  wheels exist and run on the 64 KiB-page kernel). `mai_ref_eval.py` runs their code there unpatched (their
  `read_coord` executed from `2_calc_dipole.py`). `mai_compare.py prep|port|compare` checks 8 species (naphthalene,
  two N-containing, indene, C24H24_357, C96H22_699, C66H18_713, the 252-atom C216H36_615), each at the DFT geometry
  and rattled by 0.05 Å. Port on GPU and on CPU vs reference: max|dF| 1.9e-5 eV/Å, max|dq| 4.7e-6 e,
  |dμ| ≤ 3.5e-5 e·Å; energies within one float32 ulp. The reference gives the same large forces (6.68 eV/Å for
  C96H22_699), so those come from the model, not the port.
- *2b, end to end (band positions agree; CH-stretch and in-plane intensities do not).* `mai_spectrum.py`: jitted
  single-molecule loop, R replicas vmapped; NVT is BAOAB with γ = 0.02 per ASE time unit (1.96 ps⁻¹) and the
  centre-of-mass velocity removed, not ASE's Langevin algorithm; NVE is velocity Verlet, checked against ASE (200
  steps: |dr| 2e-6 Å). Dipoles are batched (500 frames, graphs re-padded to a multiple of 4,096 edges, since the
  EPNN builder pads each batch to its own edge count and every batch recompiled) and checked against the per-frame
  path at the start, middle and end of NVE (2e-5 e·Å). Their `3_calc_IR.py` runs unmodified on the dipoles;
  `mai_spectrum_compare.py` compares band windows with the published spectrum and ranks it among the replicas.
  Naphthalene, 300 K, 32 replicas (200 ps NVT + 200 ps NVE each; MD 41 min, dipoles 44 min on one GH200):
  - NVE ⟨T⟩ 289 ± 41 K over replicas (212–386 K; one canonical snapshot of 18 atoms starts each NVE run), and
    300 × 51/54 = 283 K is expected with 3N − 3 degrees of freedom and their 1.5 N k_B convention; E_tot drift ≤
    3e-3 eV over 200 ps.
  - Band positions: the replica-mean centroids of the strong bands (476, 774 cm⁻¹ and the CH stretch) are within
    1.7 cm⁻¹ of the published ones, so time units, timestep and the 0.9578 scaling are right.
  - Intensities: the published shares sit inside the replica range in 8 of 10 windows, but at rank 0/32 for the CH
    stretch (published 0.142, replicas 0.157–0.376, median 0.230) and for 1480–1560 cm⁻¹, and 2–3/32 for 985–1290
    cm⁻¹. With the 740–820 window left out of the denominator, the 476 cm⁻¹ share is mid-range (19/32), but every
    in-plane window from 985 to 1640 cm⁻¹ and the CH stretch rank 0–4/32, at 0.5–0.8 of the replica median: a
    step, not a slope, so not a frequency-dependent processing error. The published trajectory carried less
    in-plane and CH energy than a typical draw from the released protocol. Not temperature either: across replicas
    the CH share falls slightly with NVE ⟨T⟩ (r = −0.36).
  - Mai et al. 2025 §2.4 says the NVT run used a **Nosé–Hoover** thermostat; the released
    `1_MD_calc_position.py` uses ASE Langevin (friction 0.02). Which one made the published spectra is unknown, and
    a deterministic thermostat on 18 atoms may leave modes unequally populated. ASE 3.23.0 (their pin) has no
    Nosé–Hoover chain module; only `npt.py` with `pfactor=None`, which needs a cell. The §2.4 wording also mirrors
    §2.1 (the training-data MD), so it may be a description error. Not tested; parameters would be guesses.
  - For step 3: with an identical protocol, one trajectory per species leaves small-PAH band ratios uncertain by
    tens of percent (774 share 0.22–0.59, CH stretch 0.16–0.38 over 32 seeds). Mai et al. ran one per species.
    MD cost barely rises with replicas (5.3 s per ps at R = 3, 6.2 s at R = 32, for all replicas together).
  - Decision, 2026-11-11: accept the gap as a caveat and move on. The port reproduces the released code; the
    published spectra are not a gate. Production averages several replicas per species (step 3). Not asked of
    the authors; not tested across species.
- The published `outputs/IR_txt/C10H8_330_qm0.txt` is not the 300 K spectrum (its bands differ); use
  `outputs/IR_txt_phdb_1704/IR_txt_300K_qm0.zip`.

**Step 3 driver (2026-11-11).** `d1/mlmd_run.py plan|run|status` and `d1/mlmd_spectra.py`.
- *plan* groups systems (species × T × replica, each with a fixed seed) by atom count into work units of at most
  `--batch` systems, sorted by cost, and fixes the settings in `plan.json`. Defaults: 50, 300, 600 K; 4 replicas;
  batch 100; 400,000 NVT + 400,000 NVE steps. Mai's 1,704 species give 20,448 systems in 269 units (12–100 systems,
  106 atom counts).
- *run* takes the units with index % nshards == shard (one process per GPU) and runs each as one vmapped batch:
  the step 2b loop (BAOAB NVT at their friction, velocity Verlet NVE), not jax-md, because that loop is already
  checked against ASE; molecules are not padded, so a unit compiles once per shape (persistent cache in
  `<out>/jax_cache`). The EPNN dipole is computed in the loop every 1 fs by a traceable copy of the authors' graph
  builder (same edges, 0 < d < 3.5 Å in the same order, one jraph padding graph; edge cap 1.5× the largest initial
  count, overflow flagged on every frame). Each unit checks the in-loop EPNN charges against the authors' per-frame
  path, for one system per species (up to 8), on the initial geometries and again on the thermal frames at the end
  of NVT; the gate is 1e-4 e per atom (charges, because dipole noise grows with size; tests gave ≤ 4.3e-6 e, the
  252-atom C216H36_615 included). NVT noise is keyed by (system seed, chunk), so a trajectory does not depend on its
  batch. Checkpoints (`state.npz` and `meta.json` written atomically, every 10 min and at the end) and a `--hours` limit let a job stop and resume.
  Output per unit: `dipole.npy` (frames, systems, 3), `meta.json` with QA flags (non-finite, T > 30,000 K as the
  authors' check, more than 25 NNFF neighbours, edge overflow), `state.npz`, `DONE`.
- *mlmd_spectra.py* is the authors' `calc_MD_IR` as a function (reproduces their `3_calc_IR.py` output exactly);
  it averages replicas in σ (cm²) before normalising and skips flagged systems.
- Tests on the project node (`/var/tmp/$USER-pah/d1/step3_test`): the same systems batched by 10 and by 2 give
  dipoles within ~1e-5 e·Å over 2 ps NVE; stopping after 3 chunks and resuming differs from an uninterrupted run
  only as much as two uninterrupted runs differ (GPU scatter-add order is not deterministic; up to 1e-3 e·Å
  after 2 ps). Indene (C9H8_495, 4.8 eV/Å at its PAHdb geometry) heats to ~900 K without NVT and is then chaotic
  (e-folding ~25 fs); the 200 ps NVT drains that in production. Mixed batch (50, 300, 600 K together; 17, 18 and
  252 atoms; 10 ps NVT + 2 ps NVE): each system's NVT temperature follows its own target (second-half means 47–51,
  267–296, 580–593 K against 47–50, 282–299, 565–598 expected from 3N − 3 degrees of freedom; few systems, so
  ±5 %); no flags; NVE E_tot spread ≤ 3e-2 eV (two float32 ulps at −1.8e5 eV for 252 atoms). `mlmd_spectra.py`
  ran on it (21 species × T, replicas averaged).
- Throughput: see step 4 below.

**Step 4 calibration (2026-11-11, project node, GPU 0).** Hangang calibration job: see `docs/plan/hangang_step4_handoff.md`. Six units of the full plan, two NVT and two NVE chunks
each (2 ps; `/var/tmp/$USER-pah/d1/step4_calib` and `step4_vmap`); `meta.json` now records `chunk_seconds`, and
`run --units` selects units (still split across shards).
- With the authors' descriptor code, wall time per ps fitted t = n (0.30 + 0.0030 B) s for NVT and
  n (0.34 + 0.0042 B) for NVE (n atoms, B systems; residuals < 1 s). The large term independent of B is
  `PowerSpectrumGenerator.process_data`: it loops over atom centres with `jax.lax.map` (`bessel_descriptors.py`
  l. 484), so each step runs n sequential per-centre kernels. Production at that speed: ~1,000 GPU-h.
- The driver now patches `process_data` to `jax.vmap` the same per-centre function (`_patch_descriptors` in
  `mlmd_run.py`), for the NNFF and the EPNN descriptors. Under vmap, `jnp.repeat` by `self._degeneracies` (array
  repeats) built B × n × 25 × 25 × … index constants that XLA constant-folded on the CPU for > 10 min per unit, so
  `jnp.repeat` with concrete array repeats is replaced by a gather with NumPy-built indices (bit-identical).
  The per-unit self-check now also compares forces from the vmapped descriptor with the authors' `lax.map` path
  (traced with the patch switched off; gate 1e-3 eV/Å) besides the charges; the reference paths keep a length-n
  scan in their jaxpr, the engine's has none.

  | unit (n atoms, B) | NVT s/ps, lax.map → vmap | NVE s/ps | max dq (e) | max dF (eV/Å) |
  | --- | --- | --- | --- | --- |
  | n016 (16, 12) | 4.8 → 0.8 | 6.1 → 1.6 | 4.5e-6 | 5.7e-6 |
  | n044 (44, 100) | 26.1 → 8.8 | 33.4 → 15.0 | 5.5e-6 | 8.6e-6 |
  | n086 (86, 99) | 51.0 → 18.1 | 64.9 → 29.5 | 5.4e-6 | 1.3e-5 |
  | n126 (126, 92) | 72.5 → 25.6 | 91.3 → 41.4 | 5.9e-6 | 1.1e-5 |
  | n180 (180, 84) | 98.9 → 33.9 | 125.8 → 54.6 | 5.1e-6 | 1.2e-5 |
  | n252 (252, 12) | 83.7 → 6.6 | 99.0 → 11.2 | 4.1e-6 | 1.3e-5 |

  Now t ≈ n B × 0.0022 s/ps (NVT) and n B × 0.0035 (NVE; the 1 fs dipoles cost ~60 % on top): proportional to
  the work, so batch size no longer matters for efficiency. Compiles, self-checks and setup add 60–170 s per unit
  (whole unit wall time minus its steady chunks), ~10 GPU-h over 269 units. No flags,
  no edge overflow.
- Peak GPU memory, most expensive unit (n180_u000, 84 × 180 atoms, vmapped): 28 GB (`nvidia-smi`, no
  preallocation); Hangang's GPUs have 96 GB.
- Full-length check of the final driver: the naphthalene unit of the production plan (n018_u000: 12 systems, 50/300/
  600 K × 4 replicas, 200 ps NVT + 200 ps NVE; `/var/tmp/$USER-pah/d1/step4_full`) in 9.8 min. No flags or edge
  overflow; self-check dq ≤ 2.3e-6 e, dF ≤ 2.9e-6 eV/Å; NVE ⟨T⟩ 42–50, 268–336, 529–632 K; E_tot spread over
  200 ps ≤ 0.016 eV. Against the published 300 K spectrum (`mai_spectrum_compare.py`, 4 replicas): strong-band
  centroids within 2.5 cm⁻¹; CH-stretch share 0.18–0.24 (median 0.22) and in-plane windows above the published
  ones, as in step 2b (32 replicas: CH median 0.23). So the vmapped driver reproduces the step 2b loop's spectra.
- **Estimate for the full plan** (1,704 species × 50/300/600 K × 4 replicas, 269 units): **~420 GPU-h** on this GPU
  as specified (200 ps NVT + 200 ps NVE), ~270 with 20 ps NVT. On Hangang (4 GPUs per node) that is ~105
  node-hours, e.g. a 5-task array of 24 h; the Hangang calibration job checks the speed there.
- `d1/slurm_mlmd_run.sbatch A_jax.sif D1_DIR OUT [run options]` runs one process per GPU and sets `--hours` from
  the job's time limit (minus 30 min, or 15 % of a short limit); resubmitting continues. Several nodes on one OUT
  must be one Slurm array (`--array=0-4`: task k runs shards k·4+g of 4K); never two separate jobs on one OUT, and
  resubmit with the same array size. Calibration
  on Hangang: a plan with `--nvt 4000 --nve 4000` in a fresh OUT, then
  `sbatch --time=00:30:00 d1/slurm_mlmd_run.sbatch ... --units n180_u000,n126_u000,n086_u000,n044_u000`.

On Hangang the inputs are already in place: `/appsdata/t641a02/d1/mai2025/` (zip, `src/` with code, models and
inputs) and `/appsdata/t641a02/d1/XYZ_pahdb_1704/`.

### Chemical integrity with temperature (hot test and cold check, 2026-11-11)

Mai et al. 2025 (§4) say their model is not reliable above ~600 K. `d1/mlmd_integrity.py <run> <xyz dir>` compares
each system's bond graph at the end of a run (distance < 1.2 × sum of covalent radii) with the input geometry; the
driver's own QA flags catch only part of this (25 of the hot test's changed systems).
- **Hot test** (`/var/tmp/$USER-pah/d1/hot_test`): 33 species (9–216 C, incl. N-bearing, odd-electron and
  hydrogenated ones) × 600/900/1200 K × 4 replicas, 20 ps NVT + 50 ps NVE. Intact: 105/132 at 600 K, 89/132 at
  900 K, 43/132 at 1200 K. At 1200 K most failures are H loss (41; 37 even with a 1.35 factor) and C–C changes (37).
  Naphthalene stays intact at all three.
- **Cold check** (`/var/tmp/$USER-pah/d1/cold_check`): the 9 species that changed in the hot test, at 50 and 300 K
  (4 replicas, 20 + 50 ps). Corrected 2026-11-12 from the raw end states (`integrity_full.txt` in both run
  directories; the first version of this note said "six species at 50 K", which mixed the hot-test list with the
  cold check). Replicas (of 4) whose end-state bond graph differs from the input:

  | Species | 50 K | 300 K | 600 K | 900 K | 1200 K | What changes |
  | --- | --- | --- | --- | --- | --- | --- |
  | C9H8_495 | 4 | 4 | 4 | 4 | 4 | Ring opens (C–C 1.51 → 4.5 Å at 50 K) |
  | C24H24_357 | 4 | 4 | 4 | 4 | 4 | Several C–H past 1.28 Å (1.29 at 50 K) |
  | C54H18_836 | 4 | 4 | 4 | 4 | 4 | One C–H 1.075 → 1.32 Å |
  | C60H24_812 | 4 | 4 | 4 | 4 | 4 | H atoms move between carbons |
  | C15H10_341 | 0 | 4 | 4 | 4 | 4 | C–C and C–H changes |
  | C29H17_2133 | 0 | 0 | 4 | 4 | 4 | |
  | C10H9N_473 | 0 | 0 | 1 | 4 | 4 | |
  | C27H15_2237 | 0 | 0 | 1 | 4 | 4 | |
  | C47H18_83 | 0 | 0 | 1 | 4 | 4 | |

  So 4 species fail at every temperature (16/36 at 50 K), C15H10_341 from 300 K (20/36), C29H17_2133 from 600 K, and
  three species from 600–900 K. Of the other 24 hot-test species, 23 have at least one changed replica at 1200 K
  (4 at 900 K, none at 600 K); naphthalene (C10H8_330) is the only species intact at all temperatures. This is an
  end-state check: it does not separate a persistent change from a bond caught mid-stretch. At 50 K a 1.29–1.32 Å
  C–H bond is far beyond thermal amplitude, so those are real distortions; at 900–1200 K a persistence check
  (several frames) is needed before calling a single C–H at ~1.3 Å a failure.
- **Low-temperature sweep of all 1,704 published species** (`/var/tmp/$USER-pah/d1/sweep`, 2026-11-12): 50 and 300 K,
  one replica, 20 ps NVT + 30 ps NVE, 124 units, ~14 h on GPU 0. The model changes the bond graph of **62 species
  (3.6 %)**: 17 at 50 K (15 of them also at 300 K, 2 only at 50 K) and 45 more only at 300 K. At 50 K: 4 with C–C
  changes, 7 losing a C–H bond, 4 gaining one, 2 with H moving. At 300 K (60 systems): 30 losing a C–H bond, 14 with
  H moving, 8 gaining a C–H bond, 8 with C–C changes. 20 of the 62 have an odd number of H (radicals); 1 contains N.
  At 300 K a C–H bond past the 1.28 Å cut is ~6–7 classical thermal standard deviations from equilibrium, so these
  are distortions, not a bond caught mid-stretch. List: `docs/plan/mai_model_lowT_failures.csv`. These species are
  in Mai et al.'s published 50–600 K spectra; exclude them, or check their published spectra, before use.
- Consequences: 1200 K is not usable with this model; 900 K only with the integrity filter applied per system; and
  production at any temperature needs the integrity check, with species that fail at 50 K excluded (they are in
  Mai's published 50–600 K spectra too, so check those before use).

## Seed-set DFT for the charged MLFF (GPU4PySCF, container D; prepared 2026-11-14)

Handoff for the Hangang session (context, expected results, what to report): `docs/plan/hangang_seed_dft_handoff.md`.

The B3LYP/6-31G\* labels for the 7,504 seed frames (`docs/plan/charged_mlff_plan.md` Phase 1; engine and validation in
`docs/research/pyscf_gpu_jax.md` §5) are about 87 H100-hours. On the project node that is about 33 h on both GPUs;
here, 5 nodes (20 GPUs) should take roughly 4–5 h of wall time plus queue wait. Nothing about MPS applies (none is used).

**1. Container D** (myhub, public like A, no login needed; built on the project node from
`containers/D_gpu4pyscf.def`). Pushed 2026-11-14 as an ORAS artifact; its layer digest is the SIF's sha256,
checked anonymously against the registry manifest.
```bash
module load apptainer/1.4.5
mkdir -p /appsdata/$USER/sif
export APPTAINER_CACHEDIR=/appsdata/$USER/apptainer-cache
apptainer pull /appsdata/$USER/sif/D_gpu4pyscf.sif oras://myhub.ksc.re.kr/pah/d_gpu4pyscf:gh200-2026-11-14
sha256sum /appsdata/$USER/sif/D_gpu4pyscf.sif     # must be 09afb47a44974c8b4575ba18b8664e6eb0f681bd3264c47c858be0d5ae20893b
apptainer cache clean --force
```
Run the pull on a GPU login node (`dm01`–`dm04` have no Apptainer). The first three pushes (by Claude, same day)
failed after 60–90 s with `Patch …/blobs/uploads/…: file already closed`; a later push by hand went through. If a
future push fails the same way, retry it. Fallbacks if the pull ever fails: `rsync -P` the SIF from the project
node (`/var/tmp/jjlee-pah/sif/`), or build it on a GPU compute node with
`APPTAINER_TMPDIR=/scratch/$USER/tmp apptainer build --fakeroot /appsdata/$USER/sif/D_gpu4pyscf.sif
containers/D_gpu4pyscf.def` from the repository root (about 7 min; too much for the login node's 120 CPU-minutes;
its hash will differ, and the check job in step 3 validates it).

**2. Code and inputs.** The scripts are on branch `feat-gpu-gh200`
(`cd /appsdata/$USER/pah_stage0 && git fetch && git checkout feat-gpu-gh200`). The inputs come from the project node
as one tarball (copy it to `/appsdata/$USER/d1/`), `/var/tmp/jjlee-pah/d1/seed_hangang.tgz` (5.7 MB, sha256 `d711af72…7f63`; no Mai et al. geometries):
`seed_hangang/mlff_seed/` (`frames/`, `frames_transfer/`, `gjf*/manifest.json`, `species_v2.csv`) and
`seed_hangang/seed_dft_ref/` (the project node's 202 pilot results, the reference for the check below).
```bash
cd /appsdata/$USER/d1 && tar xzf seed_hangang.tgz
```

**3. Check job (one node, about 15 min).** 15 of the pilot frames (C10–C96, both charges, 7 with stability tests),
rerun here and compared with the project node's results against the G2 targets (1e-7 Ha, 1e-5 Ha/bohr, 1e-3 D):
```bash
cd /appsdata/$USER/pah_stage0
S=/appsdata/$USER/d1/seed_hangang
sbatch --time=01:00:00 d1/slurm_dft_seed.sbatch /appsdata/$USER/sif/D_gpu4pyscf.sif $S/mlff_seed $S/check \
  --names C10H8_330_q0_f008,C10H8_330_q0_f031,C10H8_330_q1_f006,C10H8_330_q1_f031,C24H14_204_q0_f008,C24H14_204_q0_f031,C24H14_204_q1_f007,C24H14_204_q1_f030,C48H20_146_q0_f008,C48H20_146_q0_f030,C48H20_146_q1_f008,C48H20_146_q1_f031,C96H30_3317_q0_f004,C96H30_3317_q1_f004,C96H30_3317_q1_f003
python3 d1/dft_seed_run.py compare $S/seed_dft_ref $S/check      # must print PASS; compare the t_wall pairs too
```
`compare` needs only numpy on the host Python; otherwise run it with `apptainer exec D_gpu4pyscf.sif python …`.

**4. Production.** Seed the output with the pilot results (they are skipped), then one array of up to 5 nodes:
```bash
mkdir -p $S/seed_dft && cp $S/seed_dft_ref/*.json $S/seed_dft/
sbatch --array=0-4 d1/slurm_dft_seed.sbatch /appsdata/$USER/sif/D_gpu4pyscf.sif $S/mlff_seed $S/seed_dft
ls $S/seed_dft/*.json | wc -l        # 7,504 when complete
```
Each node runs 4 GPUs × 2 processes, one shard each, pinned to the GPU's cores (`nvidia-smi topo -m`). Finished
frames are skipped by file name, so resubmitting the same command, with any array size, continues; processes stop
starting new frames 30 min before the time limit. Never run two jobs on the same output directory at once.
Logs: `$S/seed_dft/logs/`.

**5. Afterwards, on the project node:** copy `$S/seed_dft/` back to `/var/tmp/jjlee-pah/d1/seed_dft/`, then
`python d1/dft_seed_run.py collect … seed_dft.extxyz` and `d1/mlff_qa.py` (xTB references, `xtb` env), as in
`docs/plan/charged_mlff_plan.md`.

## Step 4 onwards (planning notes written before step 3; superseded where they differ)

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
