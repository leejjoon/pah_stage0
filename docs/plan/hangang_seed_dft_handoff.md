# Handoff: charged-MLFF seed-set DFT on Hangang

Written 2026-11-14 on the project node (`gh200`), for a session on KISTI Hangang (account `t641a02`), person or
Claude. Branch `feat-gpu-gh200`. The command reference is `docs/plan/cluster_migration.md`, section "Seed-set DFT";
this note adds the context, the expected results and what to report back.

## What this is

D1 needs machine-learned force fields for PAH neutrals **and cations** (`docs/plan/charged_mlff_plan.md`). Phase 1
labels a seed set of 7,504 geometries with B3LYP/6-31G\*:
- 124 neutral/cation pairs from PAHdb;
- 32 frames each: minimum, displacements, normal-mode samples, bond-stretch and out-of-plane boundary cases, and
  GFN2-xTB MD at 300–1500 K.

For each frame the run computes:
- the energy, forces and dipole;
- ⟨S²⟩ for the cations;
- a wavefunction stability test for the 1,454 frames flagged in the manifest.

The engine is GPU4PySCF 1.9.2 in container D (`containers/D_gpu4pyscf.md`). Gaussian is not available, and
GPU4PySCF reproduces PAHdb's Gaussian B3LYP/6-31G\* to ≤ 2 µHa.

## Where things stand

- **Validation:** on the project node's H100, GPU4PySCF matches CPU PySCF to:

  | Quantity | Agreement |
  |---|---|
  | Energy | 2e-10 Ha |
  | Forces | 7e-7 Ha/bohr |
  | Dipole | 3e-5 D |
  | ⟨S²⟩ | 1.4e-7 |

  Details: `docs/research/pyscf_gpu_jax.md` §5.
- **Pilot:** 202 frames from 4 pairs (C10H8_330, C24H14_204, C48H20_146, C96H30_3317), 2.1 GPU-hours.
  - Everything converged in ≤ 47 SCF cycles, with no second-order (Newton) fallback.
  - All 33 stability tests were stable.
  - `mlff_qa.py` flagged nothing.
  - These results are the reference for the check job below.
- **Cost of the rest:** about 87 H100-hours in all, the pilot included.
  - The project node would take about 33 h on its two GPUs.
  - 5 Hangang nodes (20 GPUs) should take about 4–5 h of wall time.
- **Container D** is in myhub: `oras://myhub.ksc.re.kr/pah/d_gpu4pyscf:gh200-2026-11-14`. It is public, and its
  layer digest equals the SIF's sha256 `09afb47a…893b`.
- **Job script** `d1/slurm_dft_seed.sbatch`:
  - one whole node per job: 4 GPUs × 2 processes, each process pinned to its GPU's cores (from `nvidia-smi topo -m`);
  - one shard per process.

  It was tested on the project node with stand-in `module`/`squeue` commands: 8 pilot frames reproduced to 2e-12 Ha.
  **It has never run under real Slurm**, so the check job is its first real test.
- **Inputs:** `/var/tmp/jjlee-pah/d1/seed_hangang.tgz` on the project node (5.7 MB, sha256 `d711af72…7f63`). It
  holds the frames, the manifests and the 202 pilot results. The user copies it to `/appsdata/$USER/d1/`; if it is
  not there, ask for it. The session on Hangang cannot reach the project node.

## Rules

- **Mai et al. geometries:** never copy or commit them (C90H24_618, C144H30_641; Zenodo 15771437, CC BY-NC-SA).
  The tarball holds none.
- **Git:** commit only on `feat-gpu-gh200`, and push only when the user asks.
- **`.env`:** never print it.
- **Before the production array:** ask the user. It uses 5 nodes, and the account allows 5 running jobs.
- **Output directory:** never run two jobs on the same output directory at once, or they repeat frames.
- **Login nodes:** kill processes after 120 CPU-minutes. Pull images on a GPU login node, run nothing heavy there,
  and put caches under `/appsdata` or `/scratch`.

## Steps

**1. Code and image**
```bash
cd /appsdata/$USER/pah_stage0
git fetch && git checkout feat-gpu-gh200 && git pull
module load apptainer/1.4.5
mkdir -p /appsdata/$USER/sif
export APPTAINER_CACHEDIR=/appsdata/$USER/apptainer-cache
apptainer pull /appsdata/$USER/sif/D_gpu4pyscf.sif oras://myhub.ksc.re.kr/pah/d_gpu4pyscf:gh200-2026-11-14
sha256sum /appsdata/$USER/sif/D_gpu4pyscf.sif   # 09afb47a44974c8b4575ba18b8664e6eb0f681bd3264c47c858be0d5ae20893b
apptainer cache clean --force
```
The pull takes about a minute for 6.1 GB.

**2. Inputs**
```bash
cd /appsdata/$USER/d1
sha256sum seed_hangang.tgz    # d711af72fcb77c88440d5ec72fe51ad20d0000bae93f63afc873c6625ad17f63
tar xzf seed_hangang.tgz
S=/appsdata/$USER/d1/seed_hangang
ls $S/mlff_seed/frames | wc -l; ls $S/mlff_seed/frames_transfer | wc -l; ls $S/seed_dft_ref | wc -l   # 232, 16, 202
```

**3. Check job:** one node, 1 h limit, in practice about 10–15 min. It reruns 15 pilot frames:
- C10 to C96, both charges;
- 7 of them with stability tests;
- among them the most expensive pilot frame, C96H30_3317_q1_f004 (412 s on the project node).

```bash
cd /appsdata/$USER/pah_stage0
sbatch --time=01:00:00 d1/slurm_dft_seed.sbatch /appsdata/$USER/sif/D_gpu4pyscf.sif $S/mlff_seed $S/check \
  --names C10H8_330_q0_f008,C10H8_330_q0_f031,C10H8_330_q1_f006,C10H8_330_q1_f031,C24H14_204_q0_f008,C24H14_204_q0_f031,C24H14_204_q1_f007,C24H14_204_q1_f030,C48H20_146_q0_f008,C48H20_146_q0_f030,C48H20_146_q1_f008,C48H20_146_q1_f031,C96H30_3317_q0_f004,C96H30_3317_q1_f004,C96H30_3317_q1_f003
```
Then check the results:
```bash
ls $S/check/*.json | wc -l        # 15
python3 d1/dft_seed_run.py compare $S/seed_dft_ref $S/check
# no numpy on the host Python? use the image:
# apptainer exec --bind /appsdata /appsdata/$USER/sif/D_gpu4pyscf.sif python d1/dft_seed_run.py compare $S/seed_dft_ref $S/check
```

What to expect:
- **Job log** (`seed_dft_<jobid>_4294967294.log` in the repository root, or `seed_dft_<jobid>_*.log`):
  - a header naming 4 GH200 GPUs and `shards of 8`;
  - one `shard i/8: GPU g, cores a-b` line per process, with each GPU on its own 72-core block;
  - a per-process summary;
  - `frames done in OUT: 15`.

  `--names` is applied after sharding, so the 15 frames are spread unevenly. Some shards report `0 frames`, which
  is fine.
- **`compare`:** must end in `PASS`. Expect energy differences of 1e-10 Ha or smaller; the targets are 1e-7 Ha,
  1e-5 Ha/bohr, 1e-3 D and 1e-4 for ⟨S²⟩. Stability verdicts must match.
- **Timing:** the `t` pairs give Hangang's time per frame against the project node's (left column). Note the ratio.
  Expect it to be similar, perhaps up to 10–20 % slower (the H100 has 96 GB of HBM3 here, against 144 GB of HBM3e
  there, and the CPUs are shared differently). Two processes share each GPU, so per-frame times are longer than in
  the pilot, where some frames ran alone on a GPU.

If a check fails:
- **`compare` FAIL with small differences** (1e-8 to 1e-6 Ha): report it and stop. Do not loosen the targets.
- **Every process dies at once:** look at `$S/check/logs/<jobid>_t0_g*_p*.log`. A CUDA, driver or CuPy error points
  to the image or the `--nv` setup. The first run compiles CuPy kernels into `/appsdata/$USER/cache/cupy`, which
  costs a minute or so.
- **The pinning line shows the fallback split, or the cores look wrong:** the `nvidia-smi topo -m` format differs
  here. Report it; the run still works, just possibly slower.

**4. Report to the user and ask before production.** Give the job id, the `compare` summary line, the timing ratio
and anything odd.

**5. Production**, after the user agrees:
```bash
mkdir -p $S/seed_dft && cp $S/seed_dft_ref/*.json $S/seed_dft/    # the 202 pilot frames are skipped
sbatch --array=0-4 d1/slurm_dft_seed.sbatch /appsdata/$USER/sif/D_gpu4pyscf.sif $S/mlff_seed $S/seed_dft
```
The array runs as 5 nodes × 8 shards = 40 processes. The largest molecules go first, so the shards finish at about
the same time.

Progress:
```bash
ls $S/seed_dft/*.json | wc -l                                    # 7504 when complete
tail -n 1 $S/seed_dft/logs/<jobid>_t*_g*_p*.log                  # per process: [k/n] frame ... elapsed
grep -l "Traceback\|Error" $S/seed_dft/logs/*.log
```
- **If the array ends before all 7,504 are done** (time limit, a node failure or a crash): resubmit the same
  command. Any array size works, because finished frames are skipped by file name.
- **If one frame crashes its process again and again:** rerun it alone with `--names` to see the error, and report
  it. Do not drop it silently.

**6. Hand the results back**

```bash
apptainer exec --bind /appsdata /appsdata/$USER/sif/D_gpu4pyscf.sif \
  python d1/dft_seed_run.py collect $S/seed_dft $S/seed_dft.extxyz
tar czf /appsdata/$USER/d1/seed_dft_hangang.tgz -C $S seed_dft seed_dft.extxyz
```
- `collect` prints a QA summary: unconverged frames, Newton fallbacks, cation ⟨S²⟩ > 0.80, unstable frames and the
  largest forces. Report it.
- The user copies the tarball to the project node. The xTB-based `d1/mlff_qa.py` runs there (in the `xtb` env).

Things to watch in the summary:
- **Newton fallbacks with a stability test:** a frame that needed the Newton fallback *and* has a stability test
  takes an untested path (`.stability()` on the second-order SCF object; it never occurred in the pilot). List
  those frames.
- **Cation ⟨S²⟩:** the pilot cations stayed at or below 0.786. The C144H30 cation in the timing series reached
  1.04 (that species is not in the seed set). Report any cation above about 0.85.

## Files

| File | Role |
|---|---|
| `d1/slurm_dft_seed.sbatch` | Slurm job: 4 GPUs × `NPROC` (default 2) shards per node, array-aware, `--hours` from the time left |
| `d1/dft_seed_run.py` | `run` (resumable, atomic per-frame JSON), `compare`, `collect` |
| `containers/D_gpu4pyscf.def`, `.md` | Image recipe and notes (rebuild on a compute node only if the pull fails; see cluster_migration.md) |
| `docs/plan/cluster_migration.md` | Hangang facts and the command reference ("Seed-set DFT") |
| `docs/research/pyscf_gpu_jax.md` §5 | Validation, timings, the cost estimate |
| `docs/plan/charged_mlff_plan.md` | Phases 1–3 |
