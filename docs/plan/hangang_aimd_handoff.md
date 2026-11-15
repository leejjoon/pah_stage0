# Handoff: naphthalene DFT MD reference on Hangang

Written 2026-11-15 on the project node (`gh200`), for a session on KISTI Hangang (account `t641a02`), person or
Claude. Branch `feat-gpu-gh200`. Container D and the general Hangang setup are as in
`docs/plan/hangang_seed_dft_handoff.md` (steps 1–2 there: code, image); do those first if they are not done.

**Alternative route (2026-11-15):** the same checkpoints on a cloud VM's CPUs, `docs/plan/cloud_aimd_handoff.md`
(branch `cloud-aimd`). Run them in one place only; ask the user which.

## What this is

This is the DFT-level reference for the charged-MLFF work: committee review B3, part one
(`docs/plan/charged_mlff_plan.md`). The method is Born–Oppenheimer MD of naphthalene with B3LYP (VWN3)/6-31G\* and
6D d functions, the level of the training data.

The run matrix is 8 trajectories: neutral and cation × 300 and 900 K × 2 replicas. Each trajectory is:
- 2 ps of Langevin NVT, then 20 ps of NVE;
- a 0.5 fs time step;
- a dipole recorded every 1 fs.

The B4 gate's deciding number comes from these runs: the slope of the cation C–C band strength with temperature.
The ML force fields will later be judged against them.

The runs started on the project node's CPUs (conda-forge PySCF, 8 Grace threads each). The user stopped them on
2026-11-13 because they belong on another machine. They now continue on Hangang's GPUs with GPU4PySCF
(`d1/aimd_pyscf.py --gpu`).

## Where things stand

- **Checkpoints:** in `/var/tmp/jjlee-pah/d1/aimd_naph_hangang.tgz` on the project node (280 kB, sha256
  `a957f38e…b73d`). The user copies it to `/appsdata/$USER/d1/`; ask for it if it is missing. It holds:
  - the 8 `naph_q<q>_<T>K_r<r>.npz` checkpoints;
  - the two inputs `C10H8_330_q{0,1}.xyz` (PAHdb geometries; no Mai et al. data);
  - the project node's logs and launcher, in `logs_project_node/`.

  Progress so far:

  | Run | Step reached | Time reached | Steps left |
  |---|---|---|---|
  | neutral (all 4) | 7,600 of 44,000 | 3.8 ps | 36,400 |
  | cation, 300 K | 5,200 | 2.6 ps | 38,800 |
  | cation, 900 K | 5,000 | 2.5 ps | 39,000 |

  All of them are past NVT and in NVE.
- **The checkpoints are valid:**
  - the SCF never needed the Newton fallback (at most 14 cycles), so the bug fixed in commit `64b3aba` never
    triggered;
  - total-energy drift is thermal (0.1–0.3 eV between records 20 steps apart);
  - ⟨S²⟩ ≤ 0.772.
- **GPU continuation, checked on the project node (H100):**
  - at a 900 K checkpoint geometry, GPU and CPU agree to 3e-10 Ha in energy and 8e-6 Ha/bohr in forces, with the
    same grid (PySCF default: level 3, NWChem pruning, 219,648 points);
  - a resumed cation run continued on the GPU with the NVE total energy unchanged across the switch;
  - each npz records which engine ran which steps (field `engines`, e.g. `0:PySCF-CPU;5000:GPU4PySCF;`).
- **Speed per step on one H100**, with each SCF starting from the previous step's density:

  | Neutral | Cation |
  |---|---|
  | 5.7 s | 7.5 s |

  These are steady-state times from `--time-steps 20` (dipole and ⟨S²⟩ included). The first minutes of each job are
  slower: a full SCF from scratch and setup.

  - Only one trajectory per GPU makes sense: naphthalene is launch-bound, so two per GPU gain nothing.
  - Container D's own CPU PySCF is slow on Grace (27 s per SCF at 8 threads), so the CPUs are no use here.
- **Wall-time estimate:** cations about 81 h, neutrals about 58 h, plus perhaps 10 % for Hangang's GPUs. That is
  4 jobs of 24 h for the cation node and 3 for the neutral node. The job script resubmits itself.
- **Job script** `d1/slurm_aimd.sbatch`:
  - one node, one trajectory per GPU, each pinned to its GPU's cores;
  - each process checkpoints and stops 20 min before the time limit;
  - if every process stopped that way and at least one advanced, the job resubmits itself (at most 8 times);
  - it does not resubmit if a process crashed.

  It was tested on the project node with stand-in `module`/`squeue`/`sbatch`; it has not run under real Slurm.

## Rules

- **Jobs:** at most 5 per user may run on Hangang, and this uses 2 for about 3.5 days. Agree the timing with the
  seed-set DFT (`hangang_seed_dft_handoff.md`, 5 nodes for about 5 h): run that first, or run it on 3 nodes
  alongside this. Ask the user before submitting.
- **Concurrency:** never run two jobs on the same checkpoints at once. Before submitting by hand, check
  `squeue -u $USER` for a pending resubmission.
- **Engines:** do not change the method, grid or time step, and do not restart a trajectory from scratch. Continue
  the checkpoints.
- **Git and secrets:** commit only on `feat-gpu-gh200`, push only when asked, and never print `.env`.

## Steps

**1. Inputs**
```bash
cd /appsdata/$USER/d1
sha256sum aimd_naph_hangang.tgz   # a957f38ee5b9ee0332c7a5a9916e64bc607d5cd5da19d4c90362d92e5edbb73d
tar xzf aimd_naph_hangang.tgz     # -> aimd_naph/
cp -a aimd_naph aimd_naph_start   # untouched copy of the starting checkpoints
```

**2. Short check job (30 min, one node)** on a scratch copy of two runs, to see the GPU path work here before
committing two nodes for days:
```bash
cd /appsdata/$USER/pah_stage0 && git pull
mkdir -p /appsdata/$USER/d1/aimd_check && cp /appsdata/$USER/d1/aimd_naph/{C10H8_330_q0.xyz,C10H8_330_q1.xyz,naph_q0_900K_r0.npz,naph_q1_900K_r0.npz} /appsdata/$USER/d1/aimd_check/
AIMD_RESUBMITS=0 sbatch --time=00:30:00 d1/slurm_aimd.sbatch /appsdata/$USER/sif/D_gpu4pyscf.sif \
  /appsdata/$USER/d1/aimd_check naph_q0_900K_r0,naph_q1_900K_r0
```
- `AIMD_RESUBMITS=0` stops it from resubmitting; plain `sbatch` exports the environment.
- The job log `aimd_<jobid>.log` should show:
  - each run on its own GPU and 72-core block;
  - `step 7600 -> ~7,800–7,850` for the neutral and `5000 -> ~5,200` for the cation (about 25 min of stepping, the
    30 min limit minus a 4.5 min margin, at about 6–8 s per step);
  - `not resubmitting (… resubmits left 0)`.
- Per-run logs are in `aimd_check/logs/`. Check them:
  ```bash
  apptainer exec /appsdata/$USER/sif/D_gpu4pyscf.sif python -c "
  import numpy as np
  for n in ('naph_q0_900K_r0', 'naph_q1_900K_r0'):
      z = np.load('/appsdata/$USER/d1/aimd_check/' + n + '.npz'); qa = np.array(z['qa'])
      print(n, int(z['step']), z['engines'], 'Etot drift last 10 records (eV):', np.ptp(qa[-10:, 2]).round(3), 'max cycles', int(qa[-10:, 4].max()))"
  ```
  The NVE total-energy spread should match the CPU part, a few tenths of an eV at most.
- Report the seconds per step from the log, and delete `aimd_check/` afterwards.

**3. Production**, after the user agrees. Two nodes, cations on one and neutrals on the other:
```bash
cd /appsdata/$USER/pah_stage0
D=/appsdata/$USER/d1/aimd_naph; SIF=/appsdata/$USER/sif/D_gpu4pyscf.sif
sbatch d1/slurm_aimd.sbatch $SIF $D naph_q1_300K_r0,naph_q1_300K_r1,naph_q1_900K_r0,naph_q1_900K_r1
sbatch d1/slurm_aimd.sbatch $SIF $D naph_q0_300K_r0,naph_q0_300K_r1,naph_q0_900K_r0,naph_q0_900K_r1
```

**Progress:**
- Per-run logs: `$D/logs/<run>.log`, appended across jobs.
- Each job's summary goes to `aimd_<jobid>.log` in the repository root. It ends with each run's step and either a
  resubmission, `all runs done`, or `not resubmitting …`.
- **If a job does not resubmit** (a crash, or the resubmit limit reached): read the run logs, fix the cause if it
  is clear, and resubmit the same command; it resumes from the last checkpoint. If the cause is not clear, report
  it.
- **Checkpoints:** written every 100 fs (200 steps, about 20 min) and at every stop.

**4. Hand the results back** when both jobs end with `all runs done` (44,000 steps each):
```bash
cd /appsdata/$USER/d1 && tar czf aimd_naph_done.tgz aimd_naph
```
The user copies it to the project node. The analysis (spectra from the NVE dipoles, the B4 cation band-strength
slope) runs there. Report:
- the total wall time;
- the seconds per step;
- any SCF that needed many cycles (column 5 of `qa`);
- ⟨S²⟩ for the cations (column 4).
