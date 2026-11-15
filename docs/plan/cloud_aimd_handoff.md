# Handoff: naphthalene DFT MD reference on a cloud VM (Google Cloud or another provider)

Written 2026-11-15 on the project node (`gh200`), branch `cloud-aimd` (from `feat-gpu-gh200`). For the user, who
creates the VM, and for whoever (person or Claude) then works on it.

This is the CPU alternative to `docs/plan/hangang_aimd_handoff.md`. On Hangang the runs would hold two 4-GPU nodes
for about 3.5 days while using a small part of each GPU: naphthalene is too small for a GPU (18 atoms, 166 basis
functions; launch-bound, one H100 is only about 1.5× faster than 8 Grace cores). On a cloud VM it runs on plain CPUs
with the same conda-forge PySCF the runs started with.

**Run it in one place only.** Never run the same checkpoints on Hangang and in the cloud at once. If one route is
abandoned part-way, carry its latest checkpoints to the other (`make_bundle.sh` takes any tarball of the same
layout).

## Current route: gmunu (KASI), since 2026-11-15

**The 8 runs are on gmunu (`gmunu.kasi.re.kr`, port 7774 in `~/.ssh/config`), not in the cloud.** Do not run these
checkpoints anywhere else. Why: the GCP project `naph-aimd-2611` (org nais.re.kr, billing `015AF0-546A55-3F1C6E`)
has 12 vCPUs for all regions, and the request for 72 was denied automatically. gmunu has no time limit and no
quota, and its queue was empty.

- **Where:** `~/aimd` on gmunu: `env/` (conda-forge, the versions below, water energy checked), `code/`, and `run/` with
  the checkpoints, `logs/<run>.log` and `logs/slurm_<job>_<id>.out`. `/home` is shared with the compute nodes.
- **Jobs** (submitted 2026-11-15 16:08 from `~/aimd/run`; 32 cores per run, `d1/cloud/gmunu_aimd.sbatch`):

  | Job | Node type | Runs | From step |
  |---|---|---|---|
  | `aimd-q0` | Zen 2 (`--constraint=AMD-R6525-7702`) | the 4 neutrals | 7,600 |
  | `aimd-q1z3` | Zen 3 (`AMD-R6525-7713`; only node6 is up) | cation 300 K r1, 900 K r1 | 5,200, 5,000 |
  | `aimd-q1z2` | Zen 2 | cation 300 K r0, 900 K r0 | 5,200, 5,456 |

  Cation 900 K r0 ran 5,000 → 5,456 on a GCP c4a-highcpu-8 Spot VM first (`engines` shows it); the VM is deleted, its
  logs are in `run/logs/gcp/`, and the step-5,000 checkpoint is kept as `run/naph_q1_900K_r0.step5000.npz.bak`.
- **Expected:** neutrals about 4.5 days, cations 6–7 days (node6 pair faster).
- **Restarts** are automatic; every path was tested on copies before production:
  - Each job has a 3-day limit. Five minutes before it, the runs save and the job resubmits itself (up to 20 times).
  - After a node failure, Slurm requeues the job, and it resumes.
  - After `scancel`, the runs save and the job stays stopped. To continue, submit the same command again (below).
- **Progress:** `~/aimd/code/cloud/launch_aimd.sh status ~/aimd/run`, `squeue -u $USER`, `tail ~/aimd/run/logs/slurm_aimd-*.out`.
- **Resubmit by hand** (after a crash is fixed, or after `scancel`), from `~/aimd/run`:
  ```bash
  S=~/aimd/code/cloud/gmunu_aimd.sbatch; sub() { A="-J $1 --constraint=$2"; AIMD_SBATCH_ARGS="$A" sbatch $A $S ~/aimd/run $3 32; }
  sub aimd-q0   AMD-R6525-7702 naph_q0_300K_r0,naph_q0_300K_r1,naph_q0_900K_r0,naph_q0_900K_r1
  sub aimd-q1z3 AMD-R6525-7713 naph_q1_300K_r1,naph_q1_900K_r1
  sub aimd-q1z2 AMD-R6525-7702 naph_q1_300K_r0,naph_q1_900K_r0
  ```
  A run that is already going elsewhere is skipped (a per-run flock, which works across gmunu nodes).
- **Results:** when `status` shows all 8 at 44,000, copy `~/aimd/run` back to the project node, then follow step 8
  below (the analysis part).

**Timings** (s/step for a cation and a neutral; "8×8" means 8 runs at once with 8 cores each):

| Machine | Layout | Cation | Neutral |
|---|---|---|---|
| Grace (project node) | 8×8 | 13 | 8.8 |
| GCP Axion c4a | 1×8 | 12.80 | 8.58 |
| z2mini (Ryzen AI Max+ 395, Zen 5) | 2×8 | 12.54 | 8.61 |
| gmunu Zen 2 (EPYC 7702) | 8×8 | 26.4–27.5 | 19.7–20.3 |
| gmunu Zen 2 | 8×16 | 18.8–19.4 | 12.9–13.1 |
| gmunu Zen 2 | 4×32 | 15.8–16.1 | – |
| gmunu Zen 2 | 2×64; 1×64 | 15.3–15.9; 16.1 | – |
| gmunu Zen 3 (EPYC 7713) | 8×16; 1×16; 1×32 | 18.5–19.4; 16.3; 13.2 | 11.7–12.2 |

On gmunu a run stops getting faster at 32 cores, so extra nodes do not help. Faster cores do: node7 and node8 (Zen 3)
are drained ("SlurmdSpoolDir is full"); if gmunu's admin clears them, 2 cations per Zen 3 node would finish about
half a day sooner.

**Launcher changes made for this:** `launch_aimd.sh` now gives each run whole physical cores read from `lscpu`. On
z2mini the two hyperthreads of a core are numbered 0,1, so the old `cores 0-7` put 8 threads on 4 cores. `status`
no longer crashes on checkpoints that have no `engines` field. The x86 path of `setup_vm.sh` was tested on z2mini
and on gmunu.

## What this is

The DFT-level reference for the charged-MLFF work: committee review B3, part one (`docs/plan/charged_mlff_plan.md`).
Born–Oppenheimer MD of naphthalene with B3LYP (VWN3)/6-31G\*, 6D d functions, the level of the training data.

- 8 trajectories: neutral and cation × 300 and 900 K × 2 replicas.
- Each: 2 ps Langevin NVT, then 20 ps NVE; 0.5 fs steps (44,000 in all); dipole every 1 fs.
- The B4 gate's deciding number comes from these runs: the slope of the cation C–C band strength with temperature.

## Where things stand

- **Checkpoints** (the same as for Hangang; all past NVT, in NVE):

  | Run | Step reached | Steps left |
  |---|---|---|
  | neutral (all 4) | 7,600 of 44,000 | 36,400 |
  | cation, 300 K | 5,200 | 38,800 |
  | cation, 900 K | 5,000 | 39,000 |

  They are valid: the SCF never needed the Newton fallback (bug fixed in `64b3aba`), NVE drift is thermal, ⟨S²⟩ ≤
  0.772.
- **Speed measured on the project node** (Grace, Arm Neoverse V2), conda-forge PySCF, 8 cores per run, 8 runs at
  once:

  | | Neutral | Cation |
  |---|---|---|
  | seconds per step | 8.8 | 13 |
  | hours for the steps left | 89 | 141 |

  In all about **7,400 core-hours**. With 16 cores per cation run, container D's PySCF ran about 1.4× faster than
  with 8; this was not measured for the conda env. The timing check (step 4) settles both on the VM.
- **Memory:** 2.1 GB peak per run. Any machine shape with ≥ 4 GB per run is enough ("highcpu" is fine).
- **Code on this branch:**

  | File | Role |
  |---|---|
  | `d1/aimd_pyscf.py` | the MD; now also stops cleanly on SIGTERM (finishes the step, saves, exits 0) and records `--tag` in the npz `engines` field |
  | `d1/cloud/launch_aimd.sh` | `run` / `timing` / `status`; pins each run to its own cores; resumes; skips finished runs and runs another launcher already holds (flock); checkpoints every 20 fs |
  | `d1/cloud/setup_vm.sh` | `install` (micromamba, the env, a water-energy check against the project node) and `service` (systemd unit `aimd` that resumes on every boot) |
  | `d1/cloud/dft_env_linux-aarch64.lock` | the project node's env, package for package (python 3.12.16, pyscf 2.15.0, ase 3.29.0, numpy 2.5.4, scipy 1.18.1, OpenBLAS 0.3.34 openmp) |
  | `d1/cloud/gcp_keepalive.sh` | restarts a preempted GCP Spot VM until the runs are done |
  | `d1/cloud/make_bundle.sh` | builds the bundle below |

- **Tested on the project node:**
  - `setup_vm.sh install` from the bundle with a throwaway HOME: the env builds from the lock, and the water energy
    matches the project node to 1e-8 Ha.
  - The launcher with a stub in place of the MD: resume, a second launcher skipping runs already going, a stop
    (exit 3, every run saved), a crash (exit 1), completion (exit 0, `ALL_DONE`), restarting the cations at 16
    threads once the neutrals were done.
  - The real MD from a cation checkpoint: resumed at 5,000, stopped on SIGTERM at 5,002 within one step, `engines`
    = `0:PySCF-CPU;5000:PySCF-CPU@cloud-aarch64;`.
  - Not tested: the systemd unit and a real Spot preemption. Step 6 checks the unit once on the VM.
- **Bundle:** `/var/tmp/jjlee-pah/d1/aimd_naph_cloud.tgz` on the project node (about 300 kB; sha256 in
  "Bundle" below). It holds only the 8 checkpoints, the two PAHdb naphthalene geometries and the code: no logs, no
  `.env`, no Mai et al. data.

## Which machine

**Recommended: one Google Cloud `c4a-highcpu-64` Spot VM** (Axion, Arm Neoverse V2: the project node's CPU core;
one vCPU is one physical core).
- 8 runs × 8 cores. The neutrals finish after about 89 h; then restart the cations with 16 cores each on the same
  64 (step 7). Wall time about 4.5–6 days, depending on the 16-core speed-up.
- Spot is much cheaper than on-demand, and the runs lose almost nothing when preempted (SIGTERM saves after the
  current step; Google gives about 30 s). The cost of preemption is downtime: a preempted Spot VM stays stopped
  until something starts it, here `gcp_keepalive.sh` running where `gcloud` is logged in.
- On-demand is the simpler choice: no restarts needed, and `--poweroff-when-done` stops the billing at the end.
  The user decides; check prices in the GCP pricing calculator (C4A, the region, Spot and on-demand). No price is
  quoted here.

**Alternatives** (same steps; only the VM creation differs):
- **Two VMs:** cations on a 64-vCPU VM at 16 cores each, neutrals on a 32-vCPU VM at 8. Shorter wall time if 16
  cores help; about the same core-hours.
- **Other Arm VMs:** AWS Graviton4 instances (also Neoverse V2) behave the same; Azure Cobalt 100 (Neoverse N2) is
  an older core, likely slower. The lock file applies to any aarch64 Linux VM.
- **x86 VMs** (GCP C4D/C3D, AWS, Azure): `setup_vm.sh` builds the same versions from conda-forge for x86. Turn off
  SMT (GCP: `--threads-per-core=1`), or each "core" the launcher assigns is half a physical core. Untested; the
  timing check decides.
- **GPU VMs:** not worth it for this molecule (above). Container D is built for aarch64/sm_90 only.

Requirements for any provider: Linux with systemd, `curl`, `flock`, `taskset` (util-linux), outbound HTTPS to
GitHub and conda-forge for the install, a disk that survives a stop (the boot disk is enough: 30 GB), and a way to
copy 300 kB in and a few MB out.

## Rules

- **One place at a time** (above). Before starting, check that no Hangang job is running on these checkpoints.
- **Do not change the method, grid or time step, and do not restart a trajectory from scratch.** Continue the
  checkpoints. Changing the thread count is fine.
- **What leaves the project node:** only the bundle. Never copy `.env`, other checkpoints or any Mai et al. data
  (C90H24_618, C144H30_641) to the VM.
- **Git:** commit only on `cloud-aimd`; push only when the user asks. The VM does not need the repository.
- **Delete the VM** after the results are copied back; a stopped VM still bills for its disk.

## Steps

**1. Bundle** (project node; already built, rebuild only if the code or checkpoints changed):
```bash
cd /home/jjlee/projects/pah_check01 && git switch cloud-aimd
bash d1/cloud/make_bundle.sh      # -> /var/tmp/jjlee-pah/d1/aimd_naph_cloud.tgz, prints its sha256
```

**2. VM** (the user, with `gcloud` logged in to the project; `ZONE` one that offers C4A, e.g. from
`gcloud compute machine-types list --filter="name=c4a-highcpu-64" --format="value(zone)"`):
```bash
ZONE=...; VM=naph-aimd
gcloud compute instances create $VM --zone=$ZONE --machine-type=c4a-highcpu-64 \
  --provisioning-model=SPOT --instance-termination-action=STOP \
  --image-family=ubuntu-2404-lts-arm64 --image-project=ubuntu-os-cloud \
  --boot-disk-size=30GB --boot-disk-type=hyperdisk-balanced
```
- On-demand: leave out the two Spot flags.
- C4A takes Hyperdisk only. If the image family name has changed:
  `gcloud compute images list --filter="family~arm64" --format="value(family,project)"`.

**3. Install** (copy, then on the VM):
```bash
gcloud compute scp /var/tmp/jjlee-pah/d1/aimd_naph_cloud.tgz $VM:~ --zone=$ZONE
gcloud compute ssh $VM --zone=$ZONE
# on the VM:
sha256sum aimd_naph_cloud.tgz      # as in "Bundle" below
tar xzf aimd_naph_cloud.tgz && bash aimd_cloud/code/cloud/setup_vm.sh install
```
It ends with the versions, `water ... -76.40886982 Ha (project node: -76.4088698182)` and `checkpoints: 8`. The
script stops if either differs. About 5 minutes.

**4. Timing check** (about 5 minutes each; uses the starting geometry, writes no checkpoints):
```bash
~/aimd/code/cloud/launch_aimd.sh timing ~/aimd/run       # all 8 at once, 8 cores each: the production load
THREADS_Q1=16 RUNS=naph_q1_300K_r0,naph_q1_300K_r1,naph_q1_900K_r0,naph_q1_900K_r1 \
  ~/aimd/code/cloud/launch_aimd.sh timing ~/aimd/run     # the cations at 16 cores each
```
Each line gives seconds per step, SCF cycles and peak RSS. Expect about 8.8 s (neutral) and 13 s (cation) at 8
cores if Axion matches Grace. Then:

**5. Report and ask.** Give the user both timings, the predicted wall time (steps left × s/step), and whether 16
cores help the cations. The user decides whether to go on, and on Spot or on-demand.

**6. Production:**
```bash
bash ~/aimd/code/cloud/setup_vm.sh service           # Spot, with gcp_keepalive.sh running elsewhere
# or, on-demand:  bash ~/aimd/code/cloud/setup_vm.sh service --poweroff-when-done
```
- This writes and starts the systemd unit `aimd` (8 cores per run). It starts again on every boot and resumes from
  the checkpoints.
- Check the unit once: `sudo systemctl restart aimd`, wait a minute, then `tail ~/aimd/run/logs/launcher.log` should
  show `stop requested`, `8 runs unfinished`, then a fresh start from the same or later steps.
- On Spot, start the keep-alive where `gcloud` is logged in (a laptop, or the project node; it only calls gcloud):
  ```bash
  nohup bash d1/cloud/gcp_keepalive.sh naph-aimd $ZONE 10 > keepalive.log 2>&1 &
  ```
  It starts the VM whenever it is stopped and stops it for good once `~/aimd/run/ALL_DONE` exists. **Do not combine
  it with `--poweroff-when-done`** (the VM would be restarted and powered off again every 10 minutes).

**Progress:**
```bash
~/aimd/code/cloud/launch_aimd.sh status ~/aimd/run   # step, %, last QA record (T, Etot, <S2>, SCF cycles), engines
tail ~/aimd/run/logs/launcher.log                    # one block per boot: cores, starting steps, how it ended
grep -c "^=== run" ~/aimd/run/logs/naph_q1_900K_r0.log   # starts so far (1 + restarts and preemptions)
```
- Checkpoints every 20 fs (40 steps, 6–9 minutes) and at every stop.
- **A run crashed** (`launcher.log`: `exit 1`, `a run crashed`): read `~/aimd/run/logs/<run>.log`. The other runs go
  on until the next boot or restart. Fix the cause if it is clear and `sudo systemctl restart aimd`; otherwise
  report it.

**7. When the neutrals are done** (`status` shows the four `naph_q0_*` at 44,000), and only if step 4 showed 16 cores
helping the cations:
```bash
bash ~/aimd/code/cloud/setup_vm.sh service 8 16      # (add --poweroff-when-done first if on-demand)
```
The unit restarts; the cations save, then resume at 16 cores each on cores 0–63.

**8. Results back** (when `status` shows all 8 at 44,000 and `ALL_DONE` exists):
```bash
# on the VM:
tar czf ~/aimd_naph_cloud_done.tgz -C ~/aimd run
# on the project node:
gcloud compute scp $VM:~/aimd_naph_cloud_done.tgz /var/tmp/jjlee-pah/d1/ --zone=$ZONE
gcloud compute instances delete $VM --zone=$ZONE     # deletes the boot disk too
```
The analysis (spectra from the NVE dipoles, the B4 cation band-strength slope) runs on the project node. Report:
- the wall time and the number of restarts;
- seconds per step (from `status` over time, or the `seconds` field in each npz: time of the last session);
- any SCF with many cycles (QA column 5) and the cation ⟨S²⟩ (column 4);
- `engines` of each run.

## Bundle

| File | Size | sha256 |
|---|---|---|
| `/var/tmp/jjlee-pah/d1/aimd_naph_cloud.tgz` | 292 kB | `842b814be04bb114a7551fb5d8c2046060dedcec1d013900f9a43f462cebbea6` |

Built from commit `0cb8df1` (`aimd_cloud/code/VERSION`) and the Hangang inputs tarball (checkpoints as in the table
above). Rebuilding changes the hash (timestamps); give the new one to whoever installs.
