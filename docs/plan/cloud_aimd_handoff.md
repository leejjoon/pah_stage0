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

## Result and Gate B4 decision (2026-11-24)

**What ran (2026-11-12..2026-11-24):** naphthalene neutral and cation, B3LYP(VWN3)/6-31G\* 6D, PySCF on CPU and
GPU: 8 main runs (2 ps NVT + 20 ps NVE; project node, GCP, gmunu, wasp12) and 14 short replicas with saved frames
(2 ps NVT + 5 ps NVE; wasp12): 22 runs. Data: `aimd_naph_done-2026-11-21.tgz`,
`aimd_naph_short-2026-11-23.tgz`, `aimd_naph_short_q0-2026-11-24.tgz` (manifests in `data/manifests/`). Code at tag
`aimd-naph-b3p1` once merged (branch `cloud-aimd`).

**Archive (since 2026-11-25):** primary copy in `gh200:~/projects/pah-data/<name>-<tag>.tgz` (directory mode 700,
files 400, a `.sha256` sidecar each and entries in its `SHA256SUMS`), mirrored off the server in
`z2mini:~/projects/pah-data-mirror/` (same layout). Both copies passed `sha256sum -c` on 2026-11-25 and match the
manifests' `sha256`. All three were first pushed to myhub `pah-data` (2026-11-21..24); myhub is deprecated and may go
down, so its references below and the OCI digests in the manifests (`provenance`) are history, not a location. The
wasp12 and gmunu `~/aimd/run` originals are no longer the only second copy; removing them needs the user's
go-ahead.

**Finding 1, method:** at 300 K the NVE band strengths of a replica are set by the energy its few IR-active modes
happen to hold, which stays frozen for the whole run (20 ps tested). Raw replica fits therefore scatter by up to
±15 % per 100 K. Normalising each run's strength by its own mode energies (`d1/mode_energy.py`: strength / Σ I_k·E_k/kT)
removes this; it needs saved frames (`aimd_pyscf.py --frames-fs`). See "Extra cation replicas" below.

**Finding 2, the numbers** (in-plane CC band 1000–1700 cm⁻¹, strength per unit mode energy, % per 100 K, ± 1 s.e.;
runs with frames only; `mode_energy.py analyse`, outputs in the gh200 copies under `*/analysis/modes/`):

| | 300 K runs | 900 K runs | Slope |
|---|---|---|---|
| Cation | 0.989 ± 0.017 (6) | 0.884 ± 0.014 (2) | **−1.72 ± 0.34** |
| Neutral | 1.198 ± 0.073 (4) | 1.183 ± 0.082 (2) | **−0.24 ± 1.74** |
| Cation − neutral | | | **−1.5 ± 1.8** |

Raw fits over all 22 runs, for comparison: cation −2.6 %, neutral +3.7 %, difference −6.2 % (frozen-mode scatter;
not used). The neutral is the noisy side: its in-plane band is weak (harmonic 33.5 km/mol against 410 for the
cation) and only partly harmonic (measured/predicted 1.2–1.6 in 1300–1700 cm⁻¹), so the normalisation is weaker there.

**Gate B4 decision:** the rule (`docs/plan/b4_diagnostic_impact.md`, "Reading against the gate") is that the
charge-dependent correction matters if |cation − neutral| ≳ 2 % per 100 K and does not if ≲ 1.5 %. The DFT result is
−1.5 ± 1.8 %: the central value sits on the "does not matter" edge, but the uncertainty covers both outcomes (about
45 % probability of ≤ 1.5 %, 40 % of ≥ 2 %). **B3 part one does not decide B4.** It does establish that the cation CC
slope is small and negative (−1.7 ± 0.3 %, against xTB's −0.9 % and the +2.2 % MLMD neutral mean), so the deciding
uncertainty is now the neutral slope. As the B4 document staged it, the gate passes to step (b), the pilot models
on the seed set, with these 22 runs as their DFT reference. Narrowing the neutral to ±0.5 % with DFT MD would need
roughly 12× the neutral replicas (~70 short runs, about two weeks of wasp12); not proposed.

**For downstream MLMD (`feat-gpu-gh200`):** below ~35 C at 300 K, use many independent short NVE replicas from
canonical (Langevin) snapshots, or save decimated frames for the `mode_energy.py` normalisation; a few long replicas
are not enough.

## Done: all 8 runs at 44,000 steps (2026-11-21)

The neutrals finished on gmunu on 2026-11-20 at 18:16 (job 29751, launcher exit 0, no resubmit). The cations finished
on wasp12 on 2026-11-21 at 08:19 (unit `aimd` inactive). Both machines wrote `ALL_DONE`.

**Results on the project node:** `/var/tmp/jjlee-pah/d1/aimd_naph_done/` and `aimd_naph_done.tgz` (3.9 MB, sha256
`442d9aa55326feab940777b39b12dde1b369d86821ec168e63a39d5ef359a8d6`), archived 2026-11-21 as
`oras://myhub.ksc.re.kr/pah-data/aimd_naph_done:2026-11-21` (now history; primary archive since 2026-11-25
`gh200:~/projects/pah-data/aimd_naph_done-2026-11-21.tgz`, private; manifest `data/manifests/aimd_naph_done.yaml`). The layout follows the Hangang inputs tarball:
the 8 `naph_q*_*K_r*.npz` and the 2 xyz files, plus `SHA256SUMS`. The logs are in `logs/gmunu/` (with the GCP segment
in `gcp/`) and `logs/wasp12/` (with the cations' earlier gmunu logs in `gmunu/`). The hashes match the copies on
gmunu and wasp12. Every npz is at 44,000 of 44,000 steps, with no non-finite values and a full NVE dipole series
(20,000 × 3, with no empty rows).
- Highest SCF cycle count over the whole run: 9 for the neutrals, 13–15 for the cations.
- Highest cation ⟨S²⟩: 0.767–0.772.

These copies were left out: gmunu's `moved_to_wasp12/` and wasp12's `bundle_original/` (both stale), the lock files,
and gmunu's `naph_q1_900K_r0.step5000.npz.bak` (forensic only). wasp12 was cleaned on 2026-11-21: the unit and
`~/aimd*` were removed (linger left on). The empty GCP project `naph-aimd-2611` still needs deleting, which needs a
fresh `gcloud auth login`.

**First analysis (2026-11-21, `d1/aimd_slopes.py`; output in `aimd_naph_done/analysis/`):** the convention is that of
the MLMD/xTB class slopes `b4_impact.py` uses (qm=3, unclipped, replica NVE-mean T; oop 600–1000, in-plane 1000–1700,
CH stretch 2700–3300 cm⁻¹; linear in T; the slope is a fraction of the fitted 300 K value per 100 K).
- Cation in-plane slope: −8.2 % per 100 K over all 4 replicas, range −13.0 to +2.0 % over the 4 pairings of one 300 K
  and one 900 K replica.
- Neutral in-plane slope: +4.5 % per 100 K (range +0.3 to +7.8 %).
- **The gate's number is not settled.** It hinges on one replica:
  - PAHdb harmonic cation in-plane strength (uid 359): 414 km/mol.
  - Cation 300 K: r0 313, r1 **789** km/mol.
  - Cation 900 K: 341 and 339 km/mol.
  - Without 300 K r1, the cation in-plane slope is about +2 % per 100 K, close to the neutral MLMD value (+2.2–2.5 %),
    so the correction would not matter. With r1 included, the cation-minus-neutral difference is about −13 % per
    100 K, far past the 2 % threshold.
- r1's excess is all in the 1000–1300 cm⁻¹ band and is steady across four 5 ps blocks (630–820 against 134–215 km/mol
  for r0). Either energy is trapped in an IR-active mode (a cold 18-atom molecule exchanges energy slowly), or r1 is
  on a different SCF solution. Its ⟨S²⟩ and energy look normal. The checkpoints keep no frames, so this cannot be
  told apart afterwards.
- Neutral replicas also scatter: NVE T 246/284 K and 725/950 K, oop strengths ±35 % about harmonic. Two replicas per
  point are too few for slopes at the 1–2 % per 100 K level.
- What would settle it: more cation 300 K (and 900 K) replicas. Short ones suffice, since r1's excess is visible in
  every 5 ps block, for example 8 × (2 ps NVT + 5 ps NVE), about 35 h each at 32 cores. Saving frames would allow a
  normal-mode energy check. The alternative is the pilot-model route (b) of `b4_diagnostic_impact.md`.

**Extra cation replicas: done 2026-11-23 02:50, archived** (commit `8ba78b0`):
- Runs: `naph_q1_300K_r2`…`r7` and `naph_q1_900K_r2`, `r3` on wasp12 (systemd user unit `aimd`, 32 cores each),
  2 ps NVT plus 5 ps NVE (14,000 steps), NVE positions and velocities every 5 fs (`frames_pos`, `frames_vel`), seeds
  T·10 + 2r + 1. Started 2026-11-21 11:51; all at 14,000 steps by 2026-11-23 02:50 (`ALL_DONE`). Logs clean (only an
  ASE deprecation warning); max SCF 15, ⟨S²⟩ ≤ 0.776, NVE drift −1.4 to +0.8 meV/ps.
- Results: `gh200:/var/tmp/jjlee-pah/d1/aimd_naph_short/` and `aimd_naph_short.tgz` (7.4 MB, sha256 `571c10a0…`),
  hashes checked on wasp12, z2mini and gh200. Archived to `oras://myhub.ksc.re.kr/pah-data/aimd_naph_short:2026-11-23`
  (pull-back hash matches; anonymous pull refused); manifest `data/manifests/aimd_naph_short.yaml`. Since
  2026-11-25 the primary archive is `gh200:~/projects/pah-data/aimd_naph_short-2026-11-23.tgz` (z2mini mirror); myhub
  is history.
- Slopes over all 16 runs (`aimd_slopes.py`, commit `fc6522c`; `analysis/aimd_slopes_all.{txt,json}` in the gh200
  copy), fraction of the 300 K value per 100 K: cation in-plane −2.6 % (pairings −13 to +17 %, rough 1σ ±3.5 %),
  neutral in-plane +4.5 % (+0.3 to +7.8 %), cation minus neutral −7 %. **The B4 gate is still not decided.**
- Why: the 300 K cation in-plane strength varies 206–789 km/mol between replicas (900 K: 259–346). Nearly all of it is
  in 1000–1300 cm⁻¹ (56–659 km/mol; 1300–1700 is 55–185) and each replica keeps its level through every 2 ps block
  (r1 ≈ 800, r6 ≈ 600), so it is a persistent per-replica state, not noise. Untested guess: a different degree of
  charge (hole) localisation, i.e. symmetry breaking, frozen in at 300 K and averaged at 900 K; the high-strength
  runs tend to have a larger mean dipole (zero for D2h). Next check: single points on the r2–r7 frames, charge
  asymmetry against the 1000–1300 cm⁻¹ strength.
- **Electronic-state check (2026-11-23, wasp12): ruled out.** `d1/frame_sp.py` ran cold-start single points on
  500 frames of each of r2–r7 and 900 K r2, r3 (every 10 fs; 4,000 in all), with UKS stability tests (Stable=Opt)
  on 25 per run (`d1/frame_sp_summary.py`; output in the gh200 copy under `aimd_naph_short/analysis/frame_sp/`).
  Every SCF converged without Newton; every cold-start dipole matches the MD's at the same step to ≤ 4×10⁻⁴ e·Å;
  all 200 stability tests stable, no re-optimisation; ⟨S²⟩ 0.7633–0.7637 at 300 K for every replica. So the MD stayed
  on the stable ground-state solution and the replicas do not differ in electronic state. Across the six 300 K
  replicas the 1000–1300 cm⁻¹ strength tracks the rms long-axis dipole (r = 0.94; ring A − B spin asymmetry 0.71),
  which only restates that this band is charge moving between the rings.
- **Frozen mode energies: confirmed (2026-11-23).** `d1/mode_energy.py`: geometry at the MD level (the PAHdb
  geometry moves 1e-4 Å), analytic Hessian, IR intensity per mode by finite differences (harmonic in-plane/oop:
  cation 410/95 km/mol, PAHdb 414; neutral 33.5/97.9, PAHdb 34/98); then each run's energy in every mode from the
  frames, and the predicted strength = Σ I_k·E_k/kT over the band. For the six 300 K replicas with frames the
  prediction matches the measured strength in every band (in-plane measured/predicted 0.93–1.03; 1000–1300 cm⁻¹
  0.99–1.08; e.g. r3 56/57, r6 521/481 km/mol). Single modes sit at 0.14–2.8 kT and stay there through the NVE: the
  300 K spread is the canonical spread of a few mode energies, not noise or electronic state. Runs without frames
  (r0, r1) can only be estimated from their final snapshot, which works poorly. Output in the gh200 copy under
  `aimd_naph_short/analysis/modes/`.
- **Normalised slope (cation):** in-plane strength per unit mode energy (measured/predicted) is 0.989 ± 0.016 at
  297 K (6 runs) and 0.884 ± 0.014 at 909 K (2 runs): **−1.7 ± 0.4 % per 100 K**, about ten times tighter than the raw
  replica fit. 1000–1300 cm⁻¹ alone: −2.3 %/100 K. The neutral needs the same treatment before the gate can be read:
  its main runs saved no frames and its raw +4.5 % has the same problem. Proposed: short neutral replicas with
  frames (300 K ×4, 900 K ×2; 2 ps NVT + 5 ps NVE, `FRAMES_FS=5`) on wasp12.
- **Neutral short replicas: done 2026-11-24 14:07.** `naph_q0_300K_r2`…`r5` and `naph_q0_900K_r2`, `r3` on
  wasp12 (user unit `aimd`, 32 cores each, 6.8–7.0 s/step), 2 ps NVT + 5 ps NVE, frames every 5 fs, seeds T·10 + 2r;
  logs clean, 8–9 SCF cycles, NVE energy conserved. Results: `gh200:/var/tmp/jjlee-pah/d1/aimd_naph_short_q0/` and
  `aimd_naph_short_q0.tgz` (5.5 MB, sha256 `2addd067…`), hashes checked on wasp12, z2mini and gh200. Archived 2026-11-24 22:20 to
  `oras://myhub.ksc.re.kr/pah-data/aimd_naph_short_q0:2026-11-24` (pulled back, sha256 matches; anonymous pull
  refused), manifest `data/manifests/aimd_naph_short_q0.yaml`. The push waited 8 h: myhub first refused connections
  (from ~14:20), then served its new certificate without the Sectigo DV R36 intermediate; pushed with `--ca-file`
  holding the system bundle plus that intermediate (manifest `provenance`). Since 2026-11-25 the primary archive is
  `gh200:~/projects/pah-data/aimd_naph_short_q0-2026-11-24.tgz` (z2mini mirror); myhub is history. Normalised neutral slopes in the B4 section at the top.
- The narrow 11.2-type windows go negative in the 5 ps runs (unclipped truncated-ACF estimator around sharp lines);
  use only the broad bands for them.

## Where each run was (last move 2026-11-18 17:24)

| Runs | Machine | Owner since | How it runs |
|---|---|---|---|
| the 4 neutrals | gmunu, Slurm job `aimd-q0` | 2026-11-15 | below |
| the 4 cations | **wasp12** | 2026-11-18 (from gmunu, steps 19,813–22,747) | systemd user unit, next section |

Each checkpoint lives in one place only. gmunu's cation checkpoints were moved into `~/aimd/run/moved_to_wasp12/`
(stale, never resume them); wasp12's `~/aimd/run/bundle_original/` holds the stale bundle copies.

## wasp12: the 4 cations, since 2026-11-18

wasp12: 2× EPYC 9965 (Zen 5c), 384 cores and 768 threads, 754 GB RAM, Rocky Linux 9, no Slurm. It is shared with
another user, so these runs use 128 cores. `ssh wasp12` goes through the ProxyJump in `~/.ssh/config`. Cations ran
at 9.6–9.9 s/step with 8 runs × 32 cores going at once (gmunu: 15–18), so about 2.6 days to finish, around
**Nov 21**.

- **Where:** `~/aimd` on wasp12: the same layout as on gmunu. The run logs started on gmunu are in `run/logs/gmunu/`.
- **Unit:** the systemd user unit `aimd` (`~/.config/systemd/user/aimd.service`), written by
  `RUNS=naph_q1_300K_r0,naph_q1_300K_r1,naph_q1_900K_r0,naph_q1_900K_r1 AIMD_TAG=wasp12 bash ~/aimd/code/cloud/setup_vm.sh service --user 32 32`.
  Linger is on (`loginctl enable-linger`), so the unit runs without a login and starts again after a reboot. A
  restart was tested: the runs saved and resumed at the same step.
- **Progress:** `~/aimd/code/cloud/launch_aimd.sh status ~/aimd/run`, `tail ~/aimd/run/logs/launcher.log`.
- **Stop:** `systemctl --user stop aimd` (every run saves). **Start:** `systemctl --user start aimd`.
- **When a run crashes**, the launcher exits 1 and the unit stays down until it is started again. Read
  `run/logs/<run>.log` first.
- **Install notes for hosts with many CPUs:** `setup_vm.sh` now passes `--no-pyc`. micromamba's byte-compile started
  768 workers and deadlocked. The water check is also capped at 8 threads; uncapped, it fails on 768.

## gmunu (KASI): the neutrals, since 2026-11-15 (all 8 runs until 2026-11-18)

**The 8 runs were on gmunu (`gmunu.kasi.re.kr`, port 7774 in `~/.ssh/config`), not in the cloud.** Do not run these
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
- **2026-11-18:** the first 3-day resubmit worked for all three jobs (16:03; the neutrals are now job 29751). At 17:23
  the two cation jobs were cancelled (they saved and did not resubmit), and the 4 cations moved to wasp12 (above).
  Do not resubmit `aimd-q1z3` or `aimd-q1z2` on gmunu.
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
