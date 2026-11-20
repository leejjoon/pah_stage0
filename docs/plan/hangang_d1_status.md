# Hangang D1 MLMD: live status and how to resume

**Read this first when picking the Hangang D1 work up in a new session.** The plan and its reasons are in
`d1_hangang_new_species.md`; this file says where things stand. Update it whenever a job is submitted or finishes,
commit and push (the project node pulls it). Last update: 2026-11-18 KST.

## Current state (2026-11-18): read this section first

No jobs running. All runs and analyses below are done, committed and pushed (branch `gh200-containers`).

**MD libraries of new neutral species** (PAHdb v4.00; Mai et al.'s model; production spectral convention qm=3,
unclipped, replica NVE T; topology-checked):

| T | Species with 4 intact replicas | List | Spectra |
| --- | --- | --- | --- |
| 300 K | 917 | `d1/new300_library_v1.csv` | `spectra/new300_v1`, `new300sp3_v1` |
| 600 K | 810 | `d1/new600_library_v1.csv` | `spectra/new600_v1` |
| 900 K | 549 (planar only; all also in 300 and 600 K) | `d1/new900_library_v1.csv` | `spectra/new900_v1` |

(`spectra/` = `/appsdata/t641a02/d1/spectra/`.) 1,200 K: none with this model (user's hot test); plan is linear
extrapolation with uncertainties, a retrained model later.

**Temperature rules for the emission model:** `d1/tdep_rules_v1.md` (one table; sources `d1/tdep_rules_v1.csv`,
`d1/perline_widths_v2.csv`). Analysis and caveats: `d1_hangang_new_species.md`, sections from "Band trends" on.

**PAHdb v4.00 problem:** 334 neutral pure C/H PAHs without a Kekule structure are listed as closed-shell singlets and
have unrealistic intensities (331 new in v4.00). Confirmed on C30H14 uid 10770 (closed-shell reproduces PAHdb; the
triplet is 18.7 kcal/mol lower and normal). Excluded from D1 (`d1/pahdb_v4_exclude_nonkekule_v1.csv`). Report, email
draft and explainer: `docs/reports/pahdb_v4_nonkekule/` (interactive page: claude.ai artifact CrGkvwyRmPDD18tYbovxSY,
private).

**Triplet recomputation:** method validated (`d1/triplet_freq.py`, `d1/uks_ir.py`; run with `CUPY_ACCELERATORS=cub`,
container D). Cost ~ nao^2.8; all 334 ~460 GPU-h (frequencies) or ~900 (with re-optimisation); nC < 90 ~120 / ~290.
**User decision 2026-11-18: recompute the 200 species with nC < 90 (with re-optimisation).** Driver
`d1/triplet_batch.py` + `d1/slurm_triplet_batch.sbatch`, list `d1/triplet_species_v1.csv`, output
`/appsdata/t641a02/d1/triplet_v1/` (`<uid>.json`, `logs/`, `failed/`). Test job 1301293 passed (4 species, 45 min; run
times 1.05-1.4x the estimates). **Production: job 1302424, `--array=0-3 --time=24:00:00`** (at most 4 job slots: the user
keeps one free). If tasks stop at the time limit with species left, resubmit the same command (array size must stay 4).

**Open decisions (user):** (2) replace
the T0b slice pair with C52H20 uid 10452; (3) send the PAHdb email; (4) delete `spectra/new300_v1_stale_noxyz` and the
`_qm0` spectra copies. Other items: hot-frames request (1,200-2,400 K, nC < 50) for the charged-MLFF session; tell that
session about the `squeue -j $SLURM_JOB_ID` time-limit bug in `d1/slurm_dft_seed.sbatch` (fixed in our sbatch,
`3a2b7fa`).

## Scope (decided by the user, 2026-11-11)

- Mai et al.'s published grid (1,704 species × 50/300/600 K) is **not** re-run.
- Hangang runs: (a) a 48-species **overlap** with Mai's grid at 300 + 600 K × 8 replicas, (b) ~1,000 **new
  species** from PAHdb v4.00 (100 N-PAH) at 300 + 600 K × 4 replicas, after (c) a 600 K **domain screen** of the
  new species; (d) the user also asked for the same 600 K screen of **all 1,704 of Mai's species**.
- 900/1200 K wait for the user's model-stability test on the project node. Not on Hangang yet.
- Licence: MLMD outputs are CC BY-NC-SA 4.0 (Mai's models); run outputs never go into git.

## Where everything is

| What | Path |
| --- | --- |
| Repository clone for jobs (submit from here) | `/appsdata/t641a02/pah_stage0` (branch `gh200-containers`; a second clone in `/home01/t641a02/work/pah_stage0` is for reading) |
| Container A | `/appsdata/t641a02/sif/A_jax.sif` (`module load apptainer/1.4.5`) |
| Mai's code and models | `/appsdata/t641a02/d1/mai2025/src/Supplementary_Data_Code`; published spectra inside `/appsdata/t641a02/d1/mai2025/Supplementary.zip` (`outputs/IR_txt_phdb_1704/IR_txt_{50,300,600}K_qm0.zip`) |
| PAHdb v4.00 | `/appsdata/t641a02/d1/pahdb/pahdb-complete-theoretical-v4.00.xml.gz` (sha256 = `t0/pahdb-v4.00.sha256`) |
| Species xyz | Mai: `/appsdata/t641a02/d1/XYZ_pahdb_1704`; overlap: `.../XYZ_overlap_v1` (48, copies of Mai's); new: `.../XYZ_new_v1` (1,000, named `<formula>_v4u<uid>.xyz`); new at 300 K: `.../XYZ_new300_v1`, plus `.../XYZ_new300sp3_v1` (the 18 sp3) (982 = new_v1 minus 18 sp3 species, list `d1/new300_species_v1.csv`); sp3: `.../XYZ_sp3_v1` (79 of Mai's species with sp3 C, copies; list `d1/sp3_species_v1.csv`) |
| Species lists (git) | `d1/overlap_species_v1.csv`, `d1/new_species_v1.csv`, `d1/cells_v1.csv` |
| Run directories | `/appsdata/t641a02/d1/runs/{calib01, overlap_v1, screen_new_v1, screen_mai_v1, screen300_sp3_v1, new300_v1}` (each has `plan.json`, `units/`, per-GPU logs) |
| Job logs | `mlmd_<jobid>_4294967294.log` in the `/appsdata` clone root (untracked, keep out of git) |

## Jobs

All on `normal`, 1 node, 4 GPUs, `d1/slurm_mlmd_run.sbatch A_jax.sif /appsdata/$USER/d1 <run dir>`.

| Job | Run | Time limit | State | Notes |
| --- | --- | --- | --- | --- |
| 1257396 | calib01 | 0:30 | COMPLETED | Calibration, results in `cluster_migration.md` |
| 1258657 | overlap_v1 | 0:30 | COMPLETED | Test: 4 units paused at checkpoints, 0 flags |
| 1258658 | screen_new_v1 | 0:30 | FAILED | sbatch bound Mai's xyz dir, not the plan's; fixed in `6a51b79` |
| 1258659 | screen_mai_v1 | 0:30 | COMPLETED | Test: 2 units done (180 systems, 0 flags, no topology change), 4 paused |
| 1258983 | screen_new_v1 | 0:30 | COMPLETED | Test of the fixed sbatch: 4 units done, 4 paused; 1 flagged (C52H16N2_v4u9326, 600 K, replica 2, in paused n070_u000) |
| 1259236 | overlap_v1 | 10:00 | COMPLETED | 6:52:20, 41/41 units done, 4 flagged; Full run, resumes from the test's checkpoints (~25 GPU-h) |
| 1259237 | screen_new_v1 | 6:00 | COMPLETED | 4:39:14, 112/112 units done, 18 flagged; Full run (~16 GPU-h) |
| 1265580_[0-1] | new300_v1 | 0:30 | task 0 COMPLETED, task 1 TIMEOUT | Array test: task 1 (whose job id equals the array id) read both tasks' time limits from `squeue`, fell back to `--hours 23.5` and was killed at the limit; checkpoints intact (8 units at 57–76 chunks). Sbatch fixed in `3a2b7fa` (own `<array>_<task>` id; abort if the limit is unreadable) |
| 1268196_[0-1] | new300_v1 | 0:30 | COMPLETED (both tasks) | Retest passed: both read `--hours 0.423`, 8 units resumed (now 124–169 of 400 chunks), 0 flagged, 0 errors. Sbatch `3a2b7fa` is proven for arrays |
| 1270903 | new300sp3_v1 | 1:30 | COMPLETED (1:10) | 300 K run of the 18 new sp3 species: 14 fail the screen and are dropped; 4 kept (C14H16_v4u339, C27H14_v4u69, C55H20_v4u318, C96H28_v4u190) |
| 1274694_[0-1] | new300_v1 | 5:00 | COMPLETED (2:55, 2:45) | Finishing run: 97/97 units done. Screen: 62 of 982 fail (56 non-planar), 920 pass |
| 1274456_[0-1] | new300_v1 | 0:30 | COMPLETED (both) | Test passed: cache in /scratch/t641a02/jax_cache (none in OUT), only listed units ran, 0 errors; 77/97 units done, 8 in progress. Sbatch `fa885b3` proven |
| 1268834_[0-3] | new300_v1 | 12:00 | tasks 1, 3 COMPLETED; 0, 2 FAILED (only the 4 quota crashes) | 300 K production, 982 new species (~118 GPU-h). Resubmit the same command with `--array=0-3` if it pauses |
| 1265105 | screen300_sp3_v1 | 4:00 | COMPLETED (3:08) | 300 K screen of 79 sp3 species: 28/28 units done; 22 of 79 species fail (13 H lost, 5 C–C broken, 4 H moved) |
| 1259238 | screen_mai_v1 | 8:00 | COMPLETED | 5:58:51, 150/150 units done, 2 flagged; Full run (~21 GPU-h) |

The sbatch at `6a51b79` is now a proven job (1258983 COMPLETED). Check the full runs with
`sacct -j 1259236,1259237,1259238 -X -o JobID,State,ExitCode,Elapsed` and
`grep -E 'done |paused|Traceback|Error|units done' mlmd_<jobid>_*.log` (at most once a minute). If one fails, read
its logs, tell the user the cause, fix, and ask before resubmitting; never resubmit in a loop.

## Next steps

1. **Full runs: done** (1259236–1259238 COMPLETED 2026-11-11, no unit paused, no edge overflow). Resume rule for
   future runs: if one ends "paused" at its time limit, resubmit that same command once. Never two jobs on one
   run dir. The commands, for reference:
   ```bash
   cd /appsdata/t641a02/pah_stage0
   sbatch --time=10:00:00 d1/slurm_mlmd_run.sbatch /appsdata/$USER/sif/A_jax.sif /appsdata/$USER/d1 /appsdata/$USER/d1/runs/overlap_v1     # ~25 GPU-h
   sbatch --time=06:00:00 d1/slurm_mlmd_run.sbatch /appsdata/$USER/sif/A_jax.sif /appsdata/$USER/d1 /appsdata/$USER/d1/runs/screen_new_v1  # ~16 GPU-h
   sbatch --time=08:00:00 d1/slurm_mlmd_run.sbatch /appsdata/$USER/sif/A_jax.sif /appsdata/$USER/d1 /appsdata/$USER/d1/runs/screen_mai_v1  # ~21 GPU-h
   ```
   If a run stops at its time limit before all units are done ("paused"), resubmit the same command once (it
   resumes); progress: `apptainer exec -B /appsdata A_jax.sif python d1/mlmd_run.py status <run dir>`.
2. **Screen results: done 2026-11-12** (`<run dir>/screen.csv`; summary in `d1_hangang_new_species.md`, step 2).
   screen_new_v1: 139 of 1,000 species fail (132 of them non-planar); screen_mai_v1: 365 of 1,704 fail (356
   non-planar). **On hold (user, 2026-11-12):** no dropping or refilling, no
   `XYZ_new_v2` and no new-species production until the user's high-T test on the project node decides what to do
   with the failing (mostly non-planar) species. Re-run the check with (login node is fine; numpy in container A, one BLAS thread):
   ```bash
   apptainer exec --env OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1 -B /appsdata /appsdata/$USER/sif/A_jax.sif \
     python -I d1/screen_check.py <xyz dir of the plan> <run dir> <run dir>/screen.csv
   ```
   For screen_new_v1: list failing species; drop them from the new set and refill their cells
   (`select_new_species.py` needs an exclusion option for that; not written yet), then write `XYZ_new_v2` and
   `d1/new_species_v2.csv`. For screen_mai_v1: the fraction of Mai's species with H shifts or other topology
   changes at 600 K, by class and planarity; it qualifies their published 600 K spectra and informs the
   user's high-T test. Record both in `d1_hangang_new_species.md`.
2b. **300 K screen of Mai's sp3 species** (after the overlap run showed C–C ring opening at 300 K): plan
   `runs/screen300_sp3_v1` made 2026-11-12 (79 species × 300 K × 4 replicas, 200 + 200 ps, 316 systems in 28
   units, ~8 GPU-h, ~2–2.5 h on one node). **Submitted 2026-11-12 as job 1265105** (4 h limit;
   1265095 was a mistyped submit, cancelled before it started). Proven script, one job:
   ```bash
   cd /appsdata/t641a02/pah_stage0
   sbatch --time=04:00:00 d1/slurm_mlmd_run.sbatch /appsdata/$USER/sif/A_jax.sif /appsdata/$USER/d1 /appsdata/$USER/d1/runs/screen300_sp3_v1
   ```
   Then `screen_check.py /appsdata/t641a02/d1/XYZ_sp3_v1 <run dir> <run dir>/screen.csv`.
   **Done 2026-11-13: 22 of 79 fail at 300 K** (details in the plan, step 2). **User decision 2026-11-13:** the 18
   new sp3 species get the 300 K production run too, and failures are dropped after the screen. Plan
   `runs/new300sp3_v1` (xyz `XYZ_new300sp3_v1`, list `d1/new300sp3_species_v1.csv`; 72 systems in 18 units,
   ~2.6 GPU-h, ~0.8 h on one node). **Job 1270903 COMPLETED; screened: 14 of 18 fail (dropped), 4 kept.** Command (one job):
   ```bash
   sbatch --time=01:30:00 d1/slurm_mlmd_run.sbatch /appsdata/$USER/sif/A_jax.sif /appsdata/$USER/d1 /appsdata/$USER/d1/runs/new300sp3_v1
   ```
3. **Overlap analysis.** Screen done 2026-11-12: 4 of 48 fail, two of them (C19H12_348, C17H14_342, sp3 ring
   opening) at 300 K too; see the plan, step 2. Still to do: spectra with `d1/mlmd_spectra.py`; compare with Mai's published
   300 and 600 K spectra per species (generalise `d1/port/mai_spectrum_compare.py`, which does naphthalene); seed
   scatter of band shares vs Nc decides the replica count of step 4 (default 4). Run `screen_check.py` on it too.
3b. **New-species production at 300 K (user, 2026-11-12: option A).** 982 new species without sp3 C × 300 K
   × 4 replicas, 200 + 200 ps: plan `runs/new300_v1` (3,928 systems, 97 units, ~118 GPU-h). The 18 sp3 new
   species wait for the 300 K sp3 screen; 600 K and higher wait for the user's high-T test. The array mode is
   new on Hangang, so a test first: `--array=0-1 --time=00:30:00` (2 jobs), **job 1265580: task 1 ignored the time limit (fixed in `3a2b7fa`); retest 1268196 passed 2026-11-12**; production submitted as **job 1268834** (`--array=0-3 --time=12:00:00`). Then `--array=0-3 --time=12:00:00`
   (4 jobs; per-GPU max ~9.2 h; at most 5 active jobs per user, so 4 while 1265105 is queued). Changing the
   array size between the test and production is safe because the two never run at once; keep the production
   size fixed for resubmits.
4. **New-species production** (a new plan, so ask the user first): `mlmd_run.py plan XYZ_new_v2 runs/new_v1
   --temps 300,600 --replicas 4` (default 200 + 200 ps); ~235 GPU-h, ~59 node-hours; a 5-task array
   (`--array=0-4`) of 6–12 h tasks, resubmitted with the same array size until done.

## Finishing new300_v1 (2026-11-13 19:30)

After 1268834: 74/97 units done (83 % of the work), 4 in progress (n120_u001, n086_u000, n086_u001 at 360–384 of
400 chunks; n074_u000 at 0), 19 not started; ~20 GPU-h left. `new300_v1/jax_cache` deleted. Round-robin over the
whole plan would leave 12 of 16 GPUs idle, so pass the remaining units with `--units` (same list for every task,
so no unit is taken twice): 2 tasks, per-GPU max ~3.3 h. **Test 1274456 passed 2026-11-13; 5 h run 1274694 COMPLETED: new300_v1 is done (97/97 units); screened 2026-11-14, 62 of 982 fail.** The sbatch changed (cache on /scratch), so first a test
with the same list and array size, then the rest:
```bash
U=n120_u001,n086_u000,n086_u001,n072_u000,n074_u000,n082_u000,n064_u000,n166_u000,n050_u001,n054_u002,n126_u000,n046_u000,n128_u000,n186_u000,n088_u000,n112_u000,n110_u000,n154_u000,n042_u000,n182_u000,n170_u000,n140_u000,n124_u000
sbatch --array=0-1 --time=00:30:00 d1/slurm_mlmd_run.sbatch /appsdata/$USER/sif/A_jax.sif /appsdata/$USER/d1 /appsdata/$USER/d1/runs/new300_v1 --units $U   # test
sbatch --array=0-1 --time=05:00:00 d1/slurm_mlmd_run.sbatch /appsdata/$USER/sif/A_jax.sif /appsdata/$USER/d1 /appsdata/$USER/d1/runs/new300_v1 --units $U   # rest
```

## 300 K library from the new species (2026-11-14)

`d1/new300_library_v1.csv`: 917 species with 4 clean replicas at 300 K (production-convention spectra in
`spectra/new300_v1` and `spectra/new300sp3_v1`). Of the 1,000: 76 fail the screen (62 non-sp3, 14 sp3), 6 dropped by the dipole check, 1
by the bond-graph check (C96H28_v4u190). Details in
`d1_hangang_new_species.md`, step 2.

## High-T plan (user's hot-test summary and decision, 2026-11-14)

Hot test (project node): 1200 K unusable with this model; 900 K usable only with a per-system integrity filter;
600 K 105/132 replicas intact (all 27 changes in the 9 cold-check species). The Hangang screens (2,704 species)
show more failures at 600 K (non-planar), so 600 K also gets the filter. Steps (ask before each submit):
1. **Driver: per-chunk bond-graph check** (`mlmd_run.py`, `flags["bond_change_chunk"]`, criterion of
   `mlmd_integrity.py`; catches hop-and-return too). Test plan `runs/test_bond_v1`: 600 K, 10 + 10 ps, 2 replicas,
   positives C10H16_v4u333, C29H17_v4u2132, C32H18_v4u1530; negatives C46H18_v4u6921, C82H24_v4u561,
   C100H26_v4u10772. Test job (one job): `sbatch --time=00:30:00 d1/slurm_mlmd_run.sbatch
   /appsdata/$USER/sif/A_jax.sif /appsdata/$USER/d1 /appsdata/$USER/d1/runs/test_bond_v1`. **Job 1280872 COMPLETED 2026-11-15 (7:47): passed.** Positives flagged in
   6/6 systems (all at chunk 0: C29H17/C32H18 have two C–H stretched 1.08 → 1.30–1.52 Å within 1 ps, C10H16 falls
   apart), negatives 0/6, identical to `mlmd_integrity.py` on the end states. Driver `126aa42` proven.
2. 600 K production: the 857 library species that passed the 600 K screen (~109 GPU-h). Plan `runs/new600_v1`
   (xyz `XYZ_new600_v1`, list `d1/new600_species_v1.csv`; 3,428 systems, 93 units); 4-task array, per-GPU max
   ~8.6 h: `sbatch --array=0-3 --time=12:00:00 d1/slurm_mlmd_run.sbatch /appsdata/$USER/sif/A_jax.sif
   /appsdata/$USER/d1 /appsdata/$USER/d1/runs/new600_v1`. **Submitted 2026-11-15 as job 1280975** (running on 4 nodes).
3. 900 K pilot: 88 of the 857 (random within Nc bins <30, 30–50, 50–80, 80–120, ≥120; 15 planar + 5 non-planar
   per bin, only 3 planar below 30 C), full 200 + 200 ps, 4 replicas. Plan `runs/pilot900_v1` (xyz
   `XYZ_pilot900_v1`, list `d1/pilot900_species_v1.csv`; 352 systems, 48 units, ~16 GPU-h, per-GPU max ~4.3 h):
   `sbatch --time=06:00:00 d1/slurm_mlmd_run.sbatch /appsdata/$USER/sif/A_jax.sif /appsdata/$USER/d1
   /appsdata/$USER/d1/runs/pilot900_v1`. **Submitted 2026-11-15 as job 1281196.** Decide 900 K production from its yield.
4. 1200 K: none with this model (user to choose: extrapolate, empirical widths, or a retrained model).

## 900 K production (user's go, 2026-11-15 17:5x; node limit lifted: use up to 5)

**Done 2026-11-16 ~04:00: 85/85 units; library `d1/new900_library_v1.csv` (549 species).** Libraries of the new
species: 300 K 917, 600 K 810, 900 K 549; 549 in all three.

779 planar species of the 857 (list `d1/new900_species_v1.csv`, xyz `XYZ_new900_v1`), 900 K, 4 replicas, 200 + 200 ps,
per-chunk topology check; plan `runs/new900_v1` (3,116 systems, 85 units, ~99 GPU-h). Split into two disjoint unit
lists so all 5 job slots are used: `runs/new900_v1/units_A.txt` (36 largest units, ~73 GPU-h) as **job 1282825**
(`--array=0-2 --time=10:00:00 ... --units $(cat units_A.txt)`), and `units_B.txt` (49 units, ~26 GPU-h) as **job 1284012** (submitted 23:3x 2026-11-15, after 600 K job 1281347 COMPLETED, 93/93 units): `sbatch --array=0-1 --time=06:00:00
d1/slurm_mlmd_run.sbatch /appsdata/$USER/sif/A_jax.sif /appsdata/$USER/d1 /appsdata/$USER/d1/runs/new900_v1
--units $(cat /appsdata/$USER/d1/runs/new900_v1/units_B.txt)`. Resubmit each part only with its own list and
array size. Spectra must use only systems with `topo_change_chunk == -1`.

## Per-chunk check, second version (2026-11-15 11:xx)

The first per-chunk check (bond graph, 1.2 x covalent radii) counts a C-H bond as lost at 1.284 A, which hot C-H
vibrations reach: in the running jobs 1280975 (600 K) and 1281196 (900 K pilot) it flagged 244 of 1,244 and 56 of
56 started systems, of which only 8 and 5 were still changed at their last checkpoint. Replaced by the
screen_check.py criteria (H hop = nearest heavy atom changed; H lost > 2.0 A; C-C broken > 2.0 A, formed < 1.6 A),
`flags["topo_change_chunk"]` and `flags["topo_change_chunks"]`. On the current checkpoints: 2 of 1,244 (600 K)
and 3 of 56 (900 K) systems, all real events. Both jobs are still in NVT (the spectra use NVE only), so stopping
them now and resuming with the fixed driver loses no NVE coverage. Test plan `runs/test_topo_v1` (same 6 species,
900 K, 20 + 20 ps, 2 replicas). 1280975 and 1281196 cancelled 2026-11-15 ~11:15 (checkpoints
intact; 600 K: 93 units left, ~81 GPU-h; pilot: 47 units left, ~13 GPU-h). Test job 1281309 COMPLETED and passed (900 K: positives 6/6 flagged, in a changed state 30–40 of 40
chunks; negatives 0/6 over 40 chunks). Resubmitted 2026-11-15 ~11:50: **600 K job 1281347_[0-1], pilot job 1281348**. Pilot COMPLETED 16:20 (47/48 units; n048_u000 left, 4 systems); results in
`d1_hangang_new_species.md`, "900 K pilot": planar 12 % of systems changed, non-planar 48 %. User: at
most 3 nodes at a time for the resubmits: 600 K `--array=0-1 --time=14:00:00` (per-GPU max ~11.4 h) and the pilot
on 1 node `--time=05:00:00` (~3.5 h).

## Known facts and pitfalls

- **Shared clone, branch switches (2026-11-14):** at 10:05–10:07 another session checked out `feat-gpu-gh200` in
  `/appsdata/t641a02/pah_stage0`; a spectra run started then used that branch's `mlmd_spectra.py` (qm=3, no
  clip, measured T; no `qa.csv`), writing `spectra/new300_v1`. The rerun with this branch's tool is
  `spectra/new300_v1_qm0` (qm=0 + `qa.csv`). The branches have diverged (33 vs 36 commits); the production spectral
  convention (committee review B1) lives on `feat-gpu-gh200`. **Merged into `gh200-containers` 2026-11-14 (`88e61df`):**
  `mlmd_spectra.py` now defaults to the production convention (qm=3, no clip, replica NVE temperature) and keeps
  `--units`, per-unit loading and `qa.csv`; Mai's convention is `--qm 0 --clip --tnom`. Do not switch branches in
  this clone; use a separate clone or `git worktree` for other branches.

- **/appsdata file-count quota (2026-11-13):** group bs0358 is at 4,977,743 of 5,000,000 files (`lfs quota -h -g
  bs0358 /appsdata`); t641a02 owns 250,179 of them. JAX's persistent compilation cache writes then fail with
  "Disk quota exceeded" (a harmless UserWarning: it only recompiles). If the group reaches the limit, checkpoint
  and dipole writes fail too and the run crashes. Check the quota before every submit. The D1 run directories
  were thought to hold few files, but the per-run `jax_cache` dirs (XLA per-fusion autotune cache) held ~243k of
  the 250k. At the limit, 4 of the 16 GPU processes of job 1268834 crashed writing checkpoints (atomic, so no
  damage). Fix applied 2026-11-13 (user's choice): all `runs/*/jax_cache` deleted, t641a02 down to 6,800 files,
  group to 4,734,364. The other members then cleaned up: group at 1,218,999 files at 14:34. **Fix (2026-11-13):**
  `mlmd_run.py run --cache-dir`, and the sbatch now puts the cache in `/scratch/$USER/jax_cache`, shared by all
  runs (/scratch has its own group file quota; purge only costs a recompile). Jobs submitted before this commit
  still write `OUT/jax_cache`: delete it after they end. The changed sbatch is a new job again: one short test
  (`--array=0-1 --time=00:30:00` on new300_v1, after 1268834 ends) before the production resubmit.

- `sbatch --test-only` start times are worst case; short jobs backfill within minutes (half of `normal` was
  `planned` on 2026-11-11). `squeue` shows only our jobs.
- Running container-A Python on a login node: pass `--env OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1`, or OpenBLAS
  hits the login node's process limit.
- Mai's model shows 1,2-H shifts at 600 K in non-planar species (C28H16_2170, C28H16_2173 in the calibration run).
- Hangang GH200 speed: 7–8 % slower than the project node; 1.22 s per atom per system for 200 + 200 ps, plus
  ~250 s per unit.
- `/home01/t641a02/work/pah_stage0/core` (an 18 GB sparse core dump from 2026-11-10) is untracked; the user has not
  decided whether to delete it.

### Triplet recomputation finished (2026-11-19, job 1302424)
All 200 non-Kekule species with nC < 90 (198 triplets, 2 quintets) done in 16.8-18.7 h on 4 nodes, 0 failures. Results in
/appsdata/t641a02/d1/triplet_v1/<uid>.json; summary `d1/triplet_summary.py` -> `d1/triplet_summary_v1/` (csv, png).
Checks: no imaginary modes; <S^2> exceeds the exact value by median 0.15 (max 0.33); optimisation lowers E by < 0.8 kcal/mol.
Total intensity per C: PAHdb listed 136 km/mol (median), recomputed 18, normal PAHdb neutrals 24 -> PAHdb is 7x too
intense (16-84 %: 4.7-14.8x). C-H stretch share: listed 0.10, recomputed 0.48, normal 0.40. The recomputed spectra fall
on the normal-PAH trend. Not yet in the D1 libraries (MLMD needs the open-shell state; Mai's model is closed-shell).
- 2026-11-19: user decision: use the 200 recomputed spectra as harmonic data, marked as **our own** (not PAHdb):
  `d1/own_spectra/` (species id `<formula>_v4u<uid>_own`, `source=own`; README), built by `d1/own_spectra.py`;
  `d1/pahdb_v4_replacements_v1.csv` maps all 334 affected uids to replace (200) / exclude (134, nC >= 90).
  PAHdb v4.00's B3LYP/6-31G* scale factors are per mode: 0.9794 < 1111, 0.9691 1111-1800, 0.9597 above (earlier
  scripts used 0.975/0.979/0.964; fixed in triplet_summary.py and the report's singlet_vs_triplet.py, figure redone).
- MLMD for these needs a high-spin MLFF (Mai's model is closed-shell). Cost basis: one UKS energy+force point took a
  median 112 / 172 / 236 / 339 s for 30-45 / 45-60 / 60-75 / 75-90 C (optimisation steps, warm start). Same pipeline
  as the charged-MLFF plan (mlff_sample.py --mult 3, dft_seed_run.py, mlff_qa.py). Not started.
- 2026-11-20: user decision: 10 non-Kekule triplets (+ PAHdb cations) go to the charged-MLFF seed set.
  `d1/highspin_seed_species.py` -> `d1/highspin_seed_v1/`; handoff `docs/plan/handoff_highspin_to_charged_mlff.md`.
  Found: their seed pairs C60H20_10225 / C72H22_9911 are non-Kekule; the triplet lies 7.2 / 7.4 mHa below their
  broken-symmetry q0 label at f000 (same PAHdb geometry). Their decision whether to relabel.
- 2026-11-20: high-spin seed DFT run here on one node (user decision): `d1/slurm_highspin_seed.sbatch` (xTB sampling
  with feat-gpu-gh200's mlff_sample.py from the detached worktree /appsdata/t641a02/pah_stage0_hs at c93e4fc; xTB env
  /appsdata/t641a02/envs/t641a02-xtb_aarch64.tgz: tblite 0.8.0, ASE 3.29; g16_io.py write; dft_seed_run.py --stab-all).
  Test 1316132 COMPLETED (12 frames C30H14_10770, all converged/stable, q0 f000 = our E_opt). Production 1316386
  -> /appsdata/t641a02/d1/highspin_seed_run (~530 frames, est. 6-9 h). After it: `dft_seed_run.py collect`, pack for
  the project node.
