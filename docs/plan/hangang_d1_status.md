# Hangang D1 MLMD: live status and how to resume

**Read this first when picking the Hangang D1 work up in a new session.** The plan and its reasons are in
`d1_hangang_new_species.md`; this file says where things stand. Update it whenever a job is submitted or finishes,
commit and push (the project node pulls it). Last update: 2026-11-12 KST.

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

## Known facts and pitfalls

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
