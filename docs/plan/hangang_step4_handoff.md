# Handoff: D1 MLMD calibration job on Hangang (step 4)

Written 2026-11-11 on the project node, for running on KISTI Hangang (account `t641a02`). Background and results
so far: `docs/plan/cluster_migration.md`, sections "Step 3 driver" and "Step 4 calibration".

## Where things stand

- The production driver `d1/mlmd_run.py` (plan / run / status) and `d1/mlmd_spectra.py` are tested on the project
  node. The driver vmaps the NeuralIL descriptor over atoms (the authors' code loops with `lax.map`), which is
  2–3× faster; every unit checks forces and EPNN charges against the authors' original path (gates 1e-3 eV/Å and
  1e-4 e; measured ≤ 1.3e-5 eV/Å and ≤ 6e-6 e).
- A full 200 + 200 ps naphthalene unit reproduced the step 2b spectra (strong bands within 2.5 cm⁻¹ of the
  published 300 K spectrum). Peak GPU memory for the largest unit: 28 GB.
- Estimate for the full plan (1,704 species × 50/300/600 K × 4 replicas, 269 units): ~420 GPU-h on the project
  GH200, ~105 Hangang node-hours. **This job measures the speed on Hangang** and is the first run of
  `d1/slurm_mlmd_run.sbatch`.

## The calibration job

Inputs already on Hangang: `/appsdata/t641a02/d1/mai2025/src/Supplementary_Data_Code` and
`/appsdata/t641a02/d1/XYZ_pahdb_1704/`; image `/appsdata/$USER/sif/A_jax.sif`. Run everything from the repository
root (the sbatch binds `d1/` from the submit directory).

**1. Update the repository**
```bash
cd /appsdata/$USER/pah_stage0
git checkout gh200-containers && git pull
git log --oneline -1          # the commit that added this note
module load apptainer/1.4.5
```

**2. Make a short plan** (2 ps NVT + 2 ps NVE; same unit ids as production) in a fresh directory. Login node is
fine; it only reads xyz headers.
```bash
OUT=/appsdata/$USER/d1/runs/calib01
mkdir -p $OUT
apptainer exec -B /appsdata /appsdata/$USER/sif/A_jax.sif \
  python -I d1/mlmd_run.py plan /appsdata/$USER/d1/XYZ_pahdb_1704 $OUT --nvt 4000 --nve 4000
```
Expected: `1704 species x 3 T x 4 replicas = 20448 systems in 269 units (106 atom counts)`.

**3. Submit one node for 30 minutes**, one unit per GPU (180, 126, 86, 44 atoms):
```bash
sbatch --time=00:30:00 d1/slurm_mlmd_run.sbatch /appsdata/$USER/sif/A_jax.sif /appsdata/$USER/d1 $OUT \
  --units n180_u000,n126_u000,n086_u000,n044_u000
```
Each unit took ~4 min on the project node (compile included).

**4. Collect the results**
- Job log: `mlmd_<jobid>_*.log` in the repository root. It should end with each GPU's `done …` / `finished` lines
  and `4/269 units done … 0 flagged`.
- Per-GPU logs, if something fails: `$OUT/mlmd_<jobid>_t0_g<k>.log`.
- Speed and self-checks per unit (standard library only):
```bash
python3 - <<EOF
import json, glob
for f in sorted(glob.glob("$OUT/units/*/meta.json")):
    m = json.load(open(f)); c = dict(m["chunk_seconds"]); sc = m["selfcheck"]
    print(m["id"], len(m["systems"]), "NVT %.1f NVE %.1f s/ps" % (c[1], c[3]),
          "dq %.1e dF %.1e" % (sc["end_of_nvt"]["max_dq"], sc["end_of_nvt"]["max_df"]))
EOF
```

**Project-node reference** (s per ps of simulation for the whole unit, steady chunks):

| unit | systems × atoms | NVT | NVE |
| --- | --- | --- | --- |
| n180_u000 | 84 × 180 | 33.9 | 54.6 |
| n126_u000 | 92 × 126 | 25.6 | 41.4 |
| n086_u000 | 99 × 86 | 18.1 | 29.5 |
| n044_u000 | 100 × 44 | 8.8 | 15.0 |

Record the outcome (job id, node, the table above for Hangang, any failure) in `docs/plan/cluster_migration.md`
under "Step 4 calibration", commit and push, so the project node can pick it up.

## Things to know

- `--env` splits on commas, so the sbatch passes `JAX_PLATFORMS=cuda,cpu` as `APPTAINERENV_JAX_PLATFORMS`.
- The run stops itself before the job's time limit (30 min margin, or 15 % of a short limit); resubmitting the same
  command resumes from checkpoints.
- Several nodes on one output directory only as one Slurm array (`--array=0-4`): task k runs shards k·4+g of 4K.
  Never two separate jobs on the same directory; resubmit with the same array size.
- Production is a separate decision after this job: a full plan (default settings) in a new directory, then a
  24 h array. Not part of this handoff.
- Mai et al. data and models are CC BY-NC-SA 4.0: keep them (and run outputs) out of git.
