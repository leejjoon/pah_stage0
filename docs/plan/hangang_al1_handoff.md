# Handoff: active-learning round 1 (cation frames), DFT on Hangang (2026-11-20)

For a Claude Code session on KISTI Hangang (account `t641a02`), branch `feat-gpu-gh200`. The allocation ends
**2026-11-22**: run this first. Container D (`/appsdata/$USER/sif/D_gpu4pyscf.sif`), the CuPy cache and the job
script are as in `docs/plan/hangang_seed_dft_handoff.md` and `docs/plan/hangang_hot_t2md_handoff.md`.

Rules: commit only on `feat-gpu-gh200`; push only when the user asks; never print `.env`; DFT results, logs and
models stay out of git.

## What and why

The cation head of the v2 MACE model generalises poorly (validation forces 92 meV/Å against 33 on its training set);
neither a larger model nor up-weighting cation forces helped (`docs/plan/charged_mlff_plan.md`, status log
2026-11-18/19), so round 1 of Phase 3 adds cation data where the model is uncertain. On the project node
(`d1/mlff_al_select.py`): 10 ps Langevin MD with v2 for the 76 train-split cations at 300/900/1,200 K (228 runs,
frames every 50 fs after 1 ps), committee force disagreement from three models (v2, v2_large_all, v2_q1fw3), and per
trajectory up to 15 frames with score 0.08–2.0 eV/Å and every input bond stretched at most 1.5× (frames from
trajectories that had already blown apart are excluded), at least 0.3 ps apart: **2,347 frames**, plus each pair's
minimum (76 frames, already labelled; kept as `mlff_qa.py`'s energy reference, dropped when building the dataset).
Train-split pairs only: no validation, final_test or transfer species.

| Frame dir | Pairs | Frames | Stability-tested |
| --- | --- | --- | --- |
| `frames_al1/` (`*_q1_al1.extxyz`), manifest `gjf_al1/manifest.json` | 76 train cations, 11–60 C | 2,423 | 386 |

Estimated cost from the seed set's per-size timings: ~48 process-hours (2 processes per GPU), so about 1.5–2 h on
5 nodes or ~6 h on one. Expect more flags than in the seed set: these frames are the ones the model gets wrong,
several from cations (CH₂-containing: C13H10, C15H12, C19H14, C20H14, C22H16) whose MLFF trajectories later broke
apart, and from 1,200 K C–H stretches. Report the counts of unconverged frames, Newton fallbacks, ⟨S²⟩ > 0.80 and
instabilities; do not drop frames.

## Inputs

`al1_hangang.tgz` (1.9 MB, sha256 `7d58f6d72449e8afc16163e2421f1d671c9171dd6b883ed3cfcd3077656b2836`), built on the
project node at `/var/tmp/jjlee-pah/d1/al1_hangang.tgz`; no Mai et al. geometries.
The user sends it (tailcat) to `/appsdata/$USER/d1/`. Contents under `al1_hangang/`:
- `frames_al1/`, `gjf_al1/manifest.json` (no .gjf files; not needed), `frames_al1/selection.csv` (score, trajectory, time per frame);
- `ref/`: 3 frames labelled on the project node (GPU4PySCF 1.9.2, GH200): C60H20_10225_q1_al1_f035 (149 s,
  stability-tested, ⟨S²⟩ 0.850), C36H16_7143_q1_al1_f001 (32 s), C13H10_54_q1_al1_f028 (43 s, stability-tested).

## Steps

```bash
cd /appsdata/$USER/pah_stage0 && git fetch && git checkout feat-gpu-gh200 && git pull --ff-only
cd /appsdata/$USER/d1 && sha256sum al1_hangang.tgz && tar xzf al1_hangang.tgz
H=/appsdata/$USER/d1/al1_hangang
cd /appsdata/$USER/pah_stage0
# check job: the 3 reference frames, then compare with the project node (G2 targets: 1e-7 Ha, 1e-5 Ha/bohr, 1e-3 D)
FRAME_DIRS="frames_al1" sbatch --time=00:40:00 d1/slurm_dft_seed.sbatch /appsdata/$USER/sif/D_gpu4pyscf.sif \
  $H $H/check --names C60H20_10225_q1_al1_f035,C36H16_7143_q1_al1_f001,C13H10_54_q1_al1_f028
apptainer exec --bind /appsdata /appsdata/$USER/sif/D_gpu4pyscf.sif python d1/dft_seed_run.py compare $H/ref $H/check
```
Report the `compare` line to the user. Then production (5 nodes; one is enough if the queue is long):
```bash
mkdir -p $H/dft && cp $H/check/*.json $H/dft/
FRAME_DIRS="frames_al1" sbatch --array=0-4 --time=04:00:00 d1/slurm_dft_seed.sbatch /appsdata/$USER/sif/D_gpu4pyscf.sif $H $H/dft
ls $H/dft/*.json | wc -l          # 2423 when complete
```
If the job ends early, resubmit the same command; finished frames are skipped.

## Hand back

```bash
apptainer exec --bind /appsdata /appsdata/$USER/sif/D_gpu4pyscf.sif \
  python d1/dft_seed_run.py collect $H/dft $H/dft.extxyz
tar czf /appsdata/$USER/d1/al1_dft.tgz -C $H dft dft.extxyz
sha256sum /appsdata/$USER/d1/al1_dft.tgz
```
Give the user the hash and the `collect` summary, and add a status-log entry to `docs/plan/charged_mlff_plan.md`.
On the project node: `mlff_qa.py` (manifest `gjf_al1/`), add the frames to train_q1 (drop the 76 duplicate minima),
retrain v2 and compare validation and test errors with v2.
