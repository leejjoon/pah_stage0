# Handoff: hot seed batch (1,800 K) and transfer2 MD frames, DFT on Hangang (2026-11-16)

For a Claude Code session on KISTI Hangang (account `t641a02`), branch `feat-gpu-gh200`. The allocation ends
**2026-11-22**. Setup (container D, `/appsdata/$USER/sif/D_gpu4pyscf.sif`, CuPy cache) is as in
`docs/plan/hangang_seed_dft_handoff.md`; this batch uses the same job script and settings.

Rules: commit only on `feat-gpu-gh200`; push only when the user asks; never print `.env`; DFT results, logs and
models stay out of git.

## What and why

Two independent batches, sampled with xTB on the project node (Hangang has no xTB), labelled here with GPU4PySCF
B3LYP/6-31G\* (`d1/dft_seed_run.py`, production settings):

| Batch | Frame dir | Pairs | Frames | Stability-tested |
| --- | --- | --- | --- | --- |
| Hot seed: per pair and charge, the minimum + 2 normal-mode frames at 1,800 K + 4 xTB MD frames at 1,800 K (2 ps equilibration, 0.5 ps apart) | `frames_hot/` (`*_q<q>_hot.extxyz`) | 41 kept pairs with 21–40 C, all splits (30 train, 6 validation, 5 final_test) | 574 | 492 (every 1,800 K frame) |
| transfer2 MD: per pair and charge, the minimum + one xTB MD frame at 300/900/1,500 K (as for the original transfer pairs) | `frames_t2md/` (`*_q<q>_md.extxyz`) | the 16 transfer2 pairs | 128 | 6 (the usual rule) |

- **Why hot frames** (decision 2026-11-16, `docs/plan/charged_mlff_plan.md` status log): the MD temperature is the
  cascade's quantum temperature, and for 21–40 C at 8–10 eV a sizeable share of the emission comes out above 1,200 K
  (`d1/peak_temperature.py`). The production MD grid goes up to 1,500 K for these sizes; 1,800 K gives the model a
  1.2× margin above it.
- **Why transfer2 MD frames:** the 16 backfilled transfer pairs had near-minimum frames only; these give real
  300–1,500 K transfer frames.
- Frame names carry the file tag, so they cannot collide with the existing labels: `C40H22_533_q1_hot_f003`,
  `C100H26_7494_q1_md_f003`. Each run's `min` frame repeats a geometry that is already labelled; it is kept so that
  `mlff_qa.py` has its per-run energy reference. Drop the duplicates when building the training data.
- Sampling integrity (xTB): 1,800 K normal-mode frames 109/164 with a changed bond graph (stretched C–H past the
  1.2 × covalent-radius cut); 1,800 K MD 84/328 (13 persistent); transfer2 MD 5/32 at 1,500 K, none below. Nothing
  is dropped; the frames carry `graph_changed` and `changed_n`.

## Inputs

`hot_t2md_1116.tgz` (0.57 MB, sha256 `4c2dda41b43023900fb5b3493eacf42a4fba6cba6f9b99f48f9fac40f9ed2729`), built on
the project node at `/var/tmp/jjlee-pah/d1/hot_t2md_1116.tgz`. The user sends it (tailcat). Contents under
`hot_t2md_1116/`:
- `frames_hot/`, `gjf_hot/manifest.json`, `frames_t2md/`, `gjf_t2md/manifest.json`;
- `smoke_ref/`: 3 frames labelled on the project node (GPU4PySCF 1.9.2, GH200): C40H22_533_q0_hot_f003 (74 s,
  stable), C40H22_533_q1_hot_f003 (111 s, stable), C100H26_7494_q1_md_f003 (124 s);
- `hot_jobs.txt`, `t2md_jobs.txt`, `hot_one.sh`, `t2md_one.sh`: how the frames were sampled.

## Steps

```bash
cd /appsdata/$USER/pah_stage0 && git fetch && git checkout feat-gpu-gh200 && git pull --ff-only
cd /appsdata/$USER/d1 && sha256sum hot_t2md_1116.tgz && tar xzf hot_t2md_1116.tgz
H=/appsdata/$USER/d1/hot_t2md_1116
cd /appsdata/$USER/pah_stage0
# check job: the 3 reference frames, then compare with the project node (targets as for the seed set)
FRAME_DIRS="frames_hot frames_t2md" sbatch --time=00:30:00 d1/slurm_dft_seed.sbatch /appsdata/$USER/sif/D_gpu4pyscf.sif \
  $H $H/check --names C40H22_533_q0_hot_f003,C40H22_533_q1_hot_f003,C100H26_7494_q1_md_f003
apptainer exec --bind /appsdata /appsdata/$USER/sif/D_gpu4pyscf.sif python d1/dft_seed_run.py compare $H/smoke_ref $H/check
```
Report the `compare` line to the user. Then production (one node is enough, ~12–15 GPU-h on the project node's
timing, so ~3–4 h on 4 GPUs):
```bash
mkdir -p $H/dft && cp $H/check/*.json $H/dft/
FRAME_DIRS="frames_hot frames_t2md" sbatch --time=06:00:00 d1/slurm_dft_seed.sbatch /appsdata/$USER/sif/D_gpu4pyscf.sif $H $H/dft
ls $H/dft/*.json | wc -l          # 702 when complete
```
`FRAME_DIRS` is new in `d1/slurm_dft_seed.sbatch` (default `frames frames_transfer`, unchanged behaviour). If the
job ends early, resubmit the same command; finished frames are skipped.

Expect more flags than in the seed set: RKS→UKS instabilities in hot neutral frames (recorded, the label stays RKS)
and cation ⟨S²⟩ > 0.80 rising with distortion. Report the counts; do not drop frames.

## Hand back

```bash
apptainer exec --bind /appsdata /appsdata/$USER/sif/D_gpu4pyscf.sif \
  python d1/dft_seed_run.py collect $H/dft $H/dft.extxyz
tar czf /appsdata/$USER/d1/seed_dft_hot_t2md.tgz -C $H dft dft.extxyz
sha256sum /appsdata/$USER/d1/seed_dft_hot_t2md.tgz
```
Give the user the hash and the `collect` summary, and add a status-log entry to `docs/plan/charged_mlff_plan.md`.
On the project node: `mlff_qa.py` per batch (manifests `gjf_hot/`, `gjf_t2md/`), then `mlff_coverage.py` with all
datasets joined.
