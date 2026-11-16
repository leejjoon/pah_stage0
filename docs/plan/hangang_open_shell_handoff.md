# Handoff: open-shell labels for the excluded pairs and the `unstable` frames, DFT on Hangang (2026-11-17)

For a Claude Code session on KISTI Hangang (account `t641a02`), branch `feat-gpu-gh200`. The allocation ends
**2026-11-22**, so this batch runs first. Container D, the CuPy cache and the job script are as in
`docs/plan/hangang_seed_dft_handoff.md` and `docs/plan/hangang_hot_t2md_handoff.md`.

Rules: commit only on `feat-gpu-gh200`; push only when the user asks; never print `.env`; DFT results, logs and
models stay out of git.

## What and why

Phase 1 has no correct label for two groups of frames:
- the 9 open-shell pairs excluded as whole pairs (`d1/seed_exclusions_open_shell.csv`): their RKS neutral labels lie
  above a broken-symmetry UKS solution. The exclusion removes solo-H-rich, large pairs (the Wang 2026 Fig. 1 bias,
  which must be stated wherever Phase 1 results appear);
- the 49 neutral frames of kept pairs flagged `unstable` (RKS→UKS unstable), dropped from `dataset_v1`.

The state-QA side set (2026-11-17, `d1/dft_stateqa.py`) showed that the mixed-guess UKS with stability
re-optimisation finds the lower solution where one exists, and nowhere else. This batch reruns these frames with
`dft_seed_run.py run --stab-all --bs`, the mode already used for the 124 neutral minima (2026-11-15):
- every frame is stability-tested; an RKS→UKS unstable neutral frame gets a UKS solution started from the rotated
  orbitals (fallback: Guess=Mix), re-optimised on internal instabilities; the `bs_*` keys hold its energy, forces,
  dipole and ⟨S²⟩;
- the cation frames of the 9 pairs get internal stability re-optimisation (Stable=Opt) on every frame, not only the
  sampled ones (C58H20_3182 cation ⟨S²⟩ jumped 0.95 → 1.15 → 1.27 with temperature, which looks like more than one
  state).

`collect --bs-label` (new) then writes the labels: a neutral frame with a converged UKS solution below RKS takes it
(`method=UB3LYP-BS`, `bs_label=bs`); a stable one keeps RKS (`bs_label=rks`); an unstable one without a lower
converged solution keeps RKS and is marked `bs_label=bs_failed`. Whether these frames enter Phase 1 training is
decided afterwards, on the project node; this batch only makes the labels exist.

| Group | Frame dirs (on Hangang) | Frames |
| --- | --- | --- |
| seed | `seed_hangang/mlff_seed`: `frames/`, `frames_transfer/` | 431: 207 neutral + 207 cation frames of the 9 excluded pairs, 17 `unstable` neutral frames |
| hot/t2md | `hot_t2md_1116`: `frames_hot/`, `frames_t2md/` | 32 `unstable` neutral frames (31 hot, 1 t2md) |

Cost (project-node timings of the same mode): about 36 GPU-h for the seed group and 2 GPU-h for hot/t2md; with
Hangang's two processes per GPU, about 8–9 h on one node, so the seed group runs as an array of 3 nodes.

## Inputs

`os_relabel_1117.tgz`, built on the project node at `/var/tmp/jjlee-pah/d1/os_relabel_1117.tgz` (hash in the status
log of `docs/plan/charged_mlff_plan.md`). The user sends it (tailcat). Contents under `os_relabel_1117/`:
- `names_seed.txt`, `names_hot_t2md.txt`: one comma-separated line of frame names each;
- `smoke_ref/`: 4 frames run on the project node in this mode (C13H10_55_q0_f000, C34H20_820_q1_f000,
  C26H16_814_q0_f008, C22H14_307_q0_hot_f001).

The frames themselves are already on Hangang (`seed_hangang.tgz`, `hot_t2md_1116.tgz`); the frame names select them.

## Steps

```bash
cd /appsdata/$USER/pah_stage0 && git fetch && git checkout feat-gpu-gh200 && git pull --ff-only
cd /appsdata/$USER/d1 && sha256sum os_relabel_1117.tgz && tar xzf os_relabel_1117.tgz
O=/appsdata/$USER/d1/os_relabel_1117; S=/appsdata/$USER/d1/seed_hangang/mlff_seed; H=/appsdata/$USER/d1/hot_t2md_1116
SIF=/appsdata/$USER/sif/D_gpu4pyscf.sif
cd /appsdata/$USER/pah_stage0
# check jobs: the 4 reference frames (three from the seed dirs, one from the hot dir)
sbatch --time=01:00:00 d1/slurm_dft_seed.sbatch $SIF $S $O/check --stab-all --bs \
  --names C13H10_55_q0_f000,C34H20_820_q1_f000,C26H16_814_q0_f008
FRAME_DIRS="frames_hot frames_t2md" sbatch --time=01:00:00 d1/slurm_dft_seed.sbatch $SIF $H $O/check_hot --stab-all --bs \
  --names C22H14_307_q0_hot_f001
# after both: (compare now also checks the broken-symmetry energy, dE_bs_Ha, target 1e-6 Ha)
mkdir -p $O/check_all && cp $O/check/*.json $O/check_hot/*.json $O/check_all/
apptainer exec --bind /appsdata $SIF python d1/dft_seed_run.py compare $O/smoke_ref $O/check_all
```
Report the `compare` line to the user. Then production (two jobs on separate output dirs; never two jobs on one):
```bash
mkdir -p $O/dft_seed $O/dft_hot && cp $O/check/*.json $O/dft_seed/ && cp $O/check_hot/*.json $O/dft_hot/
sbatch --array=0-2 --time=06:00:00 d1/slurm_dft_seed.sbatch $SIF $S $O/dft_seed --stab-all --bs \
  --names $(cat $O/names_seed.txt)
FRAME_DIRS="frames_hot frames_t2md" sbatch --time=03:00:00 d1/slurm_dft_seed.sbatch $SIF $H $O/dft_hot --stab-all --bs \
  --names $(cat $O/names_hot_t2md.txt)
ls $O/dft_seed/*.json | wc -l; ls $O/dft_hot/*.json | wc -l     # 431 and 32 when complete
```
If a job ends early, resubmit the same command; finished frames are skipped. Check that a log line `BS:` follows
every unstable neutral frame and that `tried` is not `collapsed` for all guesses.

## Hand back

```bash
for g in seed hot; do
  apptainer exec --bind /appsdata $SIF python d1/dft_seed_run.py collect $O/dft_$g $O/dft_$g.extxyz --bs-label
done
tar czf /appsdata/$USER/d1/seed_dft_open_shell.tgz -C $O dft_seed dft_seed.extxyz dft_hot dft_hot.extxyz
sha256sum /appsdata/$USER/d1/seed_dft_open_shell.tgz
```
Give the user the hash and both `collect` summaries (the `bs_label` counts first), and add a status-log entry to
`docs/plan/charged_mlff_plan.md`. On the project node: compare the cation labels with the existing ones (a lower
re-optimised doublet means a state change), then decide with the user how these frames enter the training data.
