# Handoff: active-learning round 2 (new small species, both charges), DFT on Hangang (2026-11-21)

For a Claude Code session on KISTI Hangang (account `t641a02`), branch `feat-gpu-gh200`. The allocation ends
**2026-11-22**. Container D (`/appsdata/$USER/sif/D_gpu4pyscf.sif`), the CuPy cache and the job script are as in
`docs/plan/hangang_al1_handoff.md`. If Hangang cannot take it in time, the same package runs on the project node
(two GPUs, about 7 h).

Rules: commit only on `feat-gpu-gh200`; push only when the user asks; never print `.env`; DFT results, logs and
models stay out of git.

## What and why

Round 1 (2,423 cation frames of the 76 train species) improved cation validation forces by about 10 %
(92.0 → 82.5 meV/Å) but left the per-species gap: the worst validation and test species have five-membered rings,
CH₂/sp³ carbons or non-planar skeletons that train barely covers (`docs/plan/charged_mlff_plan.md`, status log
2026-11-21). Round 2 adds new species instead of more frames: every unused PAHdb v4.00 skeleton with ≤ 25 C
(closed-shell RB3LYP/6-31G* neutral; carbon skeleton in no existing split), all to train (`d1/mlff_species_r2.py`):
27 pairs, of which 4 five-ring, 6 with sp³ carbons (one seven-ring), about 10 twisted (helicene-like). PAHdb has a
cation for only 4; the other 23 cation minima were optimised on the project node (UB3LYP/6-31G*, GPU4PySCF +
geomeTRIC, same SCF settings as `dft_seed_run.py`). Frames: the seed recipe (`mlff_sample.py` defaults: minimum,
displacements, normal-mode sampling, GFN2-xTB MD at 300–1,500 K, C–H stretches and out-of-plane pushes), 32 per pair
and charge, both charges.

| Frame dir | Pairs | Frames | Stability-tested |
| --- | --- | --- | --- |
| `frames_al2/` (`*_q0.extxyz`, `*_q1.extxyz`), manifest `gjf_al2/manifest.json` | 27 new train pairs, 9–24 C | 1,728 (54 files × 32) | 331 |

Estimated cost: 11–69 s per frame on the reference frames (the seed set's ≤ 25 C mean was 32–60 s): 10–20 GPU-h, so
under 1 h on 4 nodes.
Report the counts of unconverged frames, Newton fallbacks, ⟨S²⟩ > 0.80 and instabilities; do not drop frames.

## Inputs

`al2_hangang.tgz` (0.8 MB, sha256 `17caf2a9e2c37f1fab739d753e1f49ecd2b857150b3306ca77b414260e2d6ef4`), built on the project node at `/var/tmp/jjlee-pah/d1/al2_hangang.tgz`;
no Mai et al. geometries. Also in the private myhub data store as `oras://myhub.ksc.re.kr/pah-data/al2_hangang:2026-11-21`
(pushed and pull-back verified from gh200; anonymous pull refused). Fetch it with a `pah-data` robot account from
Hangang's `.env` (`HARBOR_DATA_USER` / `HARBOR_DATA_SECRET`; never print them; a static arm64 `oras` binary from the
oras-project GitHub releases if none is installed):
```bash
cd /appsdata/$USER/d1 && (set -a; . /appsdata/$USER/pah_stage0/.env; set +a
  printf '%s' "$HARBOR_DATA_SECRET" | oras pull -u "$HARBOR_DATA_USER" --password-stdin myhub.ksc.re.kr/pah-data/al2_hangang:2026-11-21)
```
or the user sends it (tailcat) to `/appsdata/$USER/d1/`. Hand back the same way: `oras push ... myhub.ksc.re.kr/pah-data/al2_dft:<date> al2_dft.tgz:application/gzip`. Contents under `al2_hangang/`:
- `frames_al2/`, `gjf_al2/manifest.json` (no .gjf files), `species_r2.csv`;
- `ref/`: 3 frames labelled on the project node (GPU4PySCF 1.9.2, GH200): C24H14_2326_q1_f008 (69 s,
  stability-tested, ⟨S²⟩ 0.778), C9H8_495_q0_f020 (11 s, MD 900 K), C15H14_354_q1_f000 (20 s; the optimised cation
  minimum: its energy equals the optimiser's to 1e-9 Ha, max |F| 0.0005 eV/Å).

## Steps

```bash
cd /appsdata/$USER/pah_stage0 && git fetch && git checkout feat-gpu-gh200 && git pull --ff-only
cd /appsdata/$USER/d1 && sha256sum al2_hangang.tgz && tar xzf al2_hangang.tgz
H=/appsdata/$USER/d1/al2_hangang
cd /appsdata/$USER/pah_stage0
# check job: the reference frames, then compare with the project node (G2 targets: 1e-7 Ha, 1e-5 Ha/bohr, 1e-3 D)
FRAME_DIRS="frames_al2" sbatch --time=00:30:00 d1/slurm_dft_seed.sbatch /appsdata/$USER/sif/D_gpu4pyscf.sif \
  $H $H/check --names C24H14_2326_q1_f008,C9H8_495_q0_f020,C15H14_354_q1_f000
apptainer exec --bind /appsdata /appsdata/$USER/sif/D_gpu4pyscf.sif python d1/dft_seed_run.py compare $H/ref $H/check
```
Report the `compare` line to the user. Then production (4 nodes, the user's limit):
```bash
mkdir -p $H/dft && cp $H/check/*.json $H/dft/
FRAME_DIRS="frames_al2" sbatch --array=0-3 --time=02:00:00 d1/slurm_dft_seed.sbatch /appsdata/$USER/sif/D_gpu4pyscf.sif $H $H/dft
ls $H/dft/*.json | wc -l          # 1728 when complete
```
If the job ends early, resubmit the same command; finished frames are skipped.

## Hand back

```bash
apptainer exec --bind /appsdata /appsdata/$USER/sif/D_gpu4pyscf.sif \
  python d1/dft_seed_run.py collect $H/dft $H/dft.extxyz
tar czf /appsdata/$USER/d1/al2_dft.tgz -C $H dft dft.extxyz
sha256sum /appsdata/$USER/d1/al2_dft.tgz
```
Give the user the hash and the `collect` summary, and add a status-log entry to `docs/plan/charged_mlff_plan.md`.
On the project node: `SPECIES=<v2 tables>,al2/species_r2.csv d1/mlff_al_ingest.sh` on top of dataset v3's inputs
(the new minima are kept), retrain and compare with v3.
