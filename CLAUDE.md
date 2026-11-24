# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

This is the planning stage of the **T0 / T0b** work for the "Interpretable PAH Diagnostics by Symbolic Regression" project. No code exists yet; the repository (git, branch `main`) holds:

- `docs/plan/T0_T0b_execution_plan_GH200.md`: the authoritative work plan.
- `docs/plan/T0_T0b_plan_review.md`: the first readiness review, dated 2026-11-06.
- `docs/plan/GH200_feasibility_check_2026-11-07.md`: measured checks on the project GH200 node (venvs first, then the built containers). The plan's GH200 facts cite it.
- `containers/`: the tested Apptainer recipes (`A_jax`, `B_torch`, `B_torchsim`, `C_cpu`, and since 2026-11-14 `D_gpu4pyscf`), the in-image package lists, the image hashes and `tests/` (`run_all.sh` re-runs the feasibility checks). Images are built to local NVMe, `/var/tmp/jjlee-pah/sif/`, not into the repo. Container A is also in the public KSC Harbor registry as a SIF (ORAS artifact, byte-identical, sha256 `7304b1be…`): `apptainer pull oras://myhub.ksc.re.kr/pah/a_jax:gh200-2026-11-07`, no login needed. Pushing needs `HARBOR_USER`/`HARBOR_SECRET` from `.env`. For running the D1 MLMD work on the Slurm GH200 cluster (KISTI Hangang), see `docs/plan/cluster_migration.md`; run `docs/external/fetch_ksc_docs.sh` to mirror KISTI's Hangang, MyKSC and myhub guides as Markdown into `docs/external/` (gitignored).
  - `containers/D_gpu4pyscf.def` (2026-11-14): GPU4PySCF 1.9.2 for the charged-MLFF DFT, built from source for aarch64/sm_90 on `nvidia/cuda:12.9.1-devel` (no aarch64 wheels exist; CuPy is rebuilt because its aarch64 wheel lacks cuTENSOR). Notes and tests in `containers/D_gpu4pyscf.md`; results in `docs/research/pyscf_gpu_jax.md` §5. Also in myhub (public, byte-identical): `oras://myhub.ksc.re.kr/pah/d_gpu4pyscf:gh200-2026-11-14`. Its CUDA 12.9 runs on the host driver directly, so a host MPS daemon works (MPS gave no gain).
  - `containers/C_cpu_x86.def` rebuilds container C for x86-64 CPU-only machines, pinned to the GH200 versions through `containers/C_cpu_lock/`. It has been tested only on ARM so far. Build, test and run instructions are in `containers/C_cpu_x86.md`, and `tests/run_c.sh` holds the C checks.
- `docs/plan/T0_T0b_committee_review.md`: a later three-round committee review (Fable 5.1 and GPT-6-Astra, with Opus 5.5 as a non-voting reviewer). It corrects several of the first review's fixes.

**Both reviews have been fully applied to the plan.** Read them before changing any criterion, config snippet or species rule: they explain why the plan says what it says. Where the two reviews disagree, the committee review wins.
- `docs/references/`: source PDFs (numbered by reading-list position). `more_references/` holds raw, unprocessed downloads (PDF, epub, arXiv tarballs).
- `corpora/references/`: a papertrail corpus built from the 14 reference papers (see below).
- `.env`: `DATALAB_API_KEY`, used for the Datalab/marker PDF parsing that built the corpus. Never print it.

## Shared notes

Related sessions share information through `~/projects/pah-shared-notes`.
At the start of a task, run `git -C ~/projects/pah-shared-notes pull --rebase`
and read its `INDEX.md`, then run `./handoffs.sh` there: if a handoff is
addressed to this machine, follow "Receiving" in its `README.md` before other
work. When stopping work that another machine will continue, write a handoff
("Sending"). When you learn something another session would need,
append it to this branch's note (path: `~/projects/pah-shared-notes/branch-note.sh .`),
keep the branch's row in the project `SUMMARY.md` current, and run
`./sync.sh "message"` from that repo. Follow the rules in its `README.md`.

## Closing out a compute campaign and archiving

A campaign is a set of runs on remote machines whose results live outside git (e.g. the naphthalene DFT MD,
`docs/plan/cloud_aimd_handoff.md`). Code and documents reach `main` through branches. Data never goes in git. Git
holds a **manifest** of the data, and the data goes to a durable store. The project node's `/var/tmp` is local NVMe
with no backup, so it is a working copy, not an archive.

**Data store:** the private myhub project `pah-data`. Each dataset is a tarball pushed as an ORAS artifact,
`myhub.ksc.re.kr/pah-data/<dataset>:<YYYY-MM-DD>`. Its file digest equals the tarball's sha256. Credentials are a
`pah-data` robot account in `.env` as `HARBOR_DATA_USER` / `HARBOR_DATA_SECRET` (never print them). Pass them on
stdin rather than through `oras login`, so no login is saved:
`printf '%s' "$HARBOR_DATA_SECRET" | oras push -u "$HARBOR_DATA_USER" --password-stdin <ref> <file>:application/gzip`.
After pushing, pull the dataset back and check its sha256, then confirm an anonymous pull is refused (use
`--registry-config` with a file containing `{}`). Whether myhub keeps backups has not been checked, so keep one other
copy until it is known. First dataset: `data/manifests/aimd_naph_done.yaml`.

When every run of a campaign is finished, work through this list in order, and record each item in the handoff
document as it is done:

1. **Verify.** Every run is at its final step (`ALL_DONE` and `status`). Read the logs for crashes, SCF trouble and
   QA flags.
2. **Copy back.** Copy the results to the project node as a new directory and tarball; never overwrite earlier data.
   Write `SHA256SUMS` and check it against hashes computed on the remote machine and again after the copy. Leave out
   stale copies, lock files and forensic leftovers, and say what was left out.
3. **Archive.** Push the tarball to the data store and commit a manifest, `data/manifests/<dataset>.yaml`:
   ```yaml
   dataset: aimd_naph_done              # tarball name without .tgz
   sha256: <tarball sha256>
   size: 3.9 MB
   files: SHA256SUMS inside the tarball
   code: <commit that produced it>      # and the tag, once made (step 6)
   produced: 2026-11-15..2026-11-21 on gmunu (Slurm), wasp12 (systemd); per-run history in each npz `engines`
   settings: B3LYP(VWN3)/6-31G* 6D, 2 ps NVT + 20 ps NVE, 0.5 fs   # what is needed to reproduce it
   locations: [gh200:/var/tmp/jjlee-pah/d1/aimd_naph_done.tgz, <archive URI>]
   record: docs/plan/cloud_aimd_handoff.md
   ```
   Small final products (KB-sized analysis JSON or tables behind a decision) may be committed under `results/`.
4. **Clean up remotes, only after step 3 is confirmed.** Stop and remove services and units, delete run directories
   and installed environments on shared machines, cancel jobs, and delete cloud VMs, disks and projects. Look at each
   target before deleting it, and leave alone anything the campaign did not create.
5. **Turn the handoff into a record.** Replace the live parts ("where each run is", status, how to continue) with a
   short final account: what ran, where, which commit, the result, and the manifest. Add a dated entry to the plan's
   change log that points at it. Add an entry to the branch note in the shared notes and update the branch's
   `SUMMARY.md` row.
6. **Merge and tag.** Open a pull request to `main`; stacked branches merge oldest first. After the merge, tag the
   commit (e.g. `aimd-naph-b3p1`) so the dataset's code can be found, and delete the branch once nothing else builds
   on it.

Pushing, merging, tagging, deleting remote data and deleting cloud resources each need the user's go-ahead.

## What T0 and T0b are

- **T0, the stack benchmark:** jax-md + mace-jax against ASE + MACE (PyTorch) on a GH200 (aarch64 Grace + H100). The question is whether JAX is more than 5× faster in wall-clock time per ns per molecule at a batch of 100.
  - Run a 2×2 grid: each stack with and without cuEquivariance.
  - The PyTorch baseline includes CUDA-MPS concurrent processes, not just sequential runs.
  - Gate (plan §4.2): forces within 1e-3 eV/Å and interaction energies within 1e-4 eV/atom, both float32 (`ScaleShiftMACE` asserted, interaction energy taken directly from the model); float64 totals within 1e-6 eV/atom from an x64 conversion. Float32 totals are reported, not gated: they sit at the float32 rounding floor.
- **T0b, the vertical slice:** one tiny end-to-end pipeline:
  1. PAHdb v4.00 spectra
  2. emission cascade
  3. 2,000 synthetic populations
  4. MIRI MRS instrument model
  5. PAHFIT band measurement
  6. features table and random-forest ceiling
  7. PySR
  8. rediscovery check (a 6.2/11.2-type ionisation tracer)
  9. application to the five PDRs4All Orion Bar templates

Three Apptainer containers keep the stacks apart:
- A: JAX + mace-jax
- B: NGC PyTorch + mace-torch
- C: CPU tools (PySR, PAHFIT)

All three use Python 3.12, one venv per container, and `python -m pip`.

The repository layout the plan targets is `containers/`, `t0/`, `slice/` (driven by `make slice`), and `reports/` (plan §8).

### Non-obvious constraints (from the reviews; easy to regress)

- **PAHFIT**
  - Install it from git at a pinned commit. The PyPI 2.0 release (2022) has the old API and no `classic.yaml` or `jwst.yaml` packs.
  - PAHFIT returns no fit uncertainties; use Monte Carlo refits.
  - The classic pack has no 11.0 µm feature.
  - Fit with `maxiter=10000`: the default 1000 leaves fits unconverged without raising; check `model.fitter.message`.
- **PySR**
  - Run it unweighted: the labels are exact and PAHFIT gives no sigma. If weights are ever used, they go to `fit()`, not the constructor.
  - `pysr.install()` no longer exists. `procs` needs `parallelism="multiprocessing"`.
  - The exact single-ratio inverse is c1 + c2/(c3−r), which is complexity 7.
- **Precision parity**
  - `mace_off` defaults to float64, and so does `MaceTorchSimModel`. JAX uses TF32 for float32 matmuls on the H100. Set everything explicitly.
  - `mace-jax-from-torch` has no dtype flag and rounds the float64 checkpoint to float32 unless run with `JAX_ENABLE_X64=1`. Convert under x64, keep that float64 bundle as the master, and cast float32 copies when loading them.
  - Without `--model-name`, the converter falls back to the *medium* MACE-OFF. Always convert the exact checkpoint the PyTorch side loads.
- **Centre of mass:** the libraries' defaults differ, so set one explicit COM policy for every MD mode.
- **Julia in container C:** the juliapkg project (`/opt/julia_env`) must also be writable at run time (it takes a lock file), so bind a writable copy over it.
- **cuEquivariance:** use 0.10.x in both stacks; 0.13 is outside both stacks' pins. NGC PyTorch 25.08 and later is CUDA 13.
- **NGC 26.10** (checked 2026-11-07 against the registry): `26.10-py3` has an arm64 image with PyTorch 2.15.0a0 and CUDA 13.4, and it sets `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`. The cueq 0.10.0 torch ops extension does not link libtorch (nanobind; it needs only `libcue_ops`, `libcuda`), so there is no PyTorch ABI dependency.
- **MPS:** at most 60 clients per device (48 on older drivers). "Batch 100" means total work, not 100 processes.
- **TorchSim units:** metal units, so `timestep` is in ps and γ in ps⁻¹ (10 for the plan's 0.01 fs⁻¹). Its default γ is 20 ps⁻¹; always pass it.
- **Batched PyTorch:** TorchSim is the primary candidate and concatenated ASE `Atoms` the fallback; a week-4 pilot decides. The selected mode is never cut from the schedule.
- **Orion Bar**
  - 6.2/11.2 is *higher* in DF3 (Chown+2024 §4.3). Template order and agreement between templates are reported, never used as gates.
  - Compare the *emission-weighted* cation share with Ricca+2025 Table 3.
- **12.7/11.2** carries charge information (Chown+2024 §3.6), so using it is not evidence of a leak.
- **11.2 µm denominator:** it is weak for duo-only compact species. Check it at population level, and do *not* add a solo-H selection rule (that is the Wang 2026 Fig. 1 selection bias).
- **MACE-OFF** is neutral/closed-shell only and cannot produce cation trajectories.
- **Mai+2025** used NeuralIL (JAX) with an ASE driver, not MACE.
- **PAHdb v4 data** (downloaded 2026-11-07, not in the repo): `/var/tmp/jjlee-pah/data/pahdb-complete-theoretical-v4.00.xml(.gz)`, hash in `t0/pahdb-v4.00.sha256`. Load it with `--env TMPDIR=<local disk>` (the suite caches a 267 MB pickle in the temp dir). Its `symmetry` field is the electronic state, not the point group. T0's 20 species are in `t0/species_T0.csv`.
- **PAHdb v4** has few small cations (30 with 21–30 C). Select neutral/cation pairs first, and derive solo/duo/trio counts with `Geometry.bec()`.

## Querying the reference corpus

`corpora/references/` is a papertrail corpus. The `papertrail` CLI is installed in `.venv`. The venv uses Python 3.12, and papertrail was installed from `git@github.com:leejjoon/papertrail.git` with the `mcp`, `web` and `datalab` extras. Main commands:

```bash
.venv/bin/papertrail serve --corpus corpora/references --port 8765   # browse in a browser
.venv/bin/papertrail mcp --corpus corpora/references                 # MCP server (stdio by default)
.venv/bin/papertrail index --corpus corpora/references               # rebuild kb.sqlite from the pipeline outputs
```

`.mcp.json` registers the MCP server as `papertrail`. For questions about the references, prefer its tools: `search_fulltext`, `list_papers`, `get_paper`, `get_section`, `get_fulltext` and `get_figure_image`.

The other subcommands rebuild the corpus pipeline: `ingest-pdf`, `sections`, `classify`, `paper-pages` and `topic-pages`. `ingest-pdf` uses `DATALAB_API_KEY` from `.env`.

The corpus files can also be read directly:

- `data/kb.sqlite`
  - table `docs`: one row per paper, keyed by `doc_id`, e.g. `Chown2024_AA_PDRs4All_IV`
  - table `nodes`: parsed text blocks, with `doc_id`, `page`, `parent_title` and `text`
  - `chunks_fts`: an FTS5 index over `nodes.text` (porter tokenizer)
- `data/sections/<doc_id>.json`: the hierarchical section tree of each paper.
- `data/marker/<doc_id>.json`: raw marker output.
- `data/assets/<doc_id>/`: extracted figures.
- `documents.yaml`: metadata. Many `title`/`authors`/`year` fields are null or wrong, so trust `doc_id` instead.

`sqlite3` is not installed, so for a quick full-text search use Python:

```bash
python3 -c "
import sqlite3; c = sqlite3.connect('corpora/references/data/kb.sqlite')
for r in c.execute(\"select n.doc_id, n.page, n.parent_title, substr(n.text,1,200) from chunks_fts f join nodes n on n.id=f.rowid where chunks_fts match ? order by rank limit 10\", ('\"6.2/11.2\" ratio',)):
    print(r)"
```

When citing a claim, give the paper and section or page, as the review does (e.g. "Ricca et al. 2025, Table 1").

## Target hardware

- GH200 node: aarch64. Every container, wheel and binary must be built for ARM; x86 images will not run.
- The project node (`gh200`) is a **dual** GH200: 2 GPUs (144 GB HBM3e each), 144 cores, 18 NUMA nodes. T0 uses one superchip: `CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0`.
- Kernel with 64 KiB pages; driver 595.45.04 (CUDA 13.2). NGC `26.10-py3` is built for driver 615.71 but runs here through CUDA forward compatibility (verified); `26.03-py3` is the native match for 595.45 if that ever breaks.
- Apptainer 1.5.5 (setuid) is installed; build with `apptainer build --fakeroot`. AppArmor blocks unprivileged user namespaces, so a non-setuid Apptainer would not work.
- Container A must hold exactly one CUDA (no CUDA-toolkit base). Otherwise importing cuEquivariance before JAX makes JAX silently fall back to the CPU. `JAX_PLATFORMS=cuda` guards against it.
- Container C must use uv's CPython, not Ubuntu's `python3.12`. Ubuntu's OpenSSL 3.0 forces Julia 1.11.9, which segfaults in PySR here.
- MPS from container B: run the daemon **inside the image** (bind the host's `nvidia-cuda-mps-control`/`-server`; `t0/run_mps.sh`). B uses the forward-compat libcuda 615.71; with a host daemon (595.45) its clients connect and then hang forever in `cuInit` (feasibility §K).
- MPS limit measured at 60 clients on 595.45. Keep `CUDA_MPS_PIPE_DIRECTORY` short (e.g. `/dev/shm/mps-$USER`): over 108 bytes, the daemon dies and clients silently run without MPS.
- Container A: install CPU torch *before* mace-jax (its `e3nn==0.4.4` dependency pulls CUDA torch). `mace-jax-from-torch --torch-model` fails on PyTorch ≥ 2.6 (`weights_only`); use `--foundation off --model-name small`. mace-jax's cuEquivariance is on only with `"equivariance_config": {"backend": "cueq"}` in the bundle JSON. That alone still runs plain JAX (`method="naive"`, same as the default bundle); the CUDA kernel needs `"conv_fusion": true` as well, and even then only the convolution uses it in mace-jax 0.2.0 (feasibility §J).
- Use local NVMe (`/`) for scratch; `/home` is NFS.
- Fast CPU↔GPU NVLink-C2C makes the PyTorch baseline look better here than it will on the production cluster. Note this in the T0 memo.

## Shared notes

Related sessions share information through `~/projects/pah-shared-notes`.
At the start of a task, run `git -C ~/projects/pah-shared-notes pull --rebase`
and read its `INDEX.md`, then run `./handoffs.sh` there: if a handoff is
addressed to this machine, follow "Receiving" in its `README.md` before other
work. When stopping work that another machine will continue, write a handoff
("Sending"). When you learn something another session would need,
append it to this branch's note (path: `~/projects/pah-shared-notes/branch-note.sh .`),
keep the branch's row in the project `SUMMARY.md` current, and run
`./sync.sh "message"` from that repo. Follow the rules in its `README.md`.
On KISTI Hangang (account `t641a02`) the clone is at `~/work/pah-shared-notes`, not `~/projects/`; use that path
in the commands above. There `hostname` is the login node you are on (e.g. `glogin04`), which can differ between
sessions; name the machine that way in handoffs and note entries.
