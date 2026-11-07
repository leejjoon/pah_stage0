# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

This is the planning stage of the **T0 / T0b** work for the "Interpretable PAH Diagnostics by Symbolic Regression" project. No code exists yet; the repository (git, branch `main`) holds:

- `docs/plan/T0_T0b_execution_plan_GH200.md`: the authoritative work plan.
- `docs/plan/T0_T0b_plan_review.md`: the first readiness review, dated 2026-11-06.
- `docs/plan/GH200_feasibility_check_2026-11-07.md`: measured checks on the project GH200 node (venvs first, then the built containers). The plan's GH200 facts cite it.
- `containers/`: the four tested Apptainer recipes (`A_jax`, `B_torch`, `B_torchsim`, `C_cpu`), the in-image package lists and the image hashes. Images are built to local NVMe, `/var/tmp/jjlee-pah/sif/`, not into the repo.
- `docs/plan/T0_T0b_committee_review.md`: a later three-round committee review (Fable 5.1 and GPT-6-Astra, with Opus 5.5 as a non-voting reviewer). It corrects several of the first review's fixes.

**Both reviews have been fully applied to the plan.** Read them before changing any criterion, config snippet or species rule: they explain why the plan says what it says. Where the two reviews disagree, the committee review wins.
- `docs/references/`: source PDFs (numbered by reading-list position). `more_references/` holds raw, unprocessed downloads (PDF, epub, arXiv tarballs).
- `corpora/references/`: a papertrail corpus built from the 14 reference papers (see below).
- `.env`: `DATALAB_API_KEY`, used for the Datalab/marker PDF parsing that built the corpus. Never print it.

## What T0 and T0b are

- **T0, the stack benchmark:** jax-md + mace-jax against ASE + MACE (PyTorch) on a GH200 (aarch64 Grace + H100). The question is whether JAX is more than 5× faster in wall-clock time per ns per molecule at a batch of 100.
  - Run a 2×2 grid: each stack with and without cuEquivariance.
  - The PyTorch baseline includes CUDA-MPS concurrent processes, not just sequential runs.
  - Gate: torch and jax forces must agree to within 1e-3 eV/Å (float32).
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
- **PySR**
  - Run it unweighted: the labels are exact and PAHFIT gives no sigma. If weights are ever used, they go to `fit()`, not the constructor.
  - `pysr.install()` no longer exists. `procs` needs `parallelism="multiprocessing"`.
  - The exact single-ratio inverse is c1 + c2/(c3−r), which is complexity 7.
- **Precision parity**
  - `mace_off` defaults to float64, and so does `MaceTorchSimModel`. JAX uses TF32 for float32 matmuls on the H100. Set everything explicitly.
  - `mace-jax-from-torch` has no dtype flag; cast the parameters when loading them.
  - Without `--model-name`, the converter falls back to the *medium* MACE-OFF. Always convert the exact checkpoint the PyTorch side loads.
- **Centre of mass:** the libraries' defaults differ, so set one explicit COM policy for every MD mode.
- **Julia in container C:** the juliapkg project (`/opt/julia_env`) must also be writable at run time (it takes a lock file), so bind a writable copy over it.
- **cuEquivariance:** use 0.10.x in both stacks; 0.13 is outside both stacks' pins. NGC PyTorch 25.08 and later is CUDA 13.
- **NGC 26.10** (checked 2026-11-07 against the registry): `26.10-py3` has an arm64 image with PyTorch 2.15.0a0 and CUDA 13.4, and it sets `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`. The cueq 0.10.0 torch ops extension does not link libtorch (nanobind; it needs only `libcue_ops`, `libcuda`), so there is no PyTorch ABI dependency.
- **MPS:** at most 60 clients per device (48 on older drivers). "Batch 100" means total work, not 100 processes.
- **Batched PyTorch:** TorchSim is the primary candidate and concatenated ASE `Atoms` the fallback; a week-4 pilot decides. The selected mode is never cut from the schedule.
- **Orion Bar**
  - 6.2/11.2 is *higher* in DF3 (Chown+2024 §4.3). Template order and agreement between templates are reported, never used as gates.
  - Compare the *emission-weighted* cation share with Ricca+2025 Table 3.
- **12.7/11.2** carries charge information (Chown+2024 §3.6), so using it is not evidence of a leak.
- **11.2 µm denominator:** it is weak for duo-only compact species. Check it at population level, and do *not* add a solo-H selection rule (that is the Wang 2026 Fig. 1 selection bias).
- **MACE-OFF** is neutral/closed-shell only and cannot produce cation trajectories.
- **Mai+2025** used NeuralIL (JAX) with an ASE driver, not MACE.
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
- MPS limit measured at 60 clients on 595.45. Keep `CUDA_MPS_PIPE_DIRECTORY` short (e.g. `/dev/shm/mps-$USER`): over 108 bytes, the daemon dies and clients silently run without MPS.
- Container A: install CPU torch *before* mace-jax (its `e3nn==0.4.4` dependency pulls CUDA torch). `mace-jax-from-torch --torch-model` fails on PyTorch ≥ 2.6 (`weights_only`); use `--foundation off --model-name small`. mace-jax's cuEquivariance is on only with `"equivariance_config": {"backend": "cueq"}` in the bundle JSON.
- Use local NVMe (`/`) for scratch; `/home` is NFS.
- Fast CPU↔GPU NVLink-C2C makes the PyTorch baseline look better here than it will on the production cluster. Note this in the T0 memo.
