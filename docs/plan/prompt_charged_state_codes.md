# Prompt: survey of codes for charged and open-shell PAH molecular dynamics

Hand this to a research agent (web access needed; the papertrail corpus in `corpora/references/` is optional
background). Written 2026-11-11.

---

## Task

Survey the software and models that could give **finite-temperature (anharmonic) IR spectra of charged
polycyclic aromatic hydrocarbons (PAHs)** by molecular dynamics, and recommend which ones are worth testing. Write
the result to `docs/research/charged_state_codes.md` and finish with a short summary of your recommendation.

## Why

We compute anharmonic, temperature-dependent IR spectra of PAHs by machine-learned MD (MLMD): MD with an ML force
field, a dipole model evaluated along the trajectory, and the IR spectrum from the dipole autocorrelation. We use
the models of Mai et al. 2025 (MNRAS 541, 3073; Zenodo 15771437): a NeuralIL force-field ensemble and an EPNN charge
model, both in JAX, trained on DFT data for **neutral** PAHs, mostly at 300 K. The astrophysical use (JWST
diagnostics of PAH ionisation, e.g. the 6.2/11.2 µm band ratio) needs the same temperature dependence for
**cations** (doublets, open-shell), and ideally for anions and dications. No model we have handles them.

## What a usable option must provide

1. Energies and forces for PAH ions: charge −1, 0, +1, +2; spin multiplicity 1 or 2 (cations and anions of
   closed-shell PAHs are doublets; dications are singlets or triplets). Elements C and H, ideally N and O as well.
   Sizes from 10 to about 200 carbon atoms (16–250 atoms).
2. A dipole moment along the trajectory (directly, or through atomic charges that sum to the molecular charge).
   For an ion the dipole depends on the origin; note how each code defines it (centre of mass, centre of charge).
3. Stable MD from 50 to 1200 K over hundreds of ps (0.5 fs timestep), without unphysical bond breaking.
4. Enough speed for thousands of trajectories. Our reference: Mai's model runs about 0.0022 s per ps per atom per
   molecule, batched on one GH200 GPU.

## What to survey

Cover at least these categories. For each candidate, check the primary source (paper, documentation, repository);
do not rely on summaries.

- **ML potentials that take total charge and spin as input**, including universal or foundation models trained on
  large datasets with ions and open-shell molecules. For each, find out whether its training data actually include
  open-shell aromatic radical cations, and whether it outputs charges or dipoles.
- **ML dipole and charge models** that could pair with a separate force field: charge-equilibration or
  long-range models that give charges summing to a given total charge.
- **Retraining or fine-tuning routes**: can Mai's NeuralIL/EPNN (their training code is in the Zenodo release) or
  another architecture be fine-tuned on cation DFT data? What size of DFT dataset is typical, and is there an
  existing dataset of PAH ion conformations or AIMD frames?
- **Semi-empirical methods** (GFN-xTB family, DFTB variants): open-shell and charged support, dipoles, published
  accuracy for PAH IR spectra, and GPU or batched implementations.
- **DFT molecular dynamics codes**, for reference trajectories of a few small ions: GPU DFT codes and
  cost estimates for 10–50 ps of a naphthalene or coronene cation.
- **Prior work** on temperature-dependent or anharmonic IR spectra of PAH cations by any method (MD, VPT2,
  experiment), so we know what reference data exist.

## For each candidate, report

| Field | Notes |
| --- | --- |
| Name, version, reference | Paper and repository URLs |
| Charge / spin support | How it is passed; tested for open-shell? |
| Training data and level of theory | Do they include PAH-like radical cations? |
| Elements | C, H, N, O? |
| Dipole or charges | Built in, or needs a separate model |
| Framework, hardware | JAX / PyTorch / Fortran; GPU; **aarch64 (ARM) builds available?** |
| Speed | Published numbers, with molecule size and hardware |
| Licence | Commercial use, non-commercial (NC), share-alike; weights versus code |
| Validation | Against IR spectra, AIMD, or experiment for aromatic molecules |
| Maintenance | Last release, activity |

Then rank the candidates for our two uses: (a) a cheap validation of whether cation and neutral temperature trends
differ (a handful of small PAHs, any accuracy that captures differences), and (b) production for about 2,000
species × 4 charges × 4 temperatures.

## Constraints

- Our hardware is NVIDIA GH200 (aarch64 Grace CPU + H100), in Apptainer containers. x86-only binaries will not
  run on the GPU nodes; KISTI Hangang also has x86 CPU nodes.
- Distinguish what a source states from what you infer. Mark anything unverified. Give dates for versions and
  releases, and the date you checked.
- Do not invent capabilities, numbers or citations. If you cannot find something, say so.
- Do not send any of our project details, names or file contents to external services; search with generic terms.
