# Proposal: charge-state machine-learned force fields for temperature-dependent PAH infrared spectra

Version 1, 2026-11-12. Status: **for independent review** (shared copy for comments: https://claude.ai/artifact/S5H8iSEdkckcZRTAybUuVm; this file is the master). Project: "Interpretable PAH Diagnostics by Symbolic
Regression" (`docs/plan/big_picture_proposal.md`), data stream D1. Operational companion:
`docs/plan/charged_mlff_plan.md`. Evidence: `docs/plan/charged_tdep_validation.md`, `docs/plan/cluster_migration.md`.

This document is meant to be read on its own. Section 11 lists the questions we most want reviewers to answer.
Statements marked *(measured)* come from our own runs on the dates given; *(source)* gives the paper and section;
*(estimate)* is a projection that a named step will replace with a measurement.

---

## 1. Summary

The project needs infrared (IR) spectra of polycyclic aromatic hydrocarbons (PAHs) as a function of internal
temperature for neutral and ionised molecules, because the JWST band ratios it aims to turn into diagnostics
(6.2/11.2 µm and others) trace the ionisation state, and the bands are emitted by hot molecules. The only
machine-learned model for anharmonic PAH spectra at scale (Mai et al. 2025) covers neutrals only, was trained on
300 K geometries, and fails above ~600 K in our tests, and for some species at any temperature.

We propose to build our own training data with Gaussian 16 at the level of theory of the NASA Ames PAH database
(PAHdb v4.00: B3LYP/6-31G\*), for neutrals and cations first and anions and dications later, sampled up to 1500 K,
and to train force-field and dipole models per charge state, extending the data by active learning. A cheap
semi-empirical test (GFN2-xTB) already indicates that band *positions and widths* shift with temperature almost
identically in neutrals and cations, but it cannot settle band *intensities*, which is what the ionisation
diagnostics depend on; that is the main scientific reason for this work.

Cost *(estimate, to be replaced by Phase 0 measurements)*: ~10k CPU core-hours for a 5,800-point seed set,
~85–220k core-hours for the full 50–80k-point dataset, 350–850 GPU-hours for training and validation, and
~1,700–2,600 GPU-hours for production spectra across four charge states (inside the project's existing D1 GPU
budget of 4,000–4,500 GPU-h). Calendar time: about 3–4 months to production-ready models.

## 2. Scientific context

### 2.1 Why temperature-dependent spectra, and why ions

- The project builds a labelled library of synthetic PAH emission spectra and searches it for compact formulas that
  recover population properties (size, charge, edge structure) from JWST band measurements
  (`big_picture_proposal.md`, D1 and objectives O1–O4). The best-known charge tracers are ratios of the CC-stretch
  bands at 6–9 µm (strong in cations) to the CH out-of-plane band at 11.2 µm (strong in neutrals).
- PAHs emit after absorbing a UV photon, at high internal temperature, so emission spectra differ from 0 K
  absorption spectra: bands shift, broaden and change relative intensity (Mai et al. 2025, §1).
- The standard treatment in PAHdb-based emission modelling is empirical: a fixed 15 cm⁻¹ redshift "to simulate
  anharmonic effects" and a 15 cm⁻¹ FWHM profile on harmonic, scaled band positions *(source: Bauschlicher et al.
  2018, §4)*. The D1 plan replaces this with molecular-dynamics (MD) spectra at 300, 600, 900 and 1200 K for ~2,000
  species, so that the temperature dependence of each band is learned rather than assumed
  (`big_picture_proposal.md`, D1).
- If the temperature dependence of band intensities differs between charge states, a diagnostic built from neutral
  trends applied to ions would be biased in exactly the quantity it is meant to measure.

### 2.2 The existing model (Mai et al. 2025)

*(source: Mai et al. 2025, MNRAS 541, 3073; Zenodo 15771437)*
- MD with a machine-learned force field (NeuralIL, JAX) and a separate dipole model (EPNN); the IR spectrum is the
  Fourier transform of the dipole autocorrelation (§2.4). Spectra of 1,704 neutral PAHs (up to 216 C) at 50, 300 and
  600 K are published.
- Training data: 687 neutral PAHs from PAHdb v3.2; 17,175 configurations from classical (AIREBO) MD at 300 K plus
  random displacements; B3LYP/4-31G single points with Gaussian 16 (§2.1). Held-out test errors: forces 1.15–1.19
  ×10⁻³ Ha/bohr (≈0.06 eV/Å), dipole 0.05–0.06 D (Table 1). The training data are not in the release.
- The authors state the limits (§4): neutral molecules only; increased uncertainty for molecules unlike the
  training set; and training conformations "mainly sourced from molecular conformations at 300 K, which are in
  principle not sufficient for reliably capturing IR spectral features at very high temperatures (e.g. >600 K)".

### 2.3 Level of theory of the reference data

*(source: Ricca et al. 2025, §2.1; Bauschlicher et al. 2018, §3, Table 2)*
- PAHdb v3.20 used B3LYP/4-31G with one frequency scale factor (0.958). PAHdb v4.00 recomputed **all** spectra with
  B3LYP/6-31G\*, which adds polarisation functions, and scales them with three factors (0.964 for CH stretches,
  0.979 for 4–9 µm, 0.975 beyond 9 µm), fitted to 25 gas-phase bands.
- In the PAHdb v4.00 XML *(measured, 2026-11-12)*, the C/H species are almost all B3LYP/6-31G\*: 2,524 closed-shell
  neutrals, 1,469 doublet cations, 688 anions and ~2,400 dications (singlet and triplet); only a handful at 4-31G.
  The stored route is `b3lyp/6-31g* freq ... scf=(xqc, maxconventional=64, maxcycles=128)`.
- Mai et al.'s model is therefore one basis-set level below the harmonic library it is meant to complement.

## 3. Evidence gathered so far

### 3.1 Our port of the published model is faithful *(measured, 2026-11-10/11)*

- Our JAX port of Mai et al.'s NeuralIL + EPNN reproduces their pinned code to 1.9×10⁻⁵ eV/Å in forces and
  4.7×10⁻⁶ e in charges. Naphthalene spectra from 32 replicas match the published spectrum's strong-band centroids
  within 1.7 cm⁻¹. Throughput, after our optimisations: about 0.0022 s per ps per atom per molecule, batched on one
  GH200 GPU; the full 1,704-species × 3-temperature × 4-replica plan is ~420 GPU-h.

### 3.2 The published model breaks at high temperature, and for some species at any temperature *(measured)*

A bond-graph check (bonds = distances below 1.2 × the sum of covalent radii; end of run compared with the input
geometry) on runs of 20 ps NVT + 50 ps NVE, 4 replicas:

| Test | Systems intact |
| --- | --- |
| 33 species (9–216 C, incl. N-substituted, odd-electron, hydrogenated), 600 K | 105/132 |
| same, 900 K | 89/132 |
| same, 1200 K | 43/132 (41 with H loss, 37 with C–C changes) |
| 9 species that failed above, 50 K | 20/36 (16 changed) |
| same 9 species, 300 K | 16/36 (20 changed) |

Six species change in every replica even at 50 K (a ring opens in C9H8_495, H atoms move between carbons in
C60H24_812, C–H bonds stretch past 1.28 Å in others): the model gets these species wrong, independent of
temperature. Only three species fail because of temperature. A sweep of all 1,704 published species at 50 and 300 K
(one replica each) is running to list the species the model breaks at low temperature *(in progress: 76 of 124 units
done on 2026-11-12)*.

### 3.3 Do cations and neutrals have the same temperature dependence? A GFN2-xTB test *(measured, 2026-11-11/12)*

GFN2-xTB (tblite 0.8.0) handles charged and open-shell molecules cheaply. Naphthalene, pyrene and coronene, neutral
and cation (doublet), 300/600/900/1200 K, 4 replicas each, 20 ps Langevin NVT + 100 ps NVE, dipole about the centre
of mass every 1 fs; 96 trajectories, 64 core-hours. The same three neutrals with Mai et al.'s model served as a
control. Band trends were fitted against each replica's actual mean NVE temperature (one NVE run of a small molecule
can average tens of percent away from its target).

| Quantity (per 100 K, 300–900 K) | Result |
| --- | --- |
| Bands present in both charge states (xTB) | Shift slopes differ by ≤ 0.6 cm⁻¹ per 100 K (coronene 494/496, 1242/1238; pyrene 648/630, 800/798); exception coronene⁺ 792 (Jahn–Teller split) |
| CC-stretch bands | Mai's neutral model −4.6 to −5.4 cm⁻¹; xTB cations −3.9 to −5.0 cm⁻¹ |
| Width growth | +0.2 to +0.6 cm⁻¹ in both |
| Control: CH stretch, xTB vs Mai neutral | −9.0 to −9.3 vs −10.6 to −11.8 cm⁻¹ |
| Control: in-plane intensity share, neutrals | xTB rises 6–10 % per 100 K, Mai's model 0–4 % |
| CH-stretch intensity share | xTB cations rise 3–18 % per 100 K; neutrals within ±3 % |

Reading: band positions and widths transfer from neutrals to cations within a few cm⁻¹ up to 900 K. Intensities do
not settle: xTB fails the neutral control exactly in the in-plane region that carries the CC bands, so its
cation intensity trends cannot be trusted. Limits: three species, four replicas, xTB without spin polarisation.

### 3.4 What the DFT data will cost *(measured with PySCF, 2026-11-12; Gaussian not yet timed)*

B3LYP (Gaussian's VWN3 variant)/4-31G energy + forces + dipole, geometry displaced by 0.05 Å, 8 threads on Grace
CPU cores; restricted for neutrals, unrestricted for doublet cations:

| Species | Basis functions | Core-hours per point, neutral / cation | Cation SCF cycles | Cation ⟨S²⟩ |
| --- | --- | --- | --- | --- |
| C10H8 | 106 | 0.013 / 0.018 | 14 | 0.764 |
| C24H12 | 240 | 0.089 / 0.15 | 20 | 0.767 |
| C32H14 | 316 | 0.29 / 0.47 | 18 | 0.775 |
| C54H18 | 522 | 0.90 / 1.30 | 19 | 0.777 |
| C90H24 | 858 | 3.2 / 5.3 | 33 | 0.812 |
| C144H30 | 1356 | 9.4 / 13.4 | 30 | 1.043 |

Cost grows as (basis functions)^2.6; cations cost 1.4–1.7× neutrals. 6-31G\* has 15 rather than 9 functions on C,
so ~3.4× more. Density fitting was only 0.9–2.5× faster and changed the largest force component by up to 2×10⁻³
Ha/bohr, comparable to the model error we aim for; we do not use it. Spin contamination grows with cation size
(⟨S²⟩ 1.04 at C144 against 0.75 for a pure doublet).

### 3.5 Software and hardware available *(measured, 2026-11-12)*

- KISTI Hangang: Gaussian 16 C.02 is installed. CPU partition `normal_cpu`: 680 nodes with 256 cores and ~770 GB
  each, 24 h limit, nodes shared between jobs; about half the nodes were idle when checked. GPU partitions: 4 GH200
  per node, 24 h, at most 5 running jobs per user.
- mace-jax 0.2.0 (our container): no total-charge or spin input, but multi-head training (one output head per data
  subset); dipole readout blocks and losses exist but no assembled dipole model. MACE-OFF, the organic MACE
  foundation model, is limited to neutral, non-radical systems *(source: Kovács et al. 2025, §4)*.
- Mai et al.'s NeuralIL and EPNN training code is in their release (code Apache 2.0; models and data CC BY-NC-SA).

## 4. Objectives

1. **O-A Data:** a reference dataset of B3LYP/6-31G\* energies, forces and dipoles for PAH neutrals and cations
   (then anions and dications), 10–100 C, sampled from 0 to 1500 K, with quality flags.
2. **O-B Models:** force-field and dipole models per charge state that are accurate on held-out species and stable
   in MD from 50 to 1200 K.
3. **O-C Spectra:** temperature-dependent IR spectra (300–1200 K) of ~2,000 species per charge state for the D1
   library, computed with the existing, validated MD and spectrum pipeline.
4. **O-D Answer:** a quantitative statement of how much band intensities (and positions and widths) differ in their
   temperature dependence between charge states, with uncertainties, for use as an injected systematic in D1.

Not in scope: N- or O-substituted PAHs (deferred), pure-carbon clusters and fullerenes, quantum nuclear effects
(Fermi resonances; MD is classical, as in Mai et al. 2025), and electronic excited states.

## 5. Approach

### 5.1 Level of theory (decided: B3LYP/6-31G\*, 2026-11-12)

| | B3LYP/6-31G\* (recommended) | B3LYP/4-31G |
| --- | --- | --- |
| Consistent with | PAHdb v4.00: harmonic library, geometries, all charge states, its three scale factors | Mai et al.'s model and published spectra; PAHdb ≤ v3.20 |
| Seed set, 5,800 points *(estimate)* | ~10k core-h | ~3k core-h |
| Full data, 50–80k points, ≤ 100 C *(estimate)* | ~85–220k core-h | ~25–65k core-h |

We chose 6-31G\* (project decision, 2026-11-12): the anharmonic temperature trends and the harmonic library they modify then come from the
same electronic structure, and the extra cost is days on Hangang. Mai et al.'s data are unavailable, so the neutral
model is retrained in either case. Frequencies from MD carry the basis set's systematic offset; we would apply
PAHdb v4's three scale factors, and report the alternative of scaling to the PAHdb harmonic positions.

### 5.2 Electronic-structure protocol

- Gaussian 16: `#P (U)B3LYP/6-31G* Force NoSymm SCF=(XQC,MaxConventionalCycles=64,MaxCycle=256)`; unrestricted for
  odd-electron states (cations and anions of closed-shell PAHs are doublets); dications in the PAHdb ground-state
  multiplicity (singlet or triplet, from the PAHdb electronic-state field). `NoSymm` keeps coordinates and forces in
  the input frame.
- Recorded per point: energy, forces, dipole moment, ⟨S²⟩ before annihilation, SCF cycles, termination status.
- The dipole of an ion depends on the origin; we store it about the centre of mass, as in the neutral pipeline. A
  Phase 0 test (the same cation translated by 3 Å) establishes which origin Gaussian reports.

### 5.3 Species

- Seed set *(built)*: closed-shell neutral / doublet cation pairs of the same structure in PAHdb v4.00 (matched by a
  bond-graph hash of the two optimised geometries), 10–60 C. 655 such pairs exist; 116 are chosen, ~23 per 10-carbon
  bin, by farthest-point sampling in edge-structure fractions (solo, duo, trio, quartet CH), always including
  naphthalene, anthracene, phenanthrene, pyrene, coronene, ovalene and circumcoronene. 23 pairs are held out. Small
  pairs are scarce (25 with 10–19 C, 27 with 20–29 C), so nearly all of them are included. The anion and dication of
  each structure are recorded for later phases.
- Later: 60–100 C, more irregular edges, superhydrogenated species, species that Mai et al.'s model breaks (§3.2),
  anions and dications.

### 5.4 Configurations

- Per species and charge state *(tool built; seed sampling running)*: 5 copies of the DFT minimum with 0.05 Å random
  displacements, and 4 frames each at 300, 600, 900, 1200 and 1500 K from GFN2-xTB Langevin MD (2 ps equilibration,
  frames 0.5 ps apart, centre of mass fixed). 25 frames each, 5,800 points for the seed set.
- Frames whose bond graph changed during xTB MD are dropped: the seed set describes intact molecules. Whether to add
  deliberately distorted or dissociating frames so the models fail safely is an open question (§11). Of the seed set,
  15 % of 1500 K frames, 7 % at 1200 K and 1.4 % at 900 K were dropped, none below: 5,586 frames kept of 5,800
  *(measured, 2026-11-12)*.
- Sampling to 1500 K is the key difference from Mai et al.'s 300 K training data. xTB's own errors only affect which
  geometries are sampled; all labels come from DFT.

### 5.5 Models

Two model families on the same data, chosen after validation:
1. **NeuralIL + EPNN, one model per charge state**, with Mai et al.'s training code. Lowest risk: our MD and spectrum
   pipeline already runs these models, and their speed is known.
2. **MACE (mace-jax) with one output head per charge state** and a separate dipole model (EPNN, or a MACE dipole
   head assembled from the existing blocks). MACE is the project's planned architecture; its MD speed in our setting
   is being measured in the T0 benchmark.

Each as an ensemble of 3–4 models with different seeds, for uncertainty estimates. Charge enters as the choice of
model or head, which is enough because the charge is fixed along a trajectory.

### 5.6 Active learning

Rounds of: MD with the ensemble over many species, charge states and temperatures → frames with the largest force
disagreement → Gaussian → retrain. Each round widens the scope (60–100 C; then anions and dications). Stop when the
disagreement and the integrity failure rate stay below thresholds on species never used in training. Expected 3–5
rounds, about 10k points each.

### 5.7 Validation and acceptance gates

Thresholds below are proposals for review.

| Gate | Test | Proposed threshold |
| --- | --- | --- |
| G0 Gaussian set-up | Energies vs PySCF on the same geometries (same functional and basis); PAHdb total energies and harmonic frequencies for naphthalene neutral and cation | ≤ 1×10⁻⁵ Ha vs PySCF (integration grids differ); PAHdb energies ≤ 1×10⁻⁵ Ha (PAHdb used Gaussian 09 and 16, whose default grids differ); frequencies ≤ 2 cm⁻¹ |
| G1 Data quality | Normal termination; ⟨S²⟩ of doublets; spot check of forces against finite differences on a few frames | ≥ 95 % terminated; ⟨S²⟩ ≤ 0.80 kept, larger flagged; force check ≤ 1×10⁻⁴ Ha/bohr |
| G2 Accuracy | Held-out species, per charge state and per sampling temperature | Force MAE ≤ 0.06 eV/Å (Mai et al.'s level), including 1200–1500 K frames; dipole MAE ≤ 0.06 D (neutrals), ≤ 0.1 D about the centre of mass (ions) |
| G3 On-trajectory accuracy | DFT single points on frames from the model's own MD | MAE ≤ 1.5 × the held-out value |
| G4 Stability | Bond-graph check after 20 ps NVT + 50 ps NVE, held-out species, 4 replicas | ≥ 99 % intact at 50–900 K, ≥ 95 % at 1200 K |
| G5 Harmonic consistency | Model Hessian and dipole derivatives at the minimum vs PAHdb v4.00 (same level of theory) | Strong-band frequencies within 10 cm⁻¹ RMS of PAHdb (unscaled); relative intensities within 20 % for bands > 10 % of the strongest |
| G6 Spectra | 50 K positions vs PAHdb scaled harmonic; 300 K neutral spectra vs Mai et al.'s (allowing for the basis-set offset); cation positions vs gas-phase lab spectra where available (D2) | Reported, not gated, except gross failures |
| G7 Temperature trends | Band slopes per 100 K, neutral vs cation, compared with §3.3 | Reported with uncertainties; differences are the result, not a gate |

### 5.8 Production and integration

The accepted models run in the existing MD and spectrum pipeline (Langevin NVT, NVE, 0.5 fs, dipole every 1 fs,
spectra with the authors' `calc_MD_IR` maths, replicas averaged, integrity check per system). Spectra at 300, 600,
900 and 1200 K for ~2,000 species per charge state go to the D1 emission model; the measured charge dependence of
band trends (O-D) becomes an injected systematic.

## 6. Resources and schedule

| Phase | What | CPU core-h | GPU-h | Calendar |
| --- | --- | --- | --- | --- |
| 0 | Gaussian set-up and timing (28 jobs) | < 100 | – | 1 week |
| 1 | Seed set: 116 pairs × 2 charges × 25 frames | ~10k | – | 1–2 weeks |
| 2 | Train two model families, ensembles; gates G2–G6 | ~2k (G3 single points) | 100–300 | 3–4 weeks |
| 3 | Active learning, 3–5 rounds; to 100 C; anions, dications | ~75–210k | 250–550 | 6–8 weeks |
| 4 | Production spectra, 4 charge states × ~2,000 species × 4 T × 4 replicas | – | ~1,700–2,600 | 2–4 weeks |

All CPU figures are *(estimates)* from the PySCF pilot and will be redone from the Phase 0 Gaussian timings.
Production GPU-hours scale our measured neutral plan (420 GPU-h for 1,704 species × 3 temperatures, NeuralIL);
MACE may be slower. Wall-clock for CPU phases assumes ~5 Hangang CPU nodes at a time (1,280 cores); per-user limits
on `normal_cpu` are not yet known. Storage: parsed data are small (< 5 GB); compressed Gaussian logs are kept, of
order 100 GB *(estimate)*. Software is free except Gaussian, which Hangang licenses.

## 7. Risks and mitigations

| Risk | Likelihood | Effect | Mitigation |
| --- | --- | --- | --- |
| Spin contamination of large cations (⟨S²⟩ ≈ 1 at C144) makes their DFT data unreliable | High for > 100 C | Biased cation forces and dipoles | Cap cations at ~100 C in training; flag ⟨S²⟩; test restricted open-shell on a few |
| Unrestricted B3LYP gives a delocalised hole that differs from the true electronic structure (and from xTB) | Medium | Systematic cation error | Compare with PAHdb (same method) and with lab cation spectra; report as a limitation |
| SCF failures on distorted ions | Medium | Lost points | XQC fallback; log and drop; resample |
| Models stable on training-like species but not on unusual ones (as Mai et al.'s) | Medium | Unusable spectra for some species | Integrity check per system in production; active learning on failures; exclude persistent failures |
| Classical MD misses quantum effects (Fermi resonances), and 50 K spectra are not true low-T spectra | Certain | Known bias, mainly CH stretch | Same as Mai et al.; compare with lab and PAHdb; report |
| Gaussian slower than estimated, or queue limits low | Medium | Schedule | Phase 0 measures it; cap size; prioritise cations |
| MACE MD too slow for production | Medium | GPU budget | NeuralIL family as fallback (already integrated) |
| Too few small cation species in PAHdb | Certain | Thin coverage at 10–30 C | Add structures with geometries optimised by us; data from the anion and dication of the same structures |

## 8. Alternatives considered

1. **Apply neutral temperature trends to ions (no new data).** Supported for band positions and widths by §3.3, not
   for intensities. This remains the fallback if the project cannot afford the new data, with the §3.3 numbers as its
   uncertainty.
2. **GFN2-xTB MD for ions in production.** Cheap (§3.3) but fails the neutral control for in-plane intensities;
   its intensities are not trustworthy.
3. **Direct DFT MD (ab initio MD) for ions.** Accurate, but at 0.02–13 core-h per force evaluation (§3.4) it is
   affordable only for a few small species and short runs; useful as a spot check, not for a library.
4. **A charge- and spin-aware foundation model.** We have not verified any that covers open-shell aromatic cations
   at our level of theory; a separate literature survey is under way. A suitable one could replace the training step
   but not the validation.
5. **Keep 4-31G.** Cheaper and consistent with Mai et al., but one basis-set level below the PAHdb v4.00 library we
   combine it with (§5.1).
6. **Density fitting or GPU DFT to cut cost.** Density fitting changes forces at the level of the target model error
   (§3.4); GPU4PySCF has no ARM build. Gaussian on CPU is the practical choice.

## 9. Decision points

- **After Phase 0:** confirm the level of theory and the dipole origin; replace all CPU estimates with Gaussian
  timings; set the size cap for cations.
- **After Phase 2:** go or no-go for active learning. No-go if G2 or G4 fail with no clear data fix; then fall back
  to alternative 1 for intensities, with the seed data as a quantitative check of that fallback on 116 pairs.
- **After Phase 3:** choose the model family for production from G2–G5 and measured MD speed.

## 10. Deliverables

- Dataset (extended XYZ: positions, energy, forces, dipole about the centre of mass, charge, multiplicity, ⟨S²⟩,
  sampling temperature, source structure), with the Gaussian inputs and the selection and sampling scripts.
- Trained models per charge state, ensembles, with validation reports (G0–G7).
- Temperature-dependent spectra per charge state for the D1 library.
- A note quantifying the charge dependence of band temperature trends (O-D).
- Code: `d1/mlff_species.py`, `d1/mlff_sample.py`, `d1/g16_io.py`, `d1/slurm_g16.sbatch`, `d1/g16_phase0.py`,
  `d1/xtb_md.py`, `d1/tdep_compare.py`, `d1/mlmd_integrity.py`, `d1/dft_timing.py` (all present).

Licensing: our data and models would not derive from Mai et al.'s CC BY-NC-SA data or models; their Apache-2.0 code
may be used for training. PAHdb geometries are used as starting structures.

## 11. Questions for reviewers

1. Is B3LYP/6-31G\* the right reference level, given that the purpose is temperature *trends* to apply to PAHdb v4.00
   harmonic spectra? Would a dispersion correction or a range-separated functional matter for intramolecular PAH
   dynamics?
2. Is unrestricted B3LYP adequate for PAH cations up to ~100 C, and what ⟨S²⟩ threshold should exclude data? Is a
   restricted open-shell treatment worth testing?
3. Is per-charge-state modelling (separate models or heads) preferable to a single model with charge and spin as
   inputs, given that charge is fixed along each trajectory?
4. Is the sampling scheme (displaced minima plus xTB frames at 300–1500 K, intact molecules only) adequate, or should
   the seed set include distorted or dissociating geometries so the models fail safely?
5. Are the acceptance thresholds (§5.7) appropriate, in particular G2 (force MAE ≤ 0.06 eV/Å) and G4 (stability)?
6. Which laboratory data on temperature-dependent or gas-phase IR spectra of PAH cations should anchor G6?
7. Is 1200 K the right upper temperature for the library, and is 1500 K enough margin in the training data?
8. How should MD frequencies at 6-31G\* be scaled: PAHdb v4's three factors, or a per-band offset to PAHdb harmonic
   positions at 50 K?
9. Is the first scope (neutrals and cations, 10–60 C) the right priority for the ionisation diagnostics?

## References

- Bauschlicher, C. W., et al. 2018, ApJS, "The NASA Ames PAH IR Spectroscopic Database: Computational Version 3.00
  with Updated Content and the Introduction of Multiple Scaling Factors" (§3 Table 2; §4 emission model).
- Kovács, D. P., et al. 2025, JACS, MACE-OFF (§4 Outlook: neutral, non-radical scope).
- Mai et al. 2025, MNRAS 541, 3073; Zenodo 15771437 (§2.1 training data; §2.2 Table 1; §2.4 MD; §4 limitations).
- Ricca, A., et al. 2025, ApJS, PAHdb version 4.00 (§2.1 computational method).
- Batatia, I., et al. 2022, NeurIPS, MACE.
- Software: Gaussian 16 C.02; PySCF 2.15; tblite 0.8.0 (GFN2-xTB); ASE; JAX; mace-jax 0.2.0.
- Project documents: `big_picture_proposal.md` (D1), `cluster_migration.md` (port, validation, hot test),
  `charged_tdep_validation.md` (xTB test, DFT pilot), `charged_mlff_plan.md` (run-book).

## Glossary

- **MLMD**: molecular dynamics with a machine-learned force field; the IR spectrum is the Fourier transform of the
  dipole autocorrelation along the trajectory.
- **NVT / NVE**: MD at constant temperature (thermostat) / constant energy. Spectra come from NVE segments.
- **Replica**: an independent trajectory of the same molecule and temperature with a different random seed.
- **⟨S²⟩**: expectation value of the total spin squared; 0.75 for a pure doublet. Larger values signal spin
  contamination in unrestricted calculations.
- **Held-out species**: molecules none of whose configurations are in the training data.
- **Bond-graph check**: comparison of which atom pairs are bonded (distance below 1.2 × the sum of covalent radii)
  at the end of a run with the input geometry.
- **Solo / duo / trio / quartet**: CH groups with 1, 2, 3 or 4 adjacent H atoms on one ring edge; they set the
  11–14 µm out-of-plane bands.
