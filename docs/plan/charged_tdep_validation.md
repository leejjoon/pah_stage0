# Do PAH cations and neutrals have the same temperature dependence? (draft, 2026-11-11)

Status: **first test run** (2026-11-11; results below, under "Early test"); the plan text is still a draft. Context: `docs/plan/cluster_migration.md` (the neutral MLMD
pipeline) and the code survey prompt in `docs/plan/prompt_charged_state_codes.md`.

## Question and decision

The D1 library gets harmonic spectra for all charge states (PAHdb, the He, Mai & Wang 2026 surrogate), but
temperature dependence (band shifts, widths, intensity changes) only for neutrals, from Mai et al.'s neutral-only
models. The cheap route is to apply the neutral temperature trends to the ions' harmonic spectra.

This test decides whether that is defensible:
- **If** cation and neutral trends agree within the seed scatter of our neutral MLMD (step 2b: band shares vary by
  tens of percent between replicas for small PAHs; centroids by ~1–2 cm⁻¹), transfer them and state it as a
  limitation, with the measured difference as its uncertainty.
- **If not**, especially for the CC-stretch bands (6.2 and 7.7 µm) that carry the ionisation diagnostic, a
  charged-state force field becomes a work package (option 3).

## Method

A cheap method that handles charge and open-shell states and gives dipoles: **GFN2-xTB** MD (the `xtb` program
or `tblite`; LGPL). Absolute accuracy is modest, but the question is about a *difference* between charge states
computed with one method.

**Control, essential:** the same species as neutrals with GFN2-xTB, compared with our neutral MLMD (Mai's model).
If xTB reproduces the neutral MLMD trends (shift and broadening per 100 K, per band) reasonably, its
cation-minus-neutral difference is credible. If it does not, the test says nothing and we need DFT MD for a smaller
set.

Checks before production:
- GFN2-xTB for cations: unrestricted, doublet (`--chrg 1 --uhf 1`). GFN2 has no spin polarisation by default;
  `tblite` has a spin-polarised extension. Compare both on one species.
- Dipole of an ion depends on the origin: use the centre of mass, as for the neutrals; the ACF of a constant offset
  only adds a zero-frequency term, but check it does not leak into 300+ cm⁻¹.
- Harmonic frequencies and intensities of xTB versus DFT (PAHdb) for each pair: the known xTB offsets must be
  similar for neutral and cation, or band-by-band comparisons need per-charge scaling.

## Species (draft)

Small to medium, compact, with good reference data, and neutral/cation pairs in PAHdb:

| Species | C | Why |
| --- | --- | --- |
| naphthalene | 10 | Smallest; our step 2b reference; many lab data |
| anthracene, phenanthrene | 14 | Linear versus angular edge |
| pyrene | 16 | Classic lab species, cation spectra measured |
| coronene | 24 | Compact, high symmetry; astrophysically typical |
| ovalene or circumcoronene | 32 / 54 | Size trend, if the cost allows |

Option: add the anions and dications of naphthalene and coronene (cheap once the pipeline exists).

## Protocol

As close to the neutral MLMD as possible, so differences come from the charge, not the protocol:
- Langevin NVT then NVE, 0.5 fs, centre-of-mass motion removed; 20 ps NVT, 100–200 ps NVE (to set by the seed
  scatter).
- 300, 600, 900, 1200 K; 4 replicas per species, charge and temperature.
- Dipoles every 1 fs; spectra with `d1/mlmd_spectra.py` (the authors' `calc_MD_IR` maths), replicas averaged.
- Integrity check with `d1/mlmd_integrity.py` (bond graph against the input geometry), since ions may fragment or
  isomerise sooner.

## Measurements

Per band window (as in `d1/port/mai_spectrum_compare.py`, extended to 6.2 / 7.7 / 8.6 / 11.2 / 12.7 µm and the CH
stretch): centroid, FWHM and share of the total intensity, versus temperature, with the replica scatter.
Report d(centroid)/dT, d(FWHM)/dT and d(share)/dT for the neutral and the cation, their difference, and whether it
exceeds the scatter. Plus the control: xTB-neutral against MLMD-neutral, same quantities.

## Cost (to measure before planning)

Unknown until timed: GFN2-xTB gradients cost roughly milliseconds for naphthalene and tens of ms for coronene per
step on one core, so one 120–220 ps trajectory (240,000–440,000 steps) is hours to a day of one core. With 6
species × 2 charges × 4 T × 4 replicas = 192 trajectories, that is of order 10³ core-hours: small on Hangang's CPU
nodes (x86, AMD Turin). Time one naphthalene and one coronene trajectory first.

Software: `xtb` and `tblite` are on conda-forge; check that an aarch64 build exists if the project node is used,
otherwise run on Hangang's x86 CPU nodes in the x86 container C (`containers/C_cpu_x86.def`) with xTB added.

## Reference points (to collect)

- Lab: temperature-dependent IR of PAH cations in the gas phase, if any (the code survey asks for these); FELIX /
  FELion and PIRENEA cation spectra (named in the proposal) give positions at one temperature.
- Theory: anharmonic (VPT2) spectra of small PAH cations, if published; a short DFT MD of the naphthalene cation as
  an anchor if xTB's neutral control is poor.

## Open questions

- Is 100 ps NVE enough for band widths at 300 K for naphthalene with xTB? Step 2b used 200 ps.
- GFN2-xTB's treatment of open-shell cations (spin contamination, delocalised hole) may itself differ from DFT;
  the code survey may find a better cheap method.
- The neutral MLMD itself is extrapolating above 600 K (Mai et al. 2025 §4); the hot test
  (`/var/tmp/$USER-pah/d1/hot_test`) checks stability and integrity there.

## Early test (2026-11-11, project node)

Setup: `d1/xtb_md.py` (tblite 0.8.0 GFN2-xTB through ASE, conda-forge aarch64 build in a micromamba env at
`/var/tmp/$USER-pah/mamba/xtb`); geometries from PAHdb v4.00 (neutral/cation uids: naphthalene 330/359, pyrene
334/360, coronene 18/19); cations as doublets (`multiplicity=2`, tblite's default non-spin-polarised GFN2).
Naphthalene, pyrene, coronene × q = 0, +1 × 300/600/900/1200 K × 4 replicas, 20 ps Langevin NVT + 100 ps NVE,
0.5 fs, dipole about the centre of mass every 1 fs. 96 trajectories on 48 Grace cores, one core each: 64 core-hours in all.
Outputs: `/var/tmp/$USER-pah/d1/xtb_test/runs/`; analysis `d1/tdep_compare.py`.

Cost (one core, 120 ps, 48 running at once): naphthalene 0.25–0.29 h, pyrene 0.54–0.61 h, coronene 1.06–1.26 h
per trajectory; the cation 5–15 % slower. That is ~30 s per ps for coronene, against ~0.08 s per ps per molecule for
the batched neutral MLMD on one GPU.

Stability: E_tot drift over 100 ps NVE ≤ 0.15 eV (cations drift upward by ~0.1 eV, neutrals mostly down);
bond graphs intact in all 72 runs at 300–900 K; at 1200 K one C–H bond lost in 3 of 24 (two neutrals, one cation).

Caveat for all temperature trends here and in the MLMD: one NVE run of a small molecule has a mean temperature far
from the target (naphthalene "900 K" runs averaged 1,170 K in one replica, coronene's 775 K), because the NVE energy
is one canonical sample. Replica means help; a trend fitted against each replica's NVE ⟨T⟩ would be cleaner.

### What the numbers show (xTB only; control pending)

- **Region windows mislead.** In the xTB neutrals the in-plane CC/CH bands (1000–1700 cm⁻¹) are weak (~10⁻⁴ of the
  total per bin), so that window fills with a broad background that grows with T and its centroid moves blue
  (+45 to +86 cm⁻¹ by 1200 K), while the cations' strong CC bands move red. The share of the in-plane window
  rises 1.4–1.9× in the neutrals and falls slightly in the cations. Band-level numbers are the meaningful ones.
- **Bands present in both charge states shift alike.** Bands followed step by step in T (`tdep_compare.py`),
  change at 1200 K, neutral / cation: coronene 494/496 −4 / −7 cm⁻¹, coronene 1242/1238 −21 / −22, pyrene 648/630
  −16 / −17, pyrene 800/798 −15 / −21. Widths grow alike (+2 to +4 cm⁻¹). Differences of 0–6 cm⁻¹, often formally
  > 2 s.e. with 4 replicas but small. Exception: coronene⁺'s 792 band barely moves (coronene⁺ is Jahn–Teller
  distorted, and its oop band is split).
- **Cation CC bands shift more**: −12 to −19 cm⁻¹ at 600 K and −37 to −60 at 1200 K (1280–1474 cm⁻¹ after the 0.9578
  scaling), roughly twice the neutral CH-bend and oop shifts. xTB neutrals have no comparable strong CC band, so
  this is not yet a cation-versus-neutral difference for the same band; the neutral MLMD control (below) tests it.
- The CH stretch red-shifts by ~−20 / −45 / −67 cm⁻¹ at 600 / 900 / 1200 K in the neutrals.

### Control: xTB neutral against MLMD neutral

`/var/tmp/$USER-pah/d1/xtb_test/mlmd_ref`: the same three neutrals with Mai et al.'s model, 300–1200 K, 4 replicas,
20 ps NVT + 100 ps NVE (3 units, 48 systems; no QA flags). Integrity: 2 of 12 at 900 K lost a C–H bond (one coronene),
none at 1200 K. Comparison in `xtb_test/compare.txt` and `compare.png`.

- **CH stretch region**: the shifts agree. Change at 600 / 900 / 1200 K, MLMD vs xTB: coronene −32/−30, −76/−54,
  −84/−77; naphthalene −24/−33, −68/−67, −92/−84; pyrene −30/−31, −63/−64, −107/−85 cm⁻¹.
- **Out-of-plane bands** that pair up across methods (within 40 cm⁻¹) agree up to 900 K and part at 1200 K:
  naphthalene 774/762 −7/−7, −21/−7, −30/−14; pyrene 694/724 −4/−4, −8/−8, −16/−8. Most strong bands sit
  > 40 cm⁻¹ apart between the methods and are not paired.
- **In-plane window**: the MLMD neutral also moves blue (naphthalene +36/+56/+60), but its share stays nearly flat
  (coronene 0.99–1.01, pyrene 1.04–1.13, naphthalene 1.22–1.28), while xTB's grows to 1.5–1.9. So xTB exaggerates
  the high-T background in the neutrals' weak in-plane region; xTB shares there are not trustworthy.
- **CC bands**: the MLMD neutral's CC bands shift by −11 to −14 cm⁻¹ at 600 K, −29 to −31 at 900 K and −33 to −54 at
  1200 K (coronene 1316; pyrene 1316, 1430). That is the size of the xTB cation CC shifts (−12 to −19, −24 to −37,
  −37 to −60). So the large cation CC shifts above are **not** a charge effect: neutral CC bands (in the better
  model) shift alike.

### Slopes against each replica's actual temperature

`tdep_compare.py` now also fits each band's centroid, width and share (relative to 300 K) linearly against each
replica's NVE mean temperature, 300–900 K (`TMAX` to change), with errors from the residuals. Replica NVE ⟨T⟩ at
nominal 900 K: MLMD 800–1024 K, xTB 775–1172 K (xTB scatters more). Per 100 K:
- CC bands: MLMD neutral −4.6 ± 0.2 (coronene 1316), −5.4 ± 0.2 and −5.1 ± 0.3 (pyrene 1316, 1430); xTB cation
  −3.9 to −4.7 (coronene 1330/1394/1474), −5.0 and −4.7 (naphthalene⁺ 1280/1416), −4.0 and −4.7 (pyrene⁺
  1322/1452). Widths +0.2 to +0.6 cm⁻¹ per 100 K in both.
- Bands in both xTB charge states: slopes differ by ≤ 0.6 cm⁻¹ per 100 K (coronene 494/496, 1242/1238; pyrene
  648/630, 800/798), except coronene⁺ 792 (0.0 against −1.7 for the neutral's 820).
- Control, xTB against MLMD neutral: CH-stretch centroid −9.0 to −9.3 vs −10.6 to −11.8 per 100 K; matched oop bands
  within 2.3 cm⁻¹ per 100 K (most within 0.5). xTB's CH-stretch broadening is smaller than MLMD's (+4.0–4.4 vs +5.2–8.0).
- Shares: the cations' CH-stretch share rises 3–18 % per 100 K (xTB) where neutrals' stays within ±3 %; the neutrals'
  in-plane share rises 6–10 % per 100 K in xTB but 0–4 % in MLMD (the control failure noted above).

### Reading so far

- Band **positions and widths**: no sign of a charge-specific temperature dependence beyond a few cm⁻¹ for
  bands seen in both charge states, and the CC-band shifts in the xTB cations match those of the MLMD neutrals.
  This supports transferring neutral shift/broadening trends to cations, with a few-cm⁻¹ uncertainty (≤ 900 K).
- Band **intensities/shares** (what the 6.2/11.2 diagnostic needs): not settled. xTB's neutral in-plane shares are
  unreliable (control fails there), and cation CH-stretch shares rise steeply with T in xTB (×1.3–3.2 by 1200 K)
  from a weak 300 K band. A per-band intensity comparison needs a better method for the cation (DFT MD for one
  small cation, or a charged-state model).
- Exceptions to look at: Jahn–Teller cations (coronene⁺) behave differently in the oop region.
- Limits: 3 species, 4 replicas, nominal temperatures (NVE ⟨T⟩ scatters by tens of %), xTB non-spin-polarised.
