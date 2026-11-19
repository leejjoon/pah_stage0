# Own harmonic spectra (not PAHdb)

Spectra **we computed ourselves**. They replace PAHdb v4.00 entries that are wrong. Never mix them up with PAHdb
data or cite them as PAHdb.

## `nonkekule_highspin_v1` (2026-11-19)

- **What:** 200 neutral pure C/H PAHs, 30–88 C, that have no Kekulé structure.
- **The problem in PAHdb:** PAHdb v4.00 lists them as closed-shell singlets, with intensities about 7× too high
  (`docs/reports/pahdb_v4_nonkekule/`).
- **What we did:** recomputed them in their high-spin ground state: 198 triplets and 2 quintets.
  - Method: UB3LYP/6-31G* (6D), (99,590) grid, GPU4PySCF 1.9.2 (container D).
  - Each geometry was re-optimised, then the analytic Hessian and IR intensities were computed (`d1/uks_ir.py`).
  - Checks: no imaginary modes; ⟨S²⟩ is 0.15 above the exact value at the median.
- **Identifiers:**
  - Species id `<formula>_v4u<uid>_own`. The uid only names the PAHdb v4.00 structure, not the PAHdb data.
  - Every row has `source = own`.
- **Built by:** `python d1/own_spectra.py build ...`, from `/appsdata/t641a02/d1/triplet_v1/`.

### Files

| File | Content |
|---|---|
| `nonkekule_highspin_v1_species.csv` | One row per species: method, energy, ⟨S²⟩, checks, our total intensity and PAHdb's |
| `nonkekule_highspin_v1_sticks.csv` | All 3N−6 modes: unscaled frequency, frequency scaled with PAHdb's own factors, intensity (km/mol) |
| `nonkekule_highspin_v1_geom.xyz` | Optimised geometries (Å) |

### How to use them

- **Loading:** `from own_spectra import load_own_sticks`. Scaled frequencies use 0.9794 below 1111 cm⁻¹, 0.9691 for
  1111–1800 cm⁻¹ and 0.9597 above, the same as PAHdb v4.00. They therefore sit on the same axis as PAHdb sticks.
- **PAHdb entries:** `d1/pahdb_v4_replacements_v1.csv` covers all 334 affected PAHdb entries.
  - 200 are marked `replace`, pointing to the species above.
  - 134 are marked `exclude`: the species with 90 C or more, which were not recomputed.

### Not included

There are no MLMD (temperature) spectra for these species. Mai et al.'s model is closed-shell only.
