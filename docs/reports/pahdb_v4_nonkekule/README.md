# PAHdb v4.00: 334 open-shell PAHs listed as closed-shell, with unrealistic intensities

## The issue

In the NASA Ames PAH IR Spectroscopic Database, computational version 4.00, **334 neutral PAHs (pure C/H) are listed as
singlets, although their structure cannot be closed-shell**. Their carbon framework has no Kekulé structure; for
most of them the ground state should be a triplet.

Their computed infrared spectra are clearly wrong:
- the total intensity is **5× higher than normal**: a median of 110 km/mol per carbon atom, against 23.5;
- the CH-stretch band nearly disappears: a share of 11 % of the intensity, against 41 %.

The cause is almost certainly the wrong spin state. The same kind of molecule listed as a **triplet** (51 species) has
normal spectra. **331 of the 334 are new in v4.00**; v3.20 had only three such cases, and those are only mildly
affected.

| Neutral PAHs (pure C/H) | v3.20 | v4.00 | Intensity per C (km/mol) | CH-stretch share |
| --- | --- | --- | --- | --- |
| Normal (Kekulé structure) | 1,251 | 1,882 | 23.5 | 41 % |
| No Kekulé structure, listed as **triplet** | 51 | 51 | 22.9 | 43 % |
| No Kekulé structure, listed as **singlet** | **3** | **334** | **110** | **11 %** |

The intensity and share columns are v4.00 medians.

![Figure 1](out/fig1.png)

**Figure 1.** Every neutral PAH in v3.20 (left) and v4.00 (right). Red points are the open-shell PAHs listed as
singlets. They are nearly absent in v3.20 and form a separate, far too intense branch in v4.00.

![Figure 3](out/fig3.png)

**Figure 2.** A typical case: two isomers of C64H22 in v4.00. Top: a normal one. Bottom: an affected one, with 4× the
intensity and a weak CH stretch (note the different scales).

## Suggested action

Flag the 334 species, listed in `out/affected_v4.00.csv`, and recompute them in the correct (high-spin) state, as v3.20
did for its triplets. Until then, users should exclude them from libraries and from training data.

## Reproduce

`python pahdb_nonkekule.py pahdb-complete-theoretical-v3.20.xml.gz pahdb-complete-theoretical-v4.00.xml.gz out`

This needs Python 3 with numpy and matplotlib, and takes ~30 s. It builds the bond graph from each geometry, tests for a
Kekulé structure, and reads the listed spin state and the intensities. N-containing PAHs are not tested.
