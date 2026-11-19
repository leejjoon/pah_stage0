#!/usr/bin/env python3
"""Summary of the high-spin recomputation of the PAHdb v4.00 non-Kekule species (d1/triplet_batch.py output).

    python triplet_summary.py <pahdb v4.00 xml.gz> <triplet out dir> <exclude csv> <out csv> <out png>

For every <uid>.json in the output directory: PAHdb (listed singlet) total intensity and C-H stretch share against the
recomputed high-spin values, plus checks (imaginary modes, <S^2>, optimisation energy change). For reference, the same
quantities for the normal (Kekule) neutral pure C/H PAHdb species of the same carbon range. C-H share = intensity above
2500 cm^-1 / total; computed frequencies are scaled with PAHdb v4.00's factors (0.9794 / 0.9691 / 0.9597, as in own_spectra.py) first. Needs numpy,
matplotlib.
"""
import csv, glob, gzip, json, os, re, sys
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NS = "{http://www.astrochemistry.org/pahdb/theoretical}"


def scale(nu):
    return nu * np.where(nu < 1111.05, 0.9794, np.where(nu < 1800.05, 0.9691, 0.9597))


def counts(formula):
    d = dict((e, int(n or 1)) for e, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula))
    return d


def pahdb(xml, nmin, nmax):
    """uid -> (formula, nC, total, ch_share) for neutral pure C/H species with nmin <= nC <= nmax."""
    out = {}
    with gzip.open(xml) as f:
        for _, sp in ET.iterparse(f, events=("end",)):
            if sp.tag != NS + "specie":
                continue
            form = sp.find(NS + "formula").text; c = counts(form)
            if int(sp.find(NS + "charge").text) == 0 and set(c) <= {"C", "H"} and nmin <= c.get("C", 0) <= nmax:
                ms = sp.find(NS + "transitions").findall(NS + "mode")
                nu = np.array([float(m.find(NS + "frequency").text) for m in ms])
                I = np.array([float(m.find(NS + "intensity").text) for m in ms])
                out[int(sp.get("uid"))] = (form, c["C"], I.sum(), I[nu > 2500].sum() / I.sum())
            sp.clear()
    return out


def main(xml, odir, excl, out_csv, out_png):
    res = {int(os.path.basename(p)[:-5]): json.load(open(p)) for p in glob.glob(f"{odir}/*.json")}
    bad = {int(r["uid"]) for r in csv.DictReader(open(excl))}
    db = pahdb(xml, 20, 90)
    rows = []
    for u, d in sorted(res.items()):
        nu, I = scale(np.array(d["freq"])), np.array(d["I"]); ok = nu > 0
        tot = I[ok].sum(); ch = I[ok & (nu > 2500)].sum() / tot
        s = d["mult"] - 1; s2_exact = s / 2 * (s / 2 + 1)
        rows.append(dict(uid=u, formula=d["formula"], nC=counts(d["formula"])["C"], mult=d["mult"],
                         pahdb_total=round(db[u][2], 1), new_total=round(tot, 1), ratio=round(db[u][2] / tot, 2),
                         pahdb_ch_share=round(db[u][3], 3), new_ch_share=round(ch, 3), n_imag=d["n_imag"],
                         lowest_cm1=d["lowest"], S2=round(d["S2"], 3), S2_exact=s2_exact,
                         dE_opt_kcal=d.get("dE_opt_kcal"), hours=round(d["t_total"] / 3600, 2)))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    ref = [v for u, v in db.items() if u not in bad]
    r = {k: np.array([x[k] for x in rows], float) for k in ("nC", "pahdb_total", "new_total", "ratio", "pahdb_ch_share",
                                                             "new_ch_share", "S2", "S2_exact", "n_imag", "dE_opt_kcal")}
    print(f"{len(rows)} species; mult counts: { {m: sum(x['mult'] == m for x in rows) for m in sorted({x['mult'] for x in rows})} }")
    print(f"imaginary modes: {int((r['n_imag'] > 0).sum())} species; lowest frequency min {min(x['lowest_cm1'] for x in rows):.1f} cm^-1")
    print(f"<S^2> - exact: median {np.median(r['S2'] - r['S2_exact']):.3f}, max {np.max(r['S2'] - r['S2_exact']):.3f}")
    print(f"optimisation energy change (kcal/mol): median {np.median(r['dE_opt_kcal']):.2f}, min {np.min(r['dE_opt_kcal']):.2f}")
    print(f"total intensity per C (km/mol): PAHdb listed median {np.median(r['pahdb_total'] / r['nC']):.1f}, "
          f"recomputed {np.median(r['new_total'] / r['nC']):.1f}, normal PAHdb neutrals {np.median([x[2] / x[1] for x in ref]):.1f}")
    print(f"PAHdb / recomputed total: median {np.median(r['ratio']):.1f}, 16-84% {np.percentile(r['ratio'], 16):.1f}-{np.percentile(r['ratio'], 84):.1f}, min {r['ratio'].min():.1f}")
    print(f"C-H share: PAHdb listed median {np.median(r['pahdb_ch_share']):.3f}, recomputed {np.median(r['new_ch_share']):.3f}, "
          f"normal {np.median([x[3] for x in ref]):.3f}")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    a = ax[0]
    a.scatter([x[1] for x in ref], [x[2] / x[1] for x in ref], s=6, c="#b4b2a9", label=f"normal PAHdb neutrals ({len(ref)})")
    a.scatter(r["nC"], r["pahdb_total"] / r["nC"], s=10, c="#c2410c", label="non-Kekule, as listed in PAHdb")
    a.scatter(r["nC"], r["new_total"] / r["nC"], s=10, c="#1d4ed8", label="non-Kekule, recomputed high-spin")
    a.set_yscale("log"); a.set_xlabel("number of carbon atoms"); a.set_ylabel("total IR intensity per C (km/mol)")
    a.set_title("Total intensity"); a.legend(fontsize=8, frameon=False)
    a = ax[1]; bins = np.linspace(0, 0.7, 36)
    a.hist([x[3] for x in ref], bins, density=True, color="#b4b2a9", label="normal PAHdb neutrals")
    a.hist(r["pahdb_ch_share"], bins, density=True, histtype="step", lw=2, color="#c2410c", label="non-Kekule, as listed")
    a.hist(r["new_ch_share"], bins, density=True, histtype="step", lw=2, color="#1d4ed8", label="non-Kekule, recomputed")
    a.set_xlabel("C-H stretch share of total intensity"); a.set_ylabel("species (normalised)")
    a.set_title("3.3 um share"); a.legend(fontsize=8, frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out_png, dpi=130)


if __name__ == "__main__":
    main(*sys.argv[1:6])
