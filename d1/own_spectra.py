#!/usr/bin/env python3
"""Our own harmonic spectra that replace PAHdb v4.00 entries ("own" species), and a loader for them.

PAHdb v4.00 lists 334 neutral pure C/H PAHs without a Kekule structure as closed-shell singlets with intensities ~7x too
high (docs/reports/pahdb_v4_nonkekule/). The 200 with nC < 90 were recomputed in their high-spin ground state
(d1/triplet_batch.py -> d1/triplet_freq.py --opt: UB3LYP/6-31G* (6D), (99,590) grid, GPU4PySCF 1.9.2, geometry
re-optimised, analytic Hessian and IR intensities). Those results are OUR OWN data, not PAHdb's:
  - species id  <formula>_v4u<uid>_own  (the "_own" suffix makes code that parses int(id.split("_v4u")[1]) fail
    instead of silently loading the PAHdb entry);
  - every file carries source = "own" and the method string; the uid is the PAHdb v4.00 uid of the structure only.
The other 134 (nC >= 90) stay excluded (d1/pahdb_v4_replacements_v1.csv: action replace / exclude).

    python own_spectra.py build <triplet out dir> <pahdb v4.00 xml.gz> <exclude csv> <out dir>
writes <out dir>/nonkekule_highspin_v1_species.csv, _sticks.csv, _geom.xyz and d1/pahdb_v4_replacements_v1.csv beside
this file. Frequencies: unscaled and scaled with PAHdb v4.00's per-mode B3LYP/6-31G* factors (0.9794 below 1111 cm^-1,
0.9691 for 1111-1800, 0.9597 above; read from the scale attributes of the v4.00 XML), so they sit on PAHdb's axis.

    from own_spectra import load_own_sticks; st = load_own_sticks()   # {species id: (scaled freq, intensity)}
"""
import csv, glob, gzip, json, os, re, sys
import xml.etree.ElementTree as ET
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SET = "nonkekule_highspin_v1"
DEFAULT_DIR = f"{HERE}/own_spectra"
METHOD = "UB3LYP(b3lypg)/6-31G*(6D), grid (99,590), GPU4PySCF 1.9.2, re-optimised, analytic Hessian+IR"
NS = "{http://www.astrochemistry.org/pahdb/theoretical}"


def pahdb_scale(nu):
    nu = np.asarray(nu, float)
    return nu * np.where(nu < 1111.05, 0.9794, np.where(nu < 1800.05, 0.9691, 0.9597))


def species_id(formula, uid):
    return f"{formula}_v4u{uid}_own"


def load_own_sticks(d=DEFAULT_DIR, scaled=True):
    st = {}
    for r in csv.DictReader(open(f"{d}/{SET}_sticks.csv")):
        f, i = st.setdefault(r["species"], ([], []))
        f.append(float(r["freq_scaled" if scaled else "freq_unscaled"])); i.append(float(r["intensity_km_mol"]))
    return {k: (np.array(f), np.array(i)) for k, (f, i) in st.items()}


def build(odir, xml, excl, out):
    os.makedirs(out, exist_ok=True)
    res = {int(os.path.basename(p)[:-5]): json.load(open(p)) for p in glob.glob(f"{odir}/*.json")}
    db = {}
    with gzip.open(xml) as f:
        for _, sp in ET.iterparse(f, events=("end",)):
            if sp.tag == NS + "specie":
                if int(sp.get("uid")) in res:
                    db[int(sp.get("uid"))] = (sp.find(NS + "formula").text, sp.find(NS + "symmetry").text,
                                              sum(float(m.find(NS + "intensity").text)
                                                  for m in sp.find(NS + "transitions").findall(NS + "mode")))
                sp.clear()
    sp_rows, st_rows, xyz = [], [], []
    for u, d in sorted(res.items(), key=lambda kv: (int(re.match(r"C(\d+)", kv[1]["formula"]).group(1)), kv[0])):
        assert db[u][0] == d["formula"], (u, db[u][0], d["formula"])
        sid = species_id(d["formula"], u); nu = np.array(d["freq"]); I = np.array(d["I"]); sc = pahdb_scale(nu)
        nC, nH = (int(x) for x in re.match(r"C(\d+)H(\d+)", d["formula"]).groups()); s = (d["mult"] - 1) / 2
        sp_rows.append(dict(species=sid, source="own", pahdb_v4_uid=u, formula=d["formula"], nC=nC, nH=nH, charge=d["charge"],
                            mult=d["mult"], method=METHOD, E_Ha=d["E_opt"], S2=round(d["S2"], 4), S2_exact=s * (s + 1),
                            n_imag=d["n_imag"], lowest_cm1=d["lowest"], total_km_mol=round(float(I[nu > 0].sum()), 2),
                            pahdb_state_listed=db[u][1], pahdb_total_km_mol=round(db[u][2], 2), date="2026-11-19",
                            run="/appsdata/t641a02/d1/triplet_v1 (Hangang production job 1302424 and the earlier tests)"))
        for k, (a, b, c) in enumerate(zip(nu, sc, I)):
            st_rows.append(dict(species=sid, mode=k, freq_unscaled=round(float(a), 3), freq_scaled=round(float(b), 3),
                                intensity_km_mol=round(float(c), 5)))
        xyz += [str(len(d["sym"])), f"{sid} source=own mult={d['mult']} E_Ha={d['E_opt']:.8f} {METHOD}"]
        xyz += [f"{e} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for e, p in zip(d["sym"], d["pos"])]
    for name, rows in (("species", sp_rows), ("sticks", st_rows)):
        with open(f"{out}/{SET}_{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    open(f"{out}/{SET}_geom.xyz", "w").write("\n".join(xyz) + "\n")
    own = {r["pahdb_v4_uid"]: r["species"] for r in sp_rows}
    rep = []
    for r in csv.DictReader(open(excl)):
        u = int(r["uid"])
        rep.append(dict(pahdb_v4_uid=u, formula=r["formula"], action="replace" if u in own else "exclude",
                        species=own.get(u, ""), source="own" if u in own else "",
                        note=f"own high-spin spectrum ({SET})" if u in own else "nC >= 90, not recomputed"))
    with open(f"{HERE}/pahdb_v4_replacements_v1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rep[0])); w.writeheader(); w.writerows(rep)
    print(f"{len(sp_rows)} own species, {len(st_rows)} sticks -> {out}; replacements: "
          f"{sum(r['action'] == 'replace' for r in rep)} replace, {sum(r['action'] == 'exclude' for r in rep)} exclude")


if __name__ == "__main__":
    if sys.argv[1:2] == ["build"]:
        build(*sys.argv[2:6])
    else:
        sys.exit(__doc__)
