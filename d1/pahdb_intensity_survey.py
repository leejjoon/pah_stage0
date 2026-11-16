# Survey of PAHdb v4.00 harmonic intensities: which species have spectra dominated by 6-9 um with very large totals
# (found in 164 of the 917 neutral library species, docs/plan/d1_hangang_new_species.md, "Issue figures")?
#   python pahdb_intensity_survey.py <pahdb xml.gz> <out csv>
# Per species: uid, formula, charge, multiplicity (from the electronic state, e.g. "2-A" -> 2), structure profile of
# profile_xyz.profile_atoms (solo/duo/trio/quartet H, pentagons, sp3 C, bare edge C, N), total intensity and per C,
# shares of CH stretch (> 2700 cm^-1), in-plane (1100-1700) and oop (650-1000), and the strongest mode.
# Standard library + numpy; runs on a login node (Python >= 3.6).
import csv, gzip, os, sys
import xml.etree.ElementTree as ET
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_xyz import profile_atoms  # noqa: E402

NS = "{http://www.astrochemistry.org/pahdb/theoretical}"
SYM = {1: "H", 6: "C", 7: "N", 8: "O", 12: "Mg", 14: "Si", 26: "Fe"}


def main(xml, out):
    rows = []
    with gzip.open(xml) as f:
        for ev, el in ET.iterparse(f, events=("end",)):
            if el.tag != NS + "specie":
                continue
            uid = int(el.get("uid")); form = el.find(NS + "formula").text; q = int(el.find(NS + "charge").text)
            sym = el.find(NS + "symmetry").text or ""
            mult = int(sym.split("-")[0]) if sym[:1].isdigit() else 0
            at = [(SYM.get(int(a.find(NS + "type").text), "X"), float(a.find(NS + "x").text), float(a.find(NS + "y").text),
                   float(a.find(NS + "z").text)) for a in el.find(NS + "geometry").findall(NS + "atom")]
            ms = el.find(NS + "transitions").findall(NS + "mode")
            nu = np.array([float(m.find(NS + "frequency").text) for m in ms]); I = np.array([float(m.find(NS + "intensity").text) for m in ms])
            el.clear()
            if set(a[0] for a in at) - {"C", "H", "N", "O"} or not len(I) or I.sum() <= 0:
                continue
            try:
                p = profile_atoms(at, f"{form}_v4u{uid}")
            except Exception:
                continue
            tot = I.sum(); k = int(np.argmax(I))
            rows.append(dict(uid=uid, formula=form, charge=q, mult=mult, state=sym, nC=p["nC"], nH=p["nH"], nN=p.get("nN", 0),
                             solo=p.get("solo", 0), duo=p.get("duo", 0), trio=p.get("trio", 0), quartet=p.get("quartet", 0),
                             pent=p.get("pent", 0), sp3C=p.get("sp3C", 0), bareC=p.get("bareC", 0),
                             I_total=round(tot, 1), I_perC=round(tot / max(1, p["nC"]), 2),
                             ch_share=round(I[nu > 2700].sum() / tot, 4), ip_share=round(I[(nu >= 1100) & (nu < 1700)].sum() / tot, 4),
                             oop_share=round(I[(nu >= 650) & (nu < 1000)].sum() / tot, 4),
                             max_mode_I=round(I[k], 1), max_mode_nu=round(nu[k], 1)))
    with open(out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} species -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
