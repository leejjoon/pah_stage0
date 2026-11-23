# Ground-state screen of every PAHdb v4.00 species, including N-containing PAHs (PANHs) and ions
# (coordinator request 2026-11-23; extends d1/kekule_check.py, which covers only neutral alternant C/H species).
#   python d1/screen_spin.py <pahdb xml.gz> <out csv> [jobs]
# Question per species: is the multiplicity PAHdb lists compatible with the molecule's bonding? A closed-shell
# (RB3LYP singlet) calculation of a molecule that has no closed-shell Lewis structure gives the intensity artefact
# found for the 334 non-Kekule neutrals (docs/reports/pahdb_v4_nonkekule/).
#
# Method (structure only; no quantum chemistry):
#   1. Electron parity: n_electrons = sum(Z) - charge must be even for odd multiplicity and odd for even.
#   2. Lewis-structure bound. Atoms that need exactly one double bond: C with 3 neighbours, N with 2 neighbours
#      (pyridine type), O with 1 neighbour (carbonyl). N with 3 neighbours (in-ring "graphitic" N, N-H, amine) and O
#      with 2 neighbours keep a lone pair and need none. d0 = number of those atoms left unmatched by a maximum
#      matching of the bond graph (general graph, so pentagons are handled). Each unit of charge can remove one
#      unmatched atom (C+ / C-, or N+ with four bonds), so the smallest number of unpaired electrons is
#          n_unp = d0 - |q|            if d0 >= |q|
#                  (|q| - d0) mod 2    otherwise
#      and the smallest multiplicity is n_unp + 1. Examples: Kekule PAH 1, its cation 2; triangulene 3 (d0 = 2);
#      N-substituted coronene C23H12N neutral 2 (d0 = 1), its cation 1.
#   3. Verdict:
#        ok                listed multiplicity = bound
#        below_bound       listed multiplicity < bound: no Lewis structure with that few unpaired electrons
#                          (for a listed singlet this is the non-Kekule RKS artefact class)
#        above_bound       listed multiplicity > bound (a deliberate high-spin state, or a real high-spin ground state;
#                          not an artefact by itself)
#        parity_error      multiplicity impossible for the electron count
#        not_screened      atoms outside H/C/N/O, or carbons with fewer than 3 / more than 4 neighbours (dehydrogenated
#                          edges, sigma radicals, sp chains): parity only
#   The bound says nothing about open-shell singlets with a Kekule structure (long acenes, zigzag edges) or about
#   high-spin ground states of degenerate systems; those need a stability test.
# Also written: total intensity per C and the CH-stretch share (the empirical signature of the neutral artefact).
import csv, gzip, os, sys
import xml.etree.ElementTree as ET
from multiprocessing import Pool
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_xyz import bonds  # noqa: E402

NS = "{http://www.astrochemistry.org/pahdb/theoretical}"
SYM = {1: "H", 6: "C", 7: "N", 8: "O", 12: "Mg", 14: "Si", 26: "Fe"}


def lewis(at):
    """(d0, note): atoms needing a double bond left unmatched by a maximum matching; note != '' if not screenable."""
    if set(a[0] for a in at) - {"H", "C", "N", "O"}:
        return None, "metal/Si"
    nb = bonds(at)
    need = []
    for i, a in enumerate(at):
        k = len(nb[i])
        if a[0] == "C":
            if k == 3:
                need.append(i)
            elif k != 4:
                return None, f"C with {k} neighbours"
        elif a[0] == "N":
            if k == 2:
                need.append(i)
            elif k not in (3, 4):
                return None, f"N with {k} neighbours"
        elif a[0] == "O":
            if k == 1:
                need.append(i)
            elif k != 2:
                return None, f"O with {k} neighbours"
        elif k != 1:
            return None, f"H with {k} neighbours"
    s = set(need)
    g = nx.Graph(); g.add_nodes_from(need)
    g.add_edges_from((i, j) for i in need for j in nb[i] if j in s and i < j)
    return len(need) - 2 * len(nx.max_weight_matching(g, maxcardinality=True)), ""


def screen(rec):
    at = rec.pop("at"); q = rec["charge"]
    try:
        mult = int(rec["state"].split("-")[0])
    except ValueError:
        mult = None
    Z = {"H": 1, "C": 6, "N": 7, "O": 8, "Mg": 12, "Si": 14, "Fe": 26}
    ne = sum(Z[a[0]] for a in at) - q
    rec.update(mult=mult, n_elec=ne, nC=sum(a[0] == "C" for a in at), nH=sum(a[0] == "H" for a in at),
               nN=sum(a[0] == "N" for a in at), nO=sum(a[0] == "O" for a in at))
    d0, note = lewis(at)
    parity_ok = mult is not None and (ne + mult) % 2 == 1
    rec.update(parity_ok=int(parity_ok), d0="" if d0 is None else d0, mult_min="", note=note)
    if mult is None:
        rec["verdict"] = "no_multiplicity"
    elif not parity_ok:
        rec["verdict"] = "parity_error"
    elif d0 is None:
        rec["verdict"] = "not_screened"
    else:
        mmin = (d0 - abs(q) if d0 >= abs(q) else (abs(q) - d0) % 2) + 1
        rec["mult_min"] = mmin
        rec["verdict"] = "ok" if mult == mmin else "below_bound" if mult < mmin else "above_bound"
    return rec


def read(xml):
    with gzip.open(xml) as f:
        for ev, el in ET.iterparse(f, events=("end",)):
            if el.tag != NS + "specie":
                continue
            at = [(SYM.get(int(a.find(NS + "type").text), "X"), float(a.find(NS + "x").text), float(a.find(NS + "y").text),
                   float(a.find(NS + "z").text)) for a in el.find(NS + "geometry").findall(NS + "atom")]
            tr = [(float(m.find(NS + "frequency").text), float(m.find(NS + "intensity").text))
                  for m in el.find(NS + "transitions").findall(NS + "mode")]
            tot = sum(i for _, i in tr); nC = sum(a[0] == "C" for a in at)
            yield dict(uid=int(el.get("uid")), formula=el.find(NS + "formula").text, charge=int(el.find(NS + "charge").text),
                       state=el.find(NS + "symmetry").text, method=el.find(NS + "method").text, at=at,
                       I_total_km_mol=round(tot, 2), I_per_C=round(tot / max(nC, 1), 2),
                       CH_share=round(sum(i for f, i in tr if f > 2500) / tot, 4) if tot > 0 else "")
            el.clear()


def main(xml, out, jobs=8):
    with Pool(int(jobs)) as pool:
        rows = pool.map(screen, read(xml), chunksize=20)
    cols = ["uid", "formula", "charge", "state", "mult", "method", "nC", "nH", "nN", "nO", "n_elec", "parity_ok", "d0", "mult_min",
            "verdict", "note", "I_total_km_mol", "I_per_C", "CH_share"]
    with open(out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=cols); w.writeheader(); w.writerows(sorted(rows, key=lambda r: r["uid"]))
    from collections import Counter
    grp = lambda r: ("PANH" if r["nN"] else "O" if r["nO"] else "CH" if not r["note"].startswith("metal") else "metal",
                     "neutral" if r["charge"] == 0 else "cation" if r["charge"] > 0 else "anion")
    c = Counter((grp(r), r["verdict"]) for r in rows)
    print(f"{len(rows)} species -> {out}")
    for g in sorted(set(k[0] for k in c)):
        print(f"  {g[0]:6s} {g[1]:8s} " + ", ".join(f"{v} {c[(g, v)]}" for v in
              ("ok", "below_bound", "above_bound", "parity_error", "not_screened", "no_multiplicity") if c[(g, v)]))


if __name__ == "__main__":
    main(*sys.argv[1:4])
