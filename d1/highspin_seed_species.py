#!/usr/bin/env python3
"""Pick 10 non-Kekule (high-spin ground state) neutrals for the charged-MLFF seed set, with their PAHdb cations.
Handoff: docs/plan/handoff_highspin_to_charged_mlff.md.

    python highspin_seed_species.py <pahdb v4.00 xml.gz> <out dir>     (v1: out dir d1/highspin_seed_v1)

Pool: the 198 triplets of d1/own_spectra/nonkekule_highspin_v1_species.csv (quintets left out), minus the two already in
the charged-MLFF seed set (C60H20_10225 train, C72H22_9911 transfer_test; relabelled separately), minus species whose
cation is not in PAHdb v4.00 (matched by the bond-graph hash of mlff_species.py on the PAHdb geometries).
Per carbon bin, farthest-point sampling in (solo, duo, trio, quartet) edge fractions, uid 10770 (C30H14, the checked
case of the PAHdb report) first: 30-39 C: 3, 40-49 C: 2, 50-60 C: 3 (6 train, 1 validation, 1 final_test, assigned
with a fixed seed), 61-88 C: 2 (transfer_test). Writes <out>/species.csv and <out>/xyz/<pair>_q0.xyz (OUR UB3LYP
triplet minimum, comment line in the plain style of mlff_species.py, no key=value) and <out>/xyz/<pair>_q1.xyz (PAHdb cation geometry).
"""
import csv, gzip, os, re, sys
from xml.etree.ElementTree import iterparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mlff_species import graph_hash, kid, L  # noqa: E402

ALREADY = {10225, 9911}
BINS = [((30, 39), 3), ((40, 49), 2), ((50, 60), 3), ((61, 88), 2)]
MUST = 10770


def pahdb(xml):
    out = {}
    with gzip.open(xml) as f:
        for _, el in iterparse(f, events=("end",)):
            if L(el) != "specie":
                continue
            q = int(kid(el, "charge").text)
            if q in (0, 1):
                atoms = [(int(kid(a, "type").text), [float(kid(a, c).text) for c in "xyz"]) for a in kid(el, "geometry")]
                Z = [z for z, _ in atoms]
                if set(Z) == {1, 6}:
                    out[int(el.get("uid"))] = dict(q=q, formula=kid(el, "formula").text, method=kid(el, "method").text,
                                                   state=(kid(el, "symmetry").text or "").strip(), Z=Z,
                                                   P=np.array([p for _, p in atoms]),
                                                   edges=[int(kid(el, f"n_{k}").text) for k in ("solo", "duo", "trio", "quartet")])
            el.clear()
    return out


def main(xml, out):
    own = [r for r in csv.DictReader(open(f"{HERE}/own_spectra/nonkekule_highspin_v1_species.csv")) if r["mult"] == "3"]
    db = pahdb(xml)
    cat = {}
    for u, d in db.items():
        if d["q"] == 1:
            cat.setdefault(graph_hash(d["Z"], d["P"]), []).append(u)
    pool = []
    for r in own:
        u = int(r["pahdb_v4_uid"])
        if u in ALREADY:
            continue
        c = cat.get(graph_hash(db[u]["Z"], db[u]["P"]), [])
        if c:
            e = np.array(db[u]["edges"], float)
            pool.append(dict(r, uid=u, nC=int(r["nC"]), cat=min(c), frac=e / e.sum(), edges=db[u]["edges"]))
    print(f"pool: {len(pool)} triplets with a PAHdb cation (of {len(own)})")
    pick = []
    for (lo, hi), n in BINS:
        cand = [p for p in pool if lo <= p["nC"] <= hi]; sel = [p for p in cand if p["uid"] == MUST][:1]
        if not sel:
            sel = [min(cand, key=lambda p: (np.linalg.norm(p["frac"] - np.mean([q["frac"] for q in cand], 0)), p["uid"]))]
        while len(sel) < n:
            sel.append(max((p for p in cand if p not in sel),
                           key=lambda p: (min(np.linalg.norm(p["frac"] - s["frac"]) for s in sel), -p["uid"])))
        pick += [dict(p, bin=f"{lo}-{hi}") for p in sel]
    small = [p for p in pick if p["nC"] <= 60]; order = np.random.default_rng(0).permutation(len(small))
    for k, i in enumerate(order):
        small[i]["split"] = "train" if k < 6 else ("validation" if k == 6 else "final_test")
    for p in pick:
        p.setdefault("split", "transfer_test")
    os.makedirs(f"{out}/xyz", exist_ok=True)
    geo = {}
    blk = open(f"{HERE}/own_spectra/nonkekule_highspin_v1_geom.xyz").read().split("\n"); i = 0
    while i < len(blk) and blk[i].strip():
        n = int(blk[i]); geo[blk[i + 1].split()[0]] = blk[i:i + 2 + n]; i += 2 + n
    rows = []
    for p in pick:
        pair = f"{p['formula']}_{p['uid']}"; g = geo[p["species"]]
        open(f"{out}/xyz/{pair}_q0.xyz", "w").write("\n".join([g[0], f"OWN (not PAHdb) UB3LYP/6-31G* triplet minimum of PAHdb v4.00 uid {p['uid']} charge 0 "
                                                              f"mult 3 ({p['species']})"] + g[2:]) + "\n")
        c = db[p["cat"]]; sym = {1: "H", 6: "C"}
        open(f"{out}/xyz/{pair}_q1.xyz", "w").write("\n".join(
            [str(len(c["Z"])), f"PAHdb v4.00 uid {p['cat']} charge 1 {c['method']}/6-31g* state {c['state']}"]
            + [f"{sym[z]} {x:.6f} {y:.6f} {w:.6f}" for z, (x, y, w) in zip(c["Z"], c["P"])]) + "\n")
        rows.append(dict(pair=pair, formula=p["formula"], nC=p["nC"], bin=p["bin"], split=p["split"], uid_q0=p["uid"],
                         mult_q0=3, source_q0="own", species_own=p["species"], S2_q0=p["S2"], uid_q1=p["cat"],
                         pahdb_state_q1=c["state"], pahdb_method_q1=c["method"], n_solo=p["edges"][0],
                         n_duo=p["edges"][1], n_trio=p["edges"][2], n_quartet=p["edges"][3]))
    with open(f"{out}/species.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r["pair"], r["split"], r["uid_q1"], r["pahdb_state_q1"], r["S2_q0"], r["n_solo"], r["n_duo"], r["n_trio"], r["n_quartet"])


if __name__ == "__main__":
    main(*sys.argv[1:3])
