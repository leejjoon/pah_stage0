"""Domain screen of an mlmd_run.py output (plan: docs/plan/d1_hangang_new_species.md, step 2): for every system,
the driver's QA flags plus the bond topology at the last checkpoint against the starting geometry.
A system fails if it is non-finite, hot (T > 30,000 K), has more than 25 neighbours, sits in a unit with edge
overflow, or changed its bonds:
  - an H whose nearest heavy atom changed (H hop; nearest-atom test, so C-H stretching does not count);
  - an H more than 2.0 A from every heavy atom (H lost, i.e. dissociated);
  - a heavy-atom pair bonded at the start (d < 1.15 (r_i + r_j)) that is more than 2.0 A apart at the end;
  - a heavy-atom pair not bonded at the start that is closer than 1.6 A at the end.
A species passes only if all its systems pass. Usage (container A, numpy):
  python d1/screen_check.py XYZ_DIR OUT result.csv
"""
import sys, os, json, csv, glob
import numpy as np

R = {"C": 0.76, "H": 0.31, "N": 0.71, "O": 0.66}
BROKEN, FORMED, MAX_NEIGH, H_LOST = 2.0, 1.6, 25, 2.0


def read_xyz(path):
    with open(path) as f:
        n = int(f.readline()); f.readline()
        rows = [f.readline().split() for _ in range(n)]
    return [r[0] for r in rows], np.array([[float(x) for x in r[1:4]] for r in rows])


def topology_change(sym, x0, x1):
    sym = np.array(sym); hv, hh = np.where(sym != "H")[0], np.where(sym == "H")[0]
    rad = np.array([R[s] for s in sym])
    out = []
    if len(hh):
        n0 = hv[np.linalg.norm(x0[hh, None] - x0[None, hv], axis=-1).argmin(1)]
        d1h = np.linalg.norm(x1[hh, None] - x1[None, hv], axis=-1)
        n1, lost = hv[d1h.argmin(1)], d1h.min(1) > H_LOST
        moved = (n0 != n1) & ~lost
        if moved.any():
            out.append("H moved (%d)" % int(moved.sum()))
        if lost.any():
            out.append("H lost (%d)" % int(lost.sum()))
    d0 = np.linalg.norm(x0[hv, None] - x0[None, hv], axis=-1)
    d1 = np.linalg.norm(x1[hv, None] - x1[None, hv], axis=-1)
    bonded = (d0 < 1.15 * (rad[hv, None] + rad[None, hv])) & ~np.eye(len(hv), dtype=bool)
    nb = int(np.triu(bonded & (d1 > BROKEN)).sum())
    nf = int(np.triu(~bonded & (d1 < FORMED), 1).sum())
    if nb:
        out.append("bonds broken (%d)" % nb)
    if nf:
        out.append("bonds formed (%d)" % nf)
    return out


def main():
    xyz, out, res = sys.argv[1:4]
    rows = []
    for udir in sorted(glob.glob(os.path.join(out, "units", "*"))):
        m = json.load(open(os.path.join(udir, "meta.json")))
        done = os.path.exists(os.path.join(udir, "DONE"))
        p = np.load(os.path.join(udir, "state.npz"))["p"]
        fl = m["flags"]
        for i, s in enumerate(m["systems"]):
            sym, x0 = read_xyz(os.path.join(xyz, s["species"] + ".xyz"))
            why = []
            if not done:
                why.append("unit not done")
            if fl["nonfinite"][i]:
                why.append("non-finite")
            if fl["hot"][i]:
                why.append("hot")
            if fl["max_neighbours"][i] > MAX_NEIGH:
                why.append("neighbours %d" % fl["max_neighbours"][i])
            if fl["edge_overflow"]:
                why.append("edge overflow")
            if np.isfinite(p[i]).all():
                why += topology_change(sym, x0, p[i])
            rows.append({"species": s["species"], "T": s["T"], "replica": s["replica"], "unit": m["id"],
                         "pass": not why, "why": "; ".join(why)})
    with open(res, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    sp = {}
    for r in rows:
        sp.setdefault(r["species"], []).append(r)
    bad = {k: v for k, v in sp.items() if not all(r["pass"] for r in v)}
    print("%d systems, %d species: %d pass, %d fail -> %s" % (len(rows), len(sp), len(sp) - len(bad), len(bad), res))
    for k, v in sorted(bad.items()):
        print("  FAIL %-24s %s" % (k, " | ".join("T%g r%d: %s" % (r["T"], r["replica"], r["why"]) for r in v if not r["pass"])))


if __name__ == "__main__":
    main()
