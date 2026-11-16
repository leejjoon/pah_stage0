#!/usr/bin/env python3
"""Non-Kekule PAHs listed as closed-shell singlets in the NASA Ames PAH IR Spectroscopic Database (computational
v4.00), compared with v3.20. Reproduces every number and figure of README.md.

    python pahdb_nonkekule.py pahdb-complete-theoretical-v3.20.xml.gz pahdb-complete-theoretical-v4.00.xml.gz OUTDIR

Needs only numpy and matplotlib. For every neutral pure C/H species without sp3 carbon it:
  - builds the bond graph from the geometry (distance < 1.15 x sum of covalent radii, C 0.76 / H 0.31 A);
  - takes the sp2 carbon framework (C atoms with 3 neighbours), checks that it is bipartite (alternant) and whether it has
    a perfect matching, i.e. a Kekule structure (Hopcroft-Karp-style augmenting paths), and the sublattice imbalance
    nA - nB (for an alternant hydrocarbon, |nA - nB| = 2S of the ground state, Lieb 1989);
  - reads the electronic state from <symmetry> (e.g. "1-A'" -> multiplicity 1) and the harmonic intensities;
  - computes the total intensity per carbon atom and the share of the CH stretch (> 2700 cm^-1).
Writes OUTDIR/species_v{3.20,4.00}.csv, OUTDIR/affected_v4.00.csv, OUTDIR/table.md and OUTDIR/fig{1,2,3}.png.
"""
import csv, gzip, os, sys
import xml.etree.ElementTree as ET
import numpy as np

NS = "{http://www.astrochemistry.org/pahdb/theoretical}"
RCOV = {"H": 0.31, "C": 0.76}
SYM = {1: "H", 6: "C"}
sys.setrecursionlimit(20000)


def bonds(el, xyz):
    r = np.array([RCOV[e] for e in el]); d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    b = (d < 1.15 * (r[:, None] + r[None])) & ~np.eye(len(el), dtype=bool)
    return [list(np.nonzero(row)[0]) for row in b]


def kekule(el, nb):
    """(has a Kekule structure, sublattice imbalance), or (None, None) if the sp2 framework is not bipartite."""
    pi = [i for i, e in enumerate(el) if e == "C" and len(nb[i]) == 3]; s = set(pi)
    adj = {i: [j for j in nb[i] if j in s] for i in pi}; col = {}
    for st in pi:
        if st in col:
            continue
        col[st] = 0; stack = [st]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in col:
                    col[v] = 1 - col[u]; stack.append(v)
                elif col[v] == col[u]:
                    return None, None
    A = [i for i in pi if col[i] == 0]; B = [i for i in pi if col[i] == 1]; match = {}

    def augment(u, seen):
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                if v not in match or augment(match[v], seen):
                    match[v] = u; return True
        return False
    m = sum(augment(u, set()) for u in A)
    return len(A) == len(B) == m, len(A) - len(B)


def survey(xml_gz):
    rows = []
    with gzip.open(xml_gz) as f:
        for _, sp in ET.iterparse(f, events=("end",)):
            if sp.tag != NS + "specie":
                continue
            atoms = sp.find(NS + "geometry").findall(NS + "atom")
            el = [SYM.get(int(a.find(NS + "type").text), "X") for a in atoms]
            if int(sp.find(NS + "charge").text) != 0 or set(el) - {"C", "H"} or "H" not in el:
                sp.clear(); continue
            xyz = np.array([[float(a.find(NS + t).text) for t in ("x", "y", "z")] for a in atoms])
            nb = bonds(el, xyz)
            if any(e == "C" and len(nb[i]) == 4 for i, e in enumerate(el)):        # sp3 carbon: skip
                sp.clear(); continue
            k, imb = kekule(el, nb)
            modes = sp.find(NS + "transitions").findall(NS + "mode")
            nu = np.array([float(m.find(NS + "frequency").text) for m in modes])
            I = np.array([float(m.find(NS + "intensity").text) for m in modes])
            state = sp.find(NS + "symmetry").text or ""
            nC = el.count("C")
            rows.append(dict(uid=int(sp.get("uid")), formula=sp.find(NS + "formula").text, state=state,
                             mult=int(state.split("-")[0]) if state[:1].isdigit() else 0, nC=nC,
                             kekule="" if k is None else int(k), imbalance="" if imb is None else imb,
                             I_total=round(float(I.sum()), 2), I_per_C=round(float(I.sum()) / nC, 2),
                             CH_share=round(float(I[nu > 2700].sum() / I.sum()), 4),
                             nu=" ".join(f"{x:.2f}" for x in nu), I=" ".join(f"{x:.3f}" for x in I)))
            sp.clear()
    return rows


def cls(r):
    if r["kekule"] == "":
        return "non-alternant"
    if r["kekule"] == 1:
        return "Kekule"
    if abs(r["imbalance"]) % 2 == 1:
        return "non-Kekule, odd imbalance (radical)"
    return "non-Kekule, even imbalance, listed as singlet" if r["mult"] == 1 else \
        f"non-Kekule, even imbalance, listed as multiplicity {r['mult']}"


def main(x320, x400, out):
    os.makedirs(out, exist_ok=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    data = {}
    for v, x in (("3.20", x320), ("4.00", x400)):
        rows = survey(x); data[v] = rows
        with open(f"{out}/species_v{v}.csv", "w", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=[k for k in rows[0] if k not in ("nu", "I")], extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
    # table
    classes = ["Kekule", "non-Kekule, odd imbalance (radical)", "non-Kekule, even imbalance, listed as multiplicity 3",
               "non-Kekule, even imbalance, listed as singlet", "non-alternant"]
    lines = ["| Class (neutral, pure C/H, no sp3 C) | v3.20: n | median km/mol per C | median CH share | v4.00: n | median km/mol per C | median CH share |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for c in classes:
        cells = []
        for v in ("3.20", "4.00"):
            s = [r for r in data[v] if cls(r) == c]
            cells += [str(len(s)), f"{np.median([r['I_per_C'] for r in s]):.1f}" if s else "-",
                      f"{np.median([r['CH_share'] for r in s]):.2f}" if s else "-"]
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    bad = {v: [r for r in data[v] if cls(r) == "non-Kekule, even imbalance, listed as singlet"] for v in data}
    new = [r for r in bad["4.00"] if r["uid"] not in {b["uid"] for b in bad["3.20"]}]
    lines += ["", f"Listed as singlet without a Kekule structure: v3.20 {len(bad['3.20'])} (uid {', '.join(str(r['uid']) for r in bad['3.20'])}); "
              f"v4.00 {len(bad['4.00'])}, of which {len(new)} not in v3.20 (uid {min(r['uid'] for r in new)}-{max(r['uid'] for r in new)}); "
              f"v4.00 imbalance counts: " + ", ".join(f"|{k}|: {sum(abs(r['imbalance']) == k for r in bad['4.00'])}" for k in sorted({abs(r['imbalance']) for r in bad['4.00']}))]
    open(f"{out}/table.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
    with open(f"{out}/affected_v4.00.csv", "w", newline="") as fo:      # the list for the database team
        w = csv.writer(fo); w.writerow(["uid", "formula", "state", "sublattice_imbalance", "I_total_km_mol", "I_per_C", "CH_share", "in_v3.20"])
        old = {b["uid"] for b in bad["3.20"]}
        for r in sorted(bad["4.00"], key=lambda r: r["uid"]):
            w.writerow([r["uid"], r["formula"], r["state"], r["imbalance"], r["I_total"], r["I_per_C"], r["CH_share"], int(r["uid"] in old)])

    colors = {"Kekule": "#2a78d6", "non-Kekule, odd imbalance (radical)": "#1baf7a",
              "non-Kekule, even imbalance, listed as multiplicity 3": "#eda100",
              "non-Kekule, even imbalance, listed as singlet": "#e34948"}
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e4e3df", "grid.linewidth": 0.6, "legend.frameon": False})
    # fig 1: intensity per C vs CH share, both versions
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True, sharex=True)
    for ax, v in zip(axs, ("3.20", "4.00")):
        for c, col in colors.items():
            s = [r for r in data[v] if cls(r) == c]
            if s:
                ax.scatter([r["I_per_C"] for r in s], [100 * r["CH_share"] for r in s], s=8 if c != classes[3] else 14,
                           color=col, linewidth=0, alpha=0.8, label=f"{c} ({len(s)})")
        ax.set_xscale("log"); ax.set_xlabel("total harmonic intensity per C atom (km/mol)"); ax.set_title(f"PAHdb v{v}", loc="left")
        ax.legend(fontsize=7, loc="upper right")
    axs[0].set_ylabel("CH-stretch share of the intensity (%)")
    fig.suptitle("Neutral pure C/H PAHs (no sp3 C): non-Kekule species listed as singlets (red) have 5x the intensity and a "
                 "collapsed CH stretch", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{out}/fig1.png", dpi=140); plt.close(fig)
    # fig 2: uid of the singlet non-Kekule species against all v4.00 neutrals
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist([r["uid"] for r in data["4.00"]], bins=np.arange(0, 11000, 250), color="#9ec5f4", label="all v4.00 species in this set")
    ax.hist([r["uid"] for r in bad["4.00"]], bins=np.arange(0, 11000, 250), color="#e34948", label="non-Kekule, listed as singlet")
    ax.axvline(max(r["uid"] for r in data["3.20"]), color="#52514e", ls=":", lw=1); ax.text(max(r["uid"] for r in data["3.20"]) + 80, ax.get_ylim()[1] * 0.85, "largest uid in v3.20", fontsize=8, color="#52514e")
    ax.set_xlabel("uid"); ax.set_ylabel("species"); ax.legend(fontsize=8, loc="upper left")
    ax.set_title("PAHdb v4.00: the non-Kekule singlets are among the species added after v3.20", loc="left")
    fig.tight_layout(); fig.savefig(f"{out}/fig2.png", dpi=140); plt.close(fig)
    # fig 3: a typical example (CH share nearest the class median) next to a Kekule isomer of the same formula
    by = {}
    for r in data["4.00"]:
        by.setdefault(r["formula"], []).append(r)
    med = np.median([r["CH_share"] for r in bad["4.00"]])
    cands = [b for f, v in by.items() if any(cls(x) == "Kekule" for x in v) for b in v if cls(b) == classes[3]]
    a = min(cands, key=lambda b: abs(b["CH_share"] - med)); f = a["formula"]
    k = next(b for b in by[f] if cls(b) == "Kekule")
    x = np.arange(500, 3300, 1.0)
    fig, axs = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
    for ax, (r, col, lab) in zip(axs, ((k, "#2a78d6", "Kekule isomer"), (a, "#e34948", "non-Kekule, listed as singlet"))):
        nu = np.array(r["nu"].split(), float); I = np.array(r["I"].split(), float)
        y = sum(i * 5 ** 2 / ((x - n) ** 2 + 5 ** 2) / (np.pi * 5) for n, i in zip(nu, I))
        ax.plot(x, y, color=col, lw=1.2)
        ax.set_title(f"uid {r['uid']}: {lab}; state {r['state']}; total {r['I_total']:.0f} km/mol; CH stretch {100 * r['CH_share']:.0f} %",
                     loc="left", fontsize=9)
        ax.set_ylabel("km/mol per cm$^{-1}$")
    axs[1].set_xlabel("wavenumber (cm$^{-1}$, as listed; lines broadened to 10 cm$^{-1}$ FWHM)")
    fig.suptitle(f"{f}, two isomers in v4.00 (note the different y scales)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{out}/fig3.png", dpi=140); plt.close(fig)
    print("figures and tables in", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
