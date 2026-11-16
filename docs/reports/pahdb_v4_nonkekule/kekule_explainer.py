#!/usr/bin/env python3
"""Figures for kekule_explained.md: Kekule structures, the two-colour (sublattice) picture, and spin states.

    python kekule_explainer.py pahdb-complete-theoretical-v4.00.xml.gz OUTDIR

explain1.png  naphthalene, phenalenyl and triangulene built on a honeycomb: carbons coloured by sublattice, a maximum
              set of double bonds (thick), carbons left without a double bond (red rings) and the resulting spin.
explain2.png  the same drawing for two real C64H22 isomers from PAHdb v4.00 (uids 10013 and 10065).
Needs numpy and matplotlib.
"""
import gzip, sys
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A, B, RED, INK, GREY = "#2a78d6", "#eda100", "#e34948", "#0b0b0b", "#b9b8b2"
NS = "{http://www.astrochemistry.org/pahdb/theoretical}"
sys.setrecursionlimit(10000)


def honeycomb(hexes, a=1.4):
    pts = []
    for q, r in hexes:
        cx, cy = np.sqrt(3) * a * (q + r / 2), 1.5 * a * r
        for k in range(6):
            t = np.radians(30 + 60 * k); pts.append((round(cx + a * np.cos(t), 3), round(cy + a * np.sin(t), 3)))
    return np.array(sorted(set(pts)))


def graph(xy, cut=1.6):
    d = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    return [list(np.nonzero((row < cut) & (row > 0.1))[0]) for row in d]


def colour_and_match(nb, pi):
    s = set(pi); adj = {i: [j for j in nb[i] if j in s] for i in pi}; col = {}
    for st in pi:
        if st in col:
            continue
        col[st] = 0; stack = [st]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in col:
                    col[v] = 1 - col[u]; stack.append(v)
    match = {}

    def aug(u, seen):
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                if v not in match or aug(match[v], seen):
                    match[v] = u; return True
        return False
    for u in [i for i in pi if col[i] == 0]:
        aug(u, set())
    pairs = {(min(u, v), max(u, v)) for v, u in match.items()}
    return col, pairs


def draw(ax, xy, nb, pi, title, hxy=None, hbonds=()):
    col, pairs = colour_and_match(nb, pi)
    matched = {i for p in pairs for i in p}
    for i, j in hbonds:
        ax.plot(*zip(hxy[i], xy[j]), color=GREY, lw=0.8, zorder=1)
    for i in pi:
        for j in nb[i]:
            if j in col and i < j:
                dbl = (i, j) in pairs
                ax.plot(*zip(xy[i], xy[j]), color=INK, lw=3.2 if dbl else 1.0, zorder=2, solid_capstyle="round")
    for i in pi:
        ax.scatter(*xy[i], s=70, color=A if col[i] == 0 else B, edgecolor="white", linewidth=0.8, zorder=3)
        if i not in matched:
            ax.scatter(*xy[i], s=260, facecolor="none", edgecolor=RED, linewidth=1.8, zorder=4)
    nA = sum(col[i] == 0 for i in pi); nB = len(pi) - nA; free = len(pi) - 2 * len(pairs)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, loc="left", fontsize=10)
    return nA, nB, free


def spin_text(free):
    return {0: "all electrons paired -> closed-shell singlet",
            1: "1 unpaired electron -> doublet (radical)",
            2: "2 unpaired electrons -> triplet"}.get(free, f"{free} unpaired electrons")


def main(xml, out):
    # Figure 1: idealised molecules
    mols = [("Naphthalene C$_{10}$H$_8$", [(0, 0), (1, 0)]),
            ("Phenalenyl C$_{13}$H$_9$", [(0, 0), (1, 0), (0, 1)]),
            ("Triangulene C$_{22}$H$_{12}$", [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2)])]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, (name, hexes) in zip(axs, mols):
        xy = honeycomb(hexes); nb = graph(xy)
        nA, nB, free = draw(ax, xy, nb, list(range(len(xy))), name)
        ax.text(0.0, -0.08, f"blue {nA}, yellow {nB}: imbalance {abs(nA - nB)}\n"
                f"{'a Kekule structure exists' if free == 0 else f'no Kekule structure: {free} carbon(s) left over'}\n{spin_text(free)}",
                transform=ax.transAxes, fontsize=9, va="top")
    fig.suptitle("Thick lines: double bonds of a Kekule structure (as many as possible). Red rings: carbons that cannot get a "
                 "double bond. Colours: the two sublattices.", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95)); fig.savefig(f"{out}/explain1.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # Figure 2: real PAHdb isomers
    want = {10013: "normal isomer", 10065: "affected isomer"}; geo = {}
    with gzip.open(xml) as f:
        for _, sp in ET.iterparse(f, events=("end",)):
            if sp.tag == NS + "specie" and int(sp.get("uid")) in want:
                at = sp.find(NS + "geometry").findall(NS + "atom")
                el = ["C" if a.find(NS + "type").text == "6" else "H" for a in at]
                xyz = np.array([[float(a.find(NS + t).text) for t in ("x", "y", "z")] for a in at])
                ms = sp.find(NS + "transitions").findall(NS + "mode")
                I = sum(float(m.find(NS + "intensity").text) for m in ms)
                geo[int(sp.get("uid"))] = (el, xyz, sp.find(NS + "symmetry").text, I)
            if sp.tag == NS + "specie":
                sp.clear()
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, uid in zip(axs, (10013, 10065)):
        el, xyz, state, I = geo[uid]
        xyz = xyz - xyz.mean(0); w = np.linalg.eigh(xyz.T @ xyz)[1]; xy2 = xyz @ w[:, [2, 1]]   # molecular plane
        C = [i for i, e in enumerate(el) if e == "C"]; Hs = [i for i, e in enumerate(el) if e == "H"]
        cxy = xy2[C]; nb = graph(cxy)
        hb = [(k, int(np.argmin(np.linalg.norm(cxy - xy2[h], axis=1)))) for k, h in enumerate(Hs)]
        nA, nB, free = draw(ax, cxy, nb, list(range(len(C))), f"uid {uid}, C64H22 ({want[uid]})", hxy=xy2[Hs], hbonds=hb)
        ax.text(0.0, -0.04, f"blue {nA}, yellow {nB}: imbalance {abs(nA - nB)}; "
                f"{'Kekule structure exists' if free == 0 else f'{free} carbons left over'}\n"
                f"expected: {spin_text(free)}\nlisted in PAHdb v4.00: {state} (singlet); total intensity {I:.0f} km/mol",
                transform=ax.transAxes, fontsize=9, va="top", color=RED if free else INK)
    fig.suptitle("Two isomers of C64H22 in PAHdb v4.00, drawn the same way (grey: C-H bonds)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95)); fig.savefig(f"{out}/explain2.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print("explain1.png, explain2.png in", out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
