#!/usr/bin/env python3
"""Figure 2 of README.md: C30H14 (PAHdb v4.00 uid 10770) as listed in PAHdb, recomputed as a closed-shell singlet, and
recomputed as a triplet.

    python singlet_vs_triplet.py pahdb-complete-theoretical-v4.00.xml.gz triplet.json OUTDIR

triplet.json comes from uks_ir_test.py (test "triplet"): B3LYP/6-31G* (6D, (99,590) grid) with GPU4PySCF at the PAHdb
geometry; RKS singlet with GPU4PySCF's IR routine, UKS triplet with eval_ir_freq_intensity_uks (uks_ir.py). PAHdb
frequencies are plotted as listed (scaled); the computed ones are scaled with PAHdb v4.00's per-mode B3LYP/6-31G*
factors (0.9794 below 1111 cm^-1, 0.9691 for 1111-1800, 0.9597 above; from the XML's scale attributes) so the three
curves share an axis. Needs numpy and matplotlib.
"""
import gzip, json, sys
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NS = "{http://www.astrochemistry.org/pahdb/theoretical}"


def pahdb_sticks(xml, uid):
    with gzip.open(xml) as f:
        for _, sp in ET.iterparse(f, events=("end",)):
            if sp.tag == NS + "specie" and int(sp.get("uid")) == uid:
                ms = sp.find(NS + "transitions").findall(NS + "mode")
                return (np.array([float(m.find(NS + "frequency").text) for m in ms]),
                        np.array([float(m.find(NS + "intensity").text) for m in ms]), sp.find(NS + "symmetry").text)
            if sp.tag == NS + "specie":
                sp.clear()


def scale(nu):
    return nu * np.where(nu < 1111.05, 0.9794, np.where(nu < 1800.05, 0.9691, 0.9597))


def main(xml, js, out):
    d = json.load(open(js))
    nu0, I0, state = pahdb_sticks(xml, 10770)
    rows = [("PAHdb v4.00, listed as " + state + " (singlet)", nu0, I0, "#52514e"),
            ("recomputed: closed-shell singlet", scale(np.array(d["spectra"]["singlet"]["freq"])), np.array(d["spectra"]["singlet"]["I"]), "#e34948"),
            (f"recomputed: triplet (18.7 kcal/mol lower; <S²> {d['uks_triplet']['S2']:.2f})", scale(np.array(d["spectra"]["triplet"]["freq"])),
             np.array(d["spectra"]["triplet"]["I"]), "#2a78d6")]
    x = np.arange(400, 3300, 1.0)
    fig, axs = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    for ax, (lab, nu, I, col) in zip(axs, rows):
        m = nu > 0
        y = sum(i * 5 ** 2 / ((x - n) ** 2 + 5 ** 2) / (np.pi * 5) for n, i in zip(nu[m], I[m]))
        ax.plot(x, y, color=col, lw=1.2)
        ch = I[m][nu[m] > 2700].sum() / I[m].sum()
        ax.set_title(f"{lab}: total {I[m].sum():,.0f} km/mol, CH stretch {100 * ch:.0f} %", loc="left", fontsize=9)
        ax.set_ylabel("km/mol per cm$^{-1}$", fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[-1].set_xlabel("wavenumber (cm$^{-1}$, scaled as PAHdb; lines broadened to 10 cm$^{-1}$)")
    fig.suptitle("C30H14 (PAHdb uid 10770), three versions of its spectrum (note the different y scales)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{out}/fig_singlet_triplet.png", dpi=140); plt.close(fig)
    print("fig_singlet_triplet.png in", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
