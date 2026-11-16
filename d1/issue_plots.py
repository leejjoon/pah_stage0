# Figures of the open issues in the D1 temperature rules (docs/plan/d1_hangang_new_species.md), PNG, light surface.
#   python issue_plots.py <out dir>
# 1 spectra_compare.png   one species: MLMD vs PAHdb sticks broadened by the rules, 300 and 900 K, four regions
# 2 width_growth.png      feature width growth 300->900 K, rules vs MLMD, 60 species
# 3 pedestal.png          lower-envelope share by region vs T: MLMD and broadened sticks
# 4 hot_emission.png      share of emitted IR energy above 900 K vs Nc, by photon energy
# 5 breakdown.png         share of systems whose topology changed, by T and planarity
# 6 pahdb_anomaly.png     PAHdb v4.00 harmonic intensities: total vs CH-stretch share, library species
# Figures 1-3 use only species whose PAHdb CH-stretch share is >= 0.15 (18 % of the 917-species library have
# anomalous PAHdb intensities: totals up to 1.5e5 km/mol, almost all in 6-9 um).
import csv, glob, gzip, json, os, sys
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rules_closure as rc  # noqa: E402
from pedestal import stick_spec, floor_share, REGIONS  # noqa: E402
from profile_fit import complex_metrics, COMPLEXES  # noqa: E402

D1 = "/appsdata/t641a02/d1"; SP = f"{D1}/spectra"; REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE = {300: "#86b6ef", 900: "#1c5cab"}; ORANGE = {300: "#f4a988", 900: "#b8461b"}
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "axes.titlesize": 10, "legend.frameon": False, "lines.linewidth": 1.4})


def planar(path):
    L = open(path).read().split("\n"); n = int(L[0]); X = np.array([list(map(float, l.split()[1:4])) for l in L[2:2 + n]])
    X -= X.mean(0); return np.ptp(X @ np.linalg.eigh(X.T @ X)[1][:, 0]) < 0.3


def load_sticks(species):
    uid = {int(s.split("_v4u")[1]): s for s in species}; st = {}
    with gzip.open(f"{D1}/pahdb/pahdb-complete-theoretical-v4.00.xml.gz") as f:
        for ev, el in ET.iterparse(f, events=("end",)):
            if el.tag == rc.NS + "specie" and int(el.get("uid")) in uid:
                ms = el.find(rc.NS + "transitions").findall(rc.NS + "mode")
                st[uid[int(el.get("uid"))]] = (np.array([float(m.find(rc.NS + "frequency").text) for m in ms]),
                                               np.array([float(m.find(rc.NS + "intensity").text) for m in ms]))
            if el.tag == rc.NS + "specie":
                el.clear()
    return st


def main(out):
    os.makedirs(out, exist_ok=True)
    R = {(r["band"], r["quantity"]): r for r in csv.DictReader(open(f"{REPO}/d1/tdep_rules_v1.csv"))}
    W = [(float(r["lo_cm1"]), float(r["hi_cm1"]), float(r["fwhm_300K"]), float(r["fwhm_slope_per_100K"]))
         for r in csv.DictReader(open(f"{REPO}/d1/perline_widths_v1.csv"))]
    lib = [r["species"] for r in csv.DictReader(open(f"{REPO}/d1/new900_library_v1.csv"))]
    st_all = load_sticks(lib)
    normal = [s for s in lib if st_all[s][1][st_all[s][0] > 2800].sum() / st_all[s][1].sum() >= 0.15]
    pick = list(np.random.default_rng(3).choice(normal, 60, replace=False)); st = {s: st_all[s] for s in pick}
    dirs = {300: f"{SP}/new300_v1", 600: f"{SP}/new600_v1", 900: f"{SP}/new900_v1"}

    # 1. one species, Nc nearest 50
    sp = min(st, key=lambda s: abs(int(s[1:s.index("H")]) - 50)); nc = int(sp[1:sp.index("H")]); nu, I = st[sp]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.4), gridspec_kw={"width_ratios": [1.2, 1.6, 1.4, 1.0]})
    for ax, (lo, hi, title) in zip(axs, ((650, 1000, "Out-of-plane (11-14 um)"), (1100, 1700, "In-plane (6-9 um)"),
                                         (1700, 2800, "Gap (overtones, combinations)"), (2850, 3250, "CH stretch (3.3 um)"))):
        for T in (300, 900):
            z = np.load(f"{dirs[T]}/{sp}_{T}K.npz"); y = z["sigma"].mean(0); y = y / y.sum()
            m = (z["freq"] >= lo) & (z["freq"] < hi); ax.plot(z["freq"][m], y[m], color=BLUE[T], label=f"MD {T} K")
            ys = stick_spec(nu, I, T, nc, R, W); ys = ys / ys.sum(); m = (rc.GRID >= lo) & (rc.GRID < hi)
            ax.plot(rc.GRID[m], ys[m], color=ORANGE[T], ls="--", label=f"rules on PAHdb {T} K")
        ax.set_title(title, loc="left"); ax.set_xlabel("wavenumber (cm$^{-1}$)"); ax.set_xlim(hi, lo)
        if lo == 1700:
            ax.set_yscale("log")
    axs[0].set_ylabel("intensity / total (per cm$^{-1}$)"); axs[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"{sp} ({nc} C): our MLMD spectra vs PAHdb harmonic lines broadened and shifted by the rules "
                 "(each normalised to its total; gap panel log scale)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{out}/1_spectra_compare.png", dpi=140); plt.close(fig)

    # 2. width growth per feature, 60 species
    md = {}
    for fn in sorted(glob.glob(f"{SP}/profile_fits_v1_c*.csv")):
        for r in csv.DictReader(open(fn)):
            if r["region"] == "complex" and r["species"] in st:
                md.setdefault((r["species"], r["band"], int(r["T"])), []).append(r)
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, (b, lo, hi) in zip(axs, COMPLEXES):
        q = "fwhm" if b == "3.3" else "rmsw"; xs, ys = [], []
        for s, (nu_, I_) in st.items():
            n_ = int(s[1:s.index("H")])
            w = [complex_metrics(rc.GRID, stick_spec(nu_, I_, T, n_, R, W), lo, hi)[2 if b == "3.3" else 1] for T in (300, 900)]
            m = [np.mean([float(r[q]) for r in md[(s, b, T)]]) for T in (300, 900)]
            xs.append(w[1] - w[0]); ys.append(m[1] - m[0])
        lim = [min(xs + ys) - 1, max(xs + ys) + 1]
        ax.plot(lim, lim, color=INK2, lw=1, ls=":"); ax.scatter(xs, ys, s=22, color=CAT[0], edgecolor=SURF, linewidth=0.8)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.set_title(f"{b} um ({'FWHM' if b == '3.3' else 'rms width'})", loc="left")
        ax.set_xlabel("growth from rules (cm$^{-1}$)")
        ax.text(0.97, 0.05, f"median: rules {np.median(xs):+.1f}\nMD {np.median(ys):+.1f}", transform=ax.transAxes, ha="right",
                color=INK2, fontsize=8)
    axs[0].set_ylabel("growth in MD (cm$^{-1}$)")
    fig.suptitle("Feature width growth 300 -> 900 K, 60 species (dotted: 1:1). 3.3 and 8.6 um close; 6.2 and 7.7 um grow "
                 "faster in MD", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97)); fig.savefig(f"{out}/2_width_growth.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # 3. pedestal by region vs T
    fig, ax = plt.subplots(figsize=(6.4, 3.8)); Ts = (300, 600, 900)
    for k, (name, lo, hi) in enumerate(REGIONS):
        mdv = [np.median([floor_share(np.load(f"{dirs[T]}/{s}_{T}K.npz")["freq"], np.load(f"{dirs[T]}/{s}_{T}K.npz")["sigma"].mean(0), lo, hi)[0]
                          for s in pick]) for T in Ts]
        stv = [np.median([floor_share(rc.GRID, stick_spec(st[s][0], st[s][1], T, int(s[1:s.index("H")]), R, W), lo, hi)[0]
                          for s in st]) for T in Ts]
        ax.plot(Ts, 100 * np.array(mdv), color=CAT[k], marker="o", ms=5, label=f"{name}: MD")
        ax.plot(Ts, 100 * np.array(stv), color=CAT[k], marker="s", ms=4, ls="--", label=f"{name}: rules on PAHdb")
    ax.set_xticks(Ts); ax.set_xlabel("temperature (K)"); ax.set_ylabel("lower envelope, % of total intensity")
    ax.set_title("Background between bands (60 species, median). Only the 1700-2800 cm$^{-1}$ gap\nhas more in MD than the "
                 "broadened harmonic lines produce", loc="left")
    ax.legend(fontsize=7, ncol=2, loc="upper left"); fig.tight_layout(); fig.savefig(f"{out}/3_pedestal.png", dpi=140); plt.close(fig)

    # 4. emission above 900 K
    E = list(csv.DictReader(open(f"{SP}/emission_temp_v1.csv")))
    fig, ax = plt.subplots(figsize=(6.4, 3.8)); nc_ = np.array([int(r["nC"]) for r in E])
    for k, e in enumerate(("8", "10", "13.6")):
        y = np.array([float(r[f"frac_above_900K_{e}eV"]) for r in E])
        ax.scatter(nc_, 100 * y, s=7, color=CAT[k], alpha=0.35, linewidth=0)
        bins = np.arange(20, 230, 10); mid = [(lo + 10 / 2) for lo in bins[:-1]]
        med = [np.median(y[(nc_ >= lo) & (nc_ < lo + 10)]) if ((nc_ >= lo) & (nc_ < lo + 10)).sum() >= 3 else np.nan for lo in bins[:-1]]
        ax.plot(mid, 100 * np.array(med), color=CAT[k], label=f"photon {e} eV")
    ax.axvspan(20, 80, color=GRID, alpha=0.5, lw=0); ax.text(22, 90, "small PAHs: beyond our 900 K data", color=INK2, fontsize=8)
    ax.set_xscale("log"); ax.set_xticks([20, 30, 50, 80, 120, 200]); ax.set_xticklabels(["20", "30", "50", "80", "120", "200"])
    ax.set_xlabel("number of carbon atoms"); ax.set_ylabel("% of emitted IR energy above 900 K"); ax.set_ylim(0, 100)
    ax.set_title("Why 1200 K matters: emission above 900 K (thermal approximation,\n917 library species; dots: species, "
                 "lines: medians)", loc="left")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{out}/4_hot_emission.png", dpi=140); plt.close(fig)

    # 5. breakdown by T and planarity (systems with a topology change)
    def topo(run, xyz):
        c = {True: [0, 0], False: [0, 0]}; pl = {}
        for mf in glob.glob(f"{D1}/runs/{run}/units/*/meta.json"):
            m = json.load(open(mf)); t = m["flags"].get("topo_change_chunk")
            if t is None:
                continue
            for s, v in zip(m["systems"], t):
                p = pl.setdefault(s["species"], planar(f"{D1}/{xyz}/{s['species']}.xyz")); c[p][0] += 1; c[p][1] += v >= 0
        return c

    def screen(run, xyz):
        c = {True: [0, 0], False: [0, 0]}; pl = {}
        for r in csv.DictReader(open(f"{D1}/runs/{run}/screen.csv")):
            p = pl.setdefault(r["species"], planar(f"{D1}/{xyz}/{r['species']}.xyz")); c[p][0] += 1; c[p][1] += r["pass"] != "True"
        return c
    sets = [("300 K, 400 ps\n982 species, end state", screen("new300_v1", "XYZ_new300_v1")),
            ("600 K, 30 ps screen\n1,000 species, end state", screen("screen_new_v1", "XYZ_new_v1")),
            ("600 K, 400 ps\n857 species, every ps", topo("new600_v1", "XYZ_new600_v1")),
            ("900 K, 400 ps pilot\n88 species, every ps", topo("pilot900_v1", "XYZ_pilot900_v1")),
            ("900 K, 400 ps\n779 planar, every ps", topo("new900_v1", "XYZ_new900_v1"))]
    fig, ax = plt.subplots(figsize=(8.4, 3.8)); x = np.arange(len(sets)); bw = 0.36
    for k, (p, lab) in enumerate(((True, "planar"), (False, "non-planar"))):
        v = [100 * c[p][1] / c[p][0] if c[p][0] else np.nan for _, c in sets]
        bars = ax.bar(x + (k - 0.5) * (bw + 0.02), v, bw, color=CAT[k], label=lab)
        for xi, vi, (_, c) in zip(x, v, sets):
            if np.isfinite(vi):
                ax.text(xi + (k - 0.5) * (bw + 0.02), vi + 1.5, f"{vi:.0f}%\n(n={c[p][0]})", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([s for s, _ in sets], fontsize=8); ax.set_ylabel("% of systems with a topology change")
    ax.set_ylim(0, 75); ax.grid(axis="x", visible=False)
    ax.set_title("Where Mai et al.'s model breaks down (new species; H hops or losses, broken bonds)", loc="left")
    ax.legend(fontsize=8, loc="upper left"); fig.tight_layout(); fig.savefig(f"{out}/5_breakdown.png", dpi=140); plt.close(fig)
    # 6. PAHdb anomaly: total intensity vs CH-stretch share, with the MD CH-stretch share for comparison
    lib917 = [r["species"] for r in csv.DictReader(open(f"{REPO}/d1/new300_library_v1.csv"))]
    s917 = load_sticks(lib917)
    tot = np.array([s917[s][1].sum() for s in lib917]); chs = np.array([s917[s][1][s917[s][0] > 2800].sum() / s917[s][1].sum() for s in lib917])
    mdch = []
    for s in lib917:
        d = f"{SP}/new300_v1" if os.path.exists(f"{SP}/new300_v1/{s}_300K.npz") else f"{SP}/new300sp3_v1"
        z = np.load(f"{d}/{s}_300K.npz"); y = z["sigma"].mean(0); mdch.append(y[z["freq"] > 2800].sum() / y.sum())
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    bad = chs < 0.15
    axs[0].scatter(tot[~bad], 100 * chs[~bad], s=9, color=CAT[0], linewidth=0, label=f"CH share >= 15 % ({(~bad).sum()})")
    axs[0].scatter(tot[bad], 100 * chs[bad], s=9, color=CAT[1], linewidth=0, label=f"CH share < 15 % ({bad.sum()})")
    axs[0].set_xscale("log"); axs[0].set_xlabel("PAHdb total harmonic intensity (km/mol)"); axs[0].set_ylabel("PAHdb CH-stretch share (%)")
    axs[0].set_title("PAHdb v4.00 intensities of the 917 library species (all neutral)", loc="left"); axs[0].legend(fontsize=8)
    axs[1].scatter(100 * np.array(mdch)[~bad], 100 * chs[~bad], s=9, color=CAT[0], linewidth=0)
    axs[1].scatter(100 * np.array(mdch)[bad], 100 * chs[bad], s=9, color=CAT[1], linewidth=0)
    axs[1].plot([0, 80], [0, 80], color=INK2, ls=":", lw=1)
    axs[1].set_xlabel("MD CH-stretch share at 300 K (%)"); axs[1].set_ylabel("PAHdb CH-stretch share (%)")
    axs[1].set_title("Same species: MD (Mai's dipole model) vs PAHdb", loc="left")
    fig.tight_layout(); fig.savefig(f"{out}/6_pahdb_anomaly.png", dpi=140); plt.close(fig)
    print("figures in", out)


if __name__ == "__main__":
    main(sys.argv[1])
