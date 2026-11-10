"""Data figures for the report: static SVGs (used by every output format) and the JSON the HTML widgets read.
Inputs: data/pahdb.json (extract_pahdb.py), data/deps.json (deps.py), the MD test trajectories in $MD_DIR
(default /var/tmp/jjlee-pah/md; optional), reports/t0_timing_{calib,full}.csv.
Run from anywhere: python docs/report/scripts/make_figures.py"""
import os, csv, json, collections, statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REP = os.path.join(ROOT, "docs", "report")
FIG, DATA, WEB = (os.path.join(REP, d) for d in ("figures", "data", "web"))
os.makedirs(FIG, exist_ok=True)
MD = os.environ.get("MD_DIR", "/var/tmp/jjlee-pah/md")

# One palette for the whole report (light mode; static figures go to print). Categorical slots in fixed order.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = \
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
NEUTRAL_C, CATION_C = BLUE, ORANGE
HCLASS = {1: ("solo", BLUE, "o"), 2: ("duo", ORANGE, "s"), 3: ("trio", AQUA, "^"), 4: ("quartet", MAGENTA, "D")}
BANDS_UM = [3.3, 6.2, 7.7, 8.6, 11.2, 12.7]

plt.rcParams.update({
    "svg.fonttype": "none", "font.family": "sans-serif", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "legend.frameon": False,
    "lines.linewidth": 1.6, "figure.dpi": 100})

def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    print("wrote", name)

P = json.load(open(os.path.join(DATA, "pahdb.json")))

# ---------------------------------------------------------------- molecules: bonds and H edge classes
def molecule_graph(m):
    Z, X = np.array(m["Z"]), np.array(m["xy"])
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    bonds = [(int(i), int(j)) for i, j in zip(*np.where(np.triu((D > 0.5) & (D < 1.65), 1)))
             if not (Z[i] == 1 and Z[j] == 1)]
    C, H = np.where(Z == 6)[0], np.where(Z == 1)[0]
    host = {int(h): int(C[D[h, C].argmin()]) for h in H}            # carbon carrying each H
    ch = sorted(set(host.values()))
    parent = {c: c for c in ch}
    def find(c):
        while parent[c] != c: parent[c] = parent[parent[c]]; c = parent[c]
        return c
    for a in ch:
        for b in ch:
            if a < b and D[a, b] < 1.65: parent[find(a)] = find(b)
    size = collections.Counter(find(c) for c in ch)
    hcls = {h: size[find(c)] for h, c in host.items()}
    counts = collections.Counter(hcls.values())
    for n, (name, _, _) in HCLASS.items():                            # cross-check against species.csv
        assert counts.get(n, 0) == int(m[f"n_{name}"]), (m["formula"], name, counts)
    return bonds, hcls

mols = []
for m in P["molecules"]:
    bonds, hcls = molecule_graph(m)
    mols.append({k: m[k] for k in ("uid", "formula", "n_c", "n_h", "class", "n_solo", "n_duo", "n_trio", "n_quartet",
                                   "planarity_A")} | {"Z": m["Z"], "xy": m["xy"], "bonds": bonds,
                                                       "hclass": {str(h): c for h, c in hcls.items()}})

def draw_molecule(ax, m, scale_ref=None, legend=False):
    X = np.array(m["xy"]); Z = np.array(m["Z"])
    for i, j in m["bonds"]:
        ax.plot(*X[[i, j]].T, color="#3a3a38" if Z[i] == Z[j] == 6 else "#a9a8a3", lw=1.3 if Z[i] == Z[j] == 6 else 1.0,
                solid_capstyle="round", zorder=1)
    for h, c in m["hclass"].items():
        name, col, mk = HCLASS[min(c, 4)]
        ax.scatter(*X[int(h)], s=34, c=col, marker=mk, edgecolors="white", linewidths=0.8, zorder=3)
    ax.set_aspect("equal"); ax.axis("off")
    if scale_ref is not None:
        cx, cy = X.mean(0); ax.set_xlim(cx - scale_ref, cx + scale_ref); ax.set_ylim(cy - scale_ref * 0.8, cy + scale_ref * 0.8)
    hc = ", ".join(f"{m['n_' + nm]} {nm}" for nm in ("solo", "duo", "trio", "quartet") if int(m["n_" + nm]))
    ax.set_title(f"{m['formula']}  ({m['class']})\n{hc}", fontsize=8.5, color=INK2)

by_formula = {m["formula"]: m for m in mols}
pick = ["C24H12", "C26H16", "C54H18", "C66H24"]
fig, axs = plt.subplots(1, 4, figsize=(9.6, 2.9))
for ax, f in zip(axs, pick): draw_molecule(ax, by_formula[f], scale_ref=8.2)
handles = [plt.Line2D([], [], ls="", marker=mk, color=col, markeredgecolor="white", markersize=7, label=f"{nm} H")
           for _, (nm, col, mk) in HCLASS.items()]
handles += [plt.Line2D([], [], color="#3a3a38", lw=1.3, label="C–C bond")]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, -0.04))
save(fig, "molecules.svg")

# ---------------------------------------------------------------- spectra: sticks vs 6 eV cascade emission
grid = np.array(P["grid_cm"])
um = 1e4 / grid
def stick_um(u):
    s = np.array(P["sticks"][str(u)]); return 1e4 / s[:, 0], s[:, 1]

def band_marks(ax, ymax, labels=True):
    for b in BANDS_UM:
        ax.axvline(b, color=GRID, lw=1, zorder=0)
        if labels: ax.text(b, ymax, f"{b:g}", ha="center", va="bottom", fontsize=7.5, color=MUTED)

fig, axs = plt.subplots(2, 1, figsize=(7.6, 4.6), sharex=True, gridspec_kw={"hspace": 0.25})
n_u, c_u = P["pairs"]["coronene"]
for u, col, lab in ((n_u, NEUTRAL_C, "neutral C$_{24}$H$_{12}$"), (c_u, CATION_C, "cation C$_{24}$H$_{12}^+$")):
    x, y = stick_um(u)
    m = (x > 2.8) & (x < 20)
    axs[0].vlines(x[m], 0, y[m], color=col, lw=1.6, label=lab, alpha=0.9)
    e = np.array(P["emission"][str(u)])
    axs[1].plot(um, e, color=col, label=lab)
band_marks(axs[0], 0, labels=False)
band_marks(axs[1], 1.02)
axs[0].set_ylabel("intensity (km/mol)"); axs[0].set_title("harmonic DFT bands of a cold molecule (PAHdb v4.00)", fontsize=9, loc="left", color=INK, pad=8)
axs[1].set_ylabel("emission (peak = 1)"); axs[1].set_title("after absorbing a 6 eV photon: cascade emission, 15 cm$^{-1}$ Gaussian (grey lines: the classic band positions, µm)", fontsize=9, loc="left", color=INK, pad=14)
axs[1].set_xlabel("wavelength (µm)"); axs[1].set_xlim(2.8, 20); axs[1].set_ylim(0, 1.15)
axs[0].legend(loc="upper right");
save(fig, "spectrum_coronene.svg")

# ---------------------------------------------------------------- MD: VDOS of coronene vs harmonic sticks
def vdos(v, dt_fs=0.5):
    v = v - v.mean(0); f = np.fft.rfft(v, axis=0, n=2 * len(v)); p = (np.abs(f) ** 2).sum((1, 2))
    nu = np.fft.rfftfreq(2 * len(v), d=dt_fs * 1e-15) / 2.99792458e10
    k = np.hanning(15); k /= k.sum(); return nu, np.convolve(p, k, "same")

md_series = {}
for name, f, key in (("PyTorch (ASE)", "torch_seq_b1.npz", "vel_0"), ("JAX (jax-md)", "jax_nvt_b20.npz", "vel")):
    path = os.path.join(MD, f)
    if os.path.exists(path):
        d = np.load(path); v = d[key].astype(np.float64)
        if key == "vel": v = v[:, :36]                          # trajectory 0 = coronene, 36 atoms
        nu, p = vdos(v); m = (nu > 300) & (nu < 3500)
        md_series[name] = (nu[m], p[m] / p[m].max())
if md_series:
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    s = np.array(P["sticks"][str(n_u)])
    ax.vlines(s[:, 0], 0, s[:, 1] / s[:, 1].max(), color=MUTED, lw=1.2, label="harmonic DFT bands (PAHdb, IR intensity)")
    for (name, (nu, p)), col in zip(md_series.items(), (BLUE, ORANGE)):
        ax.plot(nu, p, color=col, label=f"MD at 600 K, {name}: VDOS")
    ax.set_xlim(300, 3400); ax.set_ylim(0, 1.12); ax.set_xlabel("wavenumber (cm$^{-1}$)"); ax.set_ylabel("normalised")
    sec = ax.secondary_xaxis("top", functions=(lambda x: 1e4 / np.maximum(x, 1), lambda x: 1e4 / np.maximum(x, 1)))
    sec.set_xticks([3.3, 5, 6.2, 8, 11.2, 20]); sec.set_xlabel("wavelength (µm)", color=INK2)
    ax.legend(loc="upper center", fontsize=8, ncol=3, bbox_to_anchor=(0.5, -0.2))
    save(fig, "md_vdos.svg")
else:
    print("MD trajectories not found; md_vdos.svg not regenerated")

# ---------------------------------------------------------------- timing grid
def rows(name):
    path = os.path.join(ROOT, "reports", f"t0_timing_{name}.csv")
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []
# A full-grid row is excluded when another user's process was on GPU 0 at the start or end of the cell
# (node_foreign_procs "start/end" not "0/0"). Filtering on the data, not the cell name, keeps a rerun of the cell.
def shared(r): return r.get("phase") == "full" and r.get("node_foreign_procs", "0/0") != "0/0"
BAD = {}

def series_key(r):
    if r["stack"] == "jax": return "JAX " + {"no": "cueq off", "backend_only": "cueq backend only", "conv_fusion": "cueq conv_fusion"}[r["cueq"]]
    mode = {"sequential": "ASE sequential", "mps_pool": f"ASE MPS pool P={r['mps_workers']}"}.get(
        r["mode"], {"torchsim": "TorchSim batched", "concat": "ASE concatenated"}.get(r["batched_impl"], r["mode"]))
    return f"{mode}, cueq {'on' if r['cueq'] == 'yes' else 'off'}"

def table(rs):
    agg = collections.defaultdict(list)
    for r in rs:
        if shared(r):
            BAD[f'{r["cell"]} ({r["started"][:16]})'] = f'other processes on GPU 0 at start/end: {r["node_foreign_procs"]}'
            continue
        agg[(series_key(r), int(r["batch"]))].append(float(r["s_per_mol_ps"]))
    return {k: {"median": statistics.median(v), "min": min(v), "max": max(v), "n": len(v)} for k, v in agg.items()}

timing = {}
for r in rows("full_excluded"):      # rows moved out of the full CSV after their cell was rerun
    BAD[f'{r["cell"]} ({r["started"][:16]})'] = f'other processes on GPU 0 at start/end: {r["node_foreign_procs"]}; cell rerun'
for ph in ("calib", "full"):
    t = table(rows(ph))
    if t: timing[ph] = [{"series": s, "batch": b, **v} for (s, b), v in sorted(t.items())]
json.dump({"bad": BAD, **timing}, open(os.path.join(DATA, "timing.json"), "w"), indent=0)

# Static plot from the full grid (median of 3 repeats, bars = min-max); falls back to the calibration pass.
show = [("JAX cueq conv_fusion", BLUE, "-"), ("JAX cueq off", BLUE, ":"),
        ("TorchSim batched, cueq on", ORANGE, "-"), ("TorchSim batched, cueq off", ORANGE, ":"),
        ("ASE concatenated, cueq on", AQUA, "-"), ("ASE concatenated, cueq off", AQUA, ":"),
        ("ASE MPS pool (P=4 at 10, P=32 at 100), cueq off", VIOLET, ":"),
        ("ASE sequential, cueq off", MUTED, ":"), ("ASE sequential, cueq on", MUTED, "-")]
LABEL_DY = {"ASE sequential, cueq on": 4, "ASE sequential, cueq off": -4, "TorchSim batched, cueq off": -5,
            "ASE concatenated, cueq off": 5}
PH = "full" if len(timing.get("full", [])) >= 31 else "calib"
cal, rng = {}, {}
for d in timing.get(PH, []):
    cal.setdefault(d["series"], {})[d["batch"]] = d["median"]; rng.setdefault(d["series"], {})[d["batch"]] = (d["min"], d["max"])
MPS = "ASE MPS pool (P=4 at 10, P=32 at 100), cueq off"     # the pool sizes the grid ran: one line through both
for P_, bb in (("4", 10), ("32", 100)):
    k = f"ASE MPS pool P={P_}, cueq off"
    if bb in cal.get(k, {}): cal.setdefault(MPS, {})[bb] = cal[k][bb]; rng.setdefault(MPS, {})[bb] = rng[k][bb]
fig, ax = plt.subplots(figsize=(7.6, 4.4))
for s, col, ls in show:
    if s not in cal: continue
    b = sorted(cal[s]); y = [cal[s][x] for x in b]
    ax.plot(b, y, ls, color=col, marker="o", ms=4, lw=1.6)
    ax.errorbar(b, y, yerr=[[y[i] - rng[s][x][0] for i, x in enumerate(b)], [rng[s][x][1] - y[i] for i, x in enumerate(b)]],
                fmt="none", ecolor=col, elinewidth=1, capsize=2)
    if b[-1] == 100: ax.annotate(s, (b[-1], y[-1]), xytext=(6, LABEL_DY.get(s, 0)), textcoords="offset points", va="center", fontsize=7.5, color=INK2)
    else: ax.annotate(s, (b[0], y[0]), xytext=(-6, LABEL_DY.get(s, 0)), textcoords="offset points", va="center", ha="right", fontsize=7.5, color=INK2)
seq = cal.get("ASE sequential, cueq off", {})
if 10 in seq:
    ax.plot([10, 100], [seq[10], seq[10]], ls=(0, (1, 3)), color=MUTED, lw=1.2)
    ax.annotate("ASE sequential (extrapolated)", (100, seq[10]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7.5, color=MUTED)
best_jax = cal.get("JAX cueq conv_fusion", {}).get(100)
if best_jax:
    ax.fill_between([80, 125], best_jax * 2, best_jax * 5, color=GRID, zorder=0)
    ax.text(70, best_jax * 3.2, "2×–5×\nJAX time", ha="right", va="center", fontsize=7.5, color=MUTED)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([1, 10, 100]); ax.set_xticklabels(["1", "10", "100"])
ax.set_xlim(0.12, 900); ax.set_xlabel("batch: molecules simulated together"); ax.set_ylabel("seconds per molecule-picosecond (lower is faster)")
ax.set_title("T0 timing grid (median of 3 repeats, 20,000 production steps): cost per molecule-ps" if PH == "full" else
             "T0 calibration pass (one repeat, 400 production steps): cost per molecule-ps", loc="left", fontsize=9, color=INK)
save(fig, "timing.svg")

# ---------------------------------------------------------------- widget data
spec = {"pairs": P["pairs"], "info": P["info"], "sticks": P["sticks"],
        "grid_cm": grid[::2].round(1).tolist(), "emission": {u: e[::2] for u, e in P["emission"].items()}}
widget = {"molecules": mols, "spectra": spec, "timing": json.load(open(os.path.join(DATA, "timing.json"))),
          "md": {k: {"nu": v[0][::2].round(1).tolist(), "p": v[1][::2].round(4).tolist()} for k, v in md_series.items()},
          "deps": json.load(open(os.path.join(DATA, "deps.json")))}
os.makedirs(WEB, exist_ok=True)
with open(os.path.join(WEB, "report-data.js"), "w") as f:
    f.write("window.REPORT_DATA = " + json.dumps(widget, separators=(",", ":")) + ";\n")
print("wrote web/report-data.js", os.path.getsize(os.path.join(WEB, "report-data.js")) // 1024, "KiB")
