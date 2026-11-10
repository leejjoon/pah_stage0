"""Summarise the T0 timing grid (plan 4.4-4.5): median and spread over repeats, the batch-100 decision ratios,
and the log-log plot of seconds per molecule-ps against batch size.

  uv run --no-project --with matplotlib==3.11.2 python t0/summarize.py [reports/t0_timing_full.csv]

Writes reports/t0_timing_summary.csv, reports/t0_timing_summary.md, reports/t0_timing_loglog.{svg,png}.
Rows with another process on GPU 0 at the start or end of the cell (node_foreign_procs != "0/0") are refused:
move them to reports/t0_timing_full_excluded.csv and rerun the cell first.
Sequential ASE is run at batch 1 and 10 only (plan 4.3: it scales linearly); its batch-100 value is the batch-10
per-molecule cost, marked as extrapolated. The decision compares JAX against the fastest PyTorch mode at batch 100.
"""
import csv, os, re, statistics, sys, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "reports", "t0_timing_full.csv")
OUT = os.path.join(ROOT, "reports")
rows = list(csv.DictReader(open(src)))
bad = [r["cell"] for r in rows if r.get("node_foreign_procs", "0/0") != "0/0"]
assert not bad, f"rows with other processes on GPU 0: {bad}; move them out and rerun those cells"

JAX_NAME = {"no": "cueq off", "backend_only": "cueq backend only", "conv_fusion": "cueq conv_fusion"}
def config(r):
    if r["stack"] == "jax": return "JAX " + JAX_NAME[r["cueq"]]
    m = {"sequential": "ASE sequential", "mps_pool": f"ASE MPS pool P={r['mps_workers']}"}.get(
        r["mode"], {"torchsim": "TorchSim batched", "concat": "ASE concatenated"}.get(r["batched_impl"], r["mode"]))
    return f"{m}, cueq {'on' if r['cueq'] == 'yes' else 'off'}"

g = collections.defaultdict(list)
for r in rows: g[(config(r), int(r["batch"]))].append(r)
summary = []
for (c, b), rs in g.items():
    v = sorted(float(r["s_per_mol_ps"]) for r in rs); med = statistics.median(v)
    f = lambda k: statistics.median(float(r[k]) for r in rs) if all(r.get(k) not in (None, "") for r in rs) else None
    summary.append(dict(config=c, stack=rs[0]["stack"], cueq_on=(rs[0]["cueq"] in ("yes", "conv_fusion")), batch=b, n=len(v),
                        median_s_per_mol_ps=med, min=v[0], max=v[-1], spread_pct=100 * (v[-1] - v[0]) / med,
                        ns_per_day_per_traj=f("ns_per_day_per_traj"), gpu_util=f("gpu_util"), gpu_mem_gb=f("gpu_mem_gb"),
                        compile_s=f("compile_s"), cuex=rs[0]["cuex"], extrapolated=False))
# sequential at batch 100: per-molecule cost is batch-independent, take batch 10
for s in [s for s in summary if s["config"].startswith("ASE sequential") and s["batch"] == 10]:
    summary.append({**s, "batch": 100, "n": 0, "extrapolated": True, "gpu_util": None, "gpu_mem_gb": None})
summary.sort(key=lambda s: (s["batch"], s["stack"] != "jax", s["median_s_per_mol_ps"]))

keys = list(summary[0].keys())
with open(os.path.join(OUT, "t0_timing_summary.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
    for s in summary: w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in s.items()})

# ---- decision (plan 4.5): batch 100, JAX against the fastest measured PyTorch mode
b100 = [s for s in summary if s["batch"] == 100]
torch100 = [s for s in b100 if s["stack"] == "torch" and not s["extrapolated"]]
best_t = min(torch100, key=lambda s: s["median_s_per_mol_ps"])
best_t_off = min((s for s in torch100 if not s["cueq_on"]), key=lambda s: s["median_s_per_mol_ps"])
jax100 = {s["config"]: s for s in b100 if s["stack"] == "jax"}
def band(x): return ">5x: JAX for the project" if x > 5 else ("2-5x: JAX for the MD loop, PyTorch kept as reference" if x >= 2 else "<2x: PyTorch stack")
fmt = lambda x: f"{x:.3g}" if x is not None else ""

md = ["# T0 timing grid: summary", "",
      f"Source: `{os.path.relpath(src, ROOT)}`, {len(rows)} rows, {len(g)} configurations, "
      f"{min(r['started'] for r in rows)[:16]} to {max(r['started'] for r in rows)[:16]}. Written by `t0/summarize.py`.", "",
      "Cost is wall-clock seconds per molecule-picosecond (lower is faster): median of the repeats, with the range "
      "(min-max) as spread. Float32, TF32 off in both stacks; GPU 0 and NUMA node 0 of the GH200.", "",
      "## Decision numbers at batch 100 (plan 4.5)", "",
      f"Fastest PyTorch mode: **{best_t['config']}**, {best_t['median_s_per_mol_ps']:.4f} s/mol-ps "
      f"(range {best_t['min']:.4f}-{best_t['max']:.4f}).", "",
      "| JAX configuration | s/mol-ps | speed-up vs fastest PyTorch | band | speed-up vs fastest PyTorch with cueq off |",
      "| --- | --- | --- | --- | --- |"]
for c in ("JAX cueq conv_fusion", "JAX cueq backend only", "JAX cueq off"):
    if c not in jax100: continue
    j = jax100[c]["median_s_per_mol_ps"]; x = best_t["median_s_per_mol_ps"] / j
    md.append(f"| {c} | {j:.4f} | **{x:.2f}x** | {band(x)} | {best_t_off['median_s_per_mol_ps'] / j:.2f}x ({best_t_off['config']}) |")
rng = {c: (best_t["min"] / j["max"], best_t["max"] / j["min"]) for c, j in jax100.items()}
cross = [c for c, (lo, hi) in rng.items() if band(lo) != band(hi)]
md += ["", "Worst and best case from the repeat ranges (slowest JAX against fastest PyTorch repeat, and the reverse): "
       + "; ".join(f"{c} {lo:.2f}x-{hi:.2f}x" for c, (lo, hi) in rng.items()) + ". "
       + (f"**These cross a band boundary: {cross}.**" if cross else "No ratio crosses a band boundary."), "",
       "Which JAX row the decision uses depends on the committee's definition of JAX \"cuEquivariance on\" "
       "(pending, 2026-11-09; plan 4.4). Caveats required by plan 4.5: the GH200's NVLink-C2C link favours the "
       "PyTorch baseline relative to the production cluster; the result depends on which cuEquivariance setting "
       "each stack ran (JAX: `cuex` methods traced per row; PyTorch: MACE with `enable_cueq`).", "",
       "## All configurations", "",
       "| batch | configuration | n | median s/mol-ps | range | spread | ns/day per traj | GPU util % | GPU mem GB | compile s |",
       "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
for s in summary:
    if s["extrapolated"]:
        md.append(f"| {s['batch']} | {s['config']} | (extrapolated from batch 10) | {s['median_s_per_mol_ps']:.4f} | | | | | | |"); continue
    lab = s['config'] + (" (includes equilibration)" if "MPS pool" in s['config'] else "")
    md.append(f"| {s['batch']} | {lab} | {s['n']} | {s['median_s_per_mol_ps']:.4f} | {s['min']:.4f}-{s['max']:.4f} | "
              f"{s['spread_pct']:.1f}% | {fmt(s['ns_per_day_per_traj'])} | {fmt(s['gpu_util'])} | {fmt(s['gpu_mem_gb'])} | {fmt(s['compile_s'])} |")
md += ["", "MPS-pool timings include the 2,000 equilibration steps (about 9% of each run; md_torch.py), unlike every other "
       "mode; the committee (2026-11-10) asked for them to be labelled, not rerun. gpu_util is averaged over the whole "
       "process, model load and equilibration included. GPU memory for the MPS pool is not captured per process (shown as 0). Sequential ASE was not run at "
       "batch 100 (plan 4.3); its row is the batch-10 per-molecule cost."]
open(os.path.join(OUT, "t0_timing_summary.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md[:16]))

# ---- the plot (plan 4.4): two panels, cueq off / on, both stacks, all PyTorch modes, log-log
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not available; plot skipped"); sys.exit(0)
INK, INK2, MUTED, GRID = "#161615", "#52514e", "#8a8984", "#e8e7e3"
COL = {"JAX": "#2a78d6", "TorchSim batched": "#eb6834", "ASE concatenated": "#1baf7a", "ASE MPS pool": "#6250d6", "ASE sequential": "#8a8984"}
def family(c): return next(k for k in COL if c.startswith(k))
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "svg.fonttype": "none", "font.family": "sans-serif"})
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
for ax, on in zip(axes, (False, True)):
    sel = collections.defaultdict(dict)
    for s in summary:
        c = s["config"]
        if c.startswith("JAX"):
            if c != ("JAX cueq conv_fusion" if on else "JAX cueq off"): continue
        elif s["cueq_on"] != on: continue
        key = c.split(",")[0]
        if key.startswith("ASE MPS pool"):          # one MPS line: the fastest P at each batch
            key = "ASE MPS pool (best P)"
            if s["batch"] in sel[key] and sel[key][s["batch"]]["median_s_per_mol_ps"] <= s["median_s_per_mol_ps"]: continue
        sel[key][s["batch"]] = s
    for key, pts in sel.items():
        col = COL[family(key)]; b = sorted(pts); meas = [x for x in b if not pts[x]["extrapolated"]]
        y = [pts[x]["median_s_per_mol_ps"] for x in meas]
        err = [[pts[x]["median_s_per_mol_ps"] - pts[x]["min"] for x in meas], [pts[x]["max"] - pts[x]["median_s_per_mol_ps"] for x in meas]]
        ax.errorbar(meas, y, yerr=err, color=col, marker="o", ms=5, lw=2, capsize=2.5)
        ext = [x for x in b if pts[x]["extrapolated"]]
        if ext: ax.plot([meas[-1], ext[0]], [y[-1], pts[ext[0]]["median_s_per_mol_ps"]], ls=(0, (1, 2.5)), color=col, lw=1.6)
        last = b[-1]; lab = key + (" (extrapolated)" if pts[last]["extrapolated"] else "")
        if key.startswith("ASE MPS"): lab += "".join(f"\nP={pts[x]['config'].split('P=')[1].split(',')[0]} at b={x}" for x in b if x == last)
        ax.annotate(f"{lab}  {pts[last]['median_s_per_mol_ps']:.3g}", (last, pts[last]["median_s_per_mol_ps"]), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=7.5, color=INK2)
    jb = jax100["JAX cueq conv_fusion" if on else "JAX cueq off"]["median_s_per_mol_ps"]
    ax.fill_between([75, 135], jb * 2, jb * 5, color=GRID, zorder=0, lw=0)
    ax.text(68, jb * 3.2, "2x-5x the\nJAX time", ha="right", va="center", fontsize=7, color=MUTED)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([1, 10, 100]); ax.set_xticklabels(["1", "10", "100"])
    ax.set_xlim(0.7, 2500); ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.set_xlabel("batch: molecules simulated together")
    ax.set_title(f"cuEquivariance {'on (JAX: conv_fusion)' if on else 'off'}", loc="left", fontsize=9, color=INK)
axes[0].set_ylabel("seconds per molecule-ps (lower is faster)")
fig.suptitle("T0 timing grid: median of 3 repeats, bars = min-max; float32, no TF32; GH200 GPU 0", x=0.01, ha="left", fontsize=9.5, color=INK)
fig.tight_layout()
for ext in ("svg", "png"): fig.savefig(os.path.join(OUT, f"t0_timing_loglog.{ext}"), dpi=160 if ext == "png" else None)
print("wrote reports/t0_timing_summary.{csv,md}, reports/t0_timing_loglog.{svg,png}")
