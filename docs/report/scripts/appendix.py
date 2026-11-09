"""Write generated/appendix.md: every distribution installed in each image, with its version, its one-line summary
(from its own metadata) and which installed packages require it. Source: data/deps.json (scripts/deps.py)."""
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))
REP = os.path.join(HERE, "..")
D = json.load(open(os.path.join(REP, "data", "deps.json")))
TITLES = {"A_jax": "Container A (`A_jax.sif`)", "B_torchsim": "Container B + TorchSim (`B_torchsim.sif`)",
          "C_cpu": "Container C (`C_cpu.sif`)"}
esc = lambda s: s.replace("|", "\\|").replace("*", "\\*").replace("_", "\\_")

out = []
for img, pk in D.items():
    out.append(f"## {TITLES[img]}\n")
    if img == "B_torchsim":
        out.append("**B** marks the 36 distributions our recipes added (`B_torch.def`, `B_torchsim.def`); the others "
                   "came with the NGC 26.10 image. `B_torch.sif` is the same list without torch-sim-atomistic, "
                   "nvalchemi-toolkit-ops, warp-lang, tables and their own dependencies.\n")
    out.append(f"{len(pk)} distributions. **R** marks those the recipe names.\n")
    out.append("| Package | Version | | Summary (from the package) | Required by |\n| --- | --- | --- | --- | --- |")
    for n in sorted(pk):
        p = pk[n]
        flag = "R" if p.get("requested") else ("B" if p.get("origin") == "added" else "")
        req = ", ".join(p["required_by"][:6]) + (" …" if len(p["required_by"]) > 6 else "")
        out.append(f"| {esc(p['name'])} | {esc(p['version'][:24])} | {flag} | {esc(p['summary'][:110])} | {esc(req)} |")
    out.append("")
os.makedirs(os.path.join(REP, "generated"), exist_ok=True)
open(os.path.join(REP, "generated", "appendix.md"), "w").write("\n".join(out) + "\n")
print("wrote generated/appendix.md")
