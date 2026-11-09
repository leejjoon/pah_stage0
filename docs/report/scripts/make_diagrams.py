"""Explanatory diagrams for the report, written as plain SVG (no plotting library), so they stay small, sharp and
editable, and work in HTML, EPUB and PDF alike. Each function writes one file into figures/.
Conventions: container A = blue, B = orange, C = aqua (the same in every figure); our own code = dark ink.
Run: python docs/report/scripts/make_diagrams.py"""
import os, json, html

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
DATA = os.path.join(HERE, "..", "data")

INK, INK2, MUTED, LINE, PAPER = "#0b0b0b", "#52514e", "#8a8984", "#c9c8c2", "#ffffff"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"
TINT = {BLUE: "#e6f0fc", ORANGE: "#fdebe3", AQUA: "#dff4ec", YELLOW: "#fdf1d6", MAGENTA: "#fbe9f0", VIOLET: "#ebe8fa",
        MUTED: "#f1f0ed", INK: "#f1f0ed"}
CA, CB, CC = BLUE, ORANGE, AQUA


class SVG:
    def __init__(self, w, h, title):
        self.w, self.h, self.items = w, h, []
        self.title = title

    def add(self, s): self.items.append(s)

    def text(self, x, y, s, size=12, color=INK2, anchor="middle", weight="normal", italic=False, cls=None):
        lines = s.split("\n") if isinstance(s, str) else s
        style = ' font-style="italic"' if italic else ""
        c = f' class="{cls}"' if cls else ""
        for i, ln in enumerate(lines):
            self.add(f'<text x="{x}" y="{y + i * size * 1.3:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
                     f'font-weight="{weight}"{style}{c}>{html.escape(ln)}</text>')

    def box(self, x, y, w, h, title="", body="", color=MUTED, fill=None, rx=8, dashed=False, tsize=12.5, bsize=11,
            title_color=INK, data=None, stroke_w=1.4):
        fill = fill or TINT.get(color, "#f6f6f4")
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        d = f' data-key="{html.escape(data)}"' if data else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{color}" '
                 f'stroke-width="{stroke_w}"{dash}{d}/>')
        lines_t = title.split("\n") if title else []
        lines_b = body.split("\n") if body else []
        total = len(lines_t) * tsize * 1.3 + len(lines_b) * bsize * 1.3 + (4 if lines_t and lines_b else 0)
        ty = y + h / 2 - total / 2 + tsize * 0.95
        if lines_t:
            self.text(x + w / 2, ty, lines_t, tsize, title_color, weight="600")
            ty += len(lines_t) * tsize * 1.3 + 4
        if lines_b:
            self.text(x + w / 2, ty - tsize * 0.95 + bsize * 0.95, lines_b, bsize, INK2)

    def arrow(self, x1, y1, x2, y2, color=MUTED, width=1.6, curve=0, dashed=False, head=True, label=None, lsize=10.5,
              lpos=0.5, ldx=0, ldy=-6):
        mid = f"{(x1 + x2) / 2 + curve * (y2 - y1) / 100:.1f},{(y1 + y2) / 2 - curve * (x2 - x1) / 100:.1f}"
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        mk = f' marker-end="url(#ah-{color[1:]})"' if head else ""
        self.add(f'<path d="M{x1},{y1} Q{mid} {x2},{y2}" fill="none" stroke="{color}" stroke-width="{width}"{dash}{mk}/>')
        self._markers.add(color)
        if label:
            self.text((x1 + x2) / 2 + ldx, (y1 + y2) / 2 + ldy, label, lsize, INK2)

    _markers = set()

    def save(self, name):
        defs = "".join(f'<marker id="ah-{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                       f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for c in sorted(self._markers))
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" '
               f'font-family="Inter, \'Helvetica Neue\', Arial, sans-serif" role="img">'
               f'<title>{html.escape(self.title)}</title><defs>{defs}</defs>'
               f'<rect width="{self.w}" height="{self.h}" fill="{PAPER}"/>' + "\n".join(self.items) + "</svg>\n")
        open(os.path.join(FIG, name), "w").write(svg)
        SVG._markers = set()
        print("wrote", name)


# ------------------------------------------------------------------------------------------------ 1. the big picture
def big_picture():
    s = SVG(980, 360, "From a UV photon to a band-ratio formula")
    xs = [20, 180, 340, 500, 660, 820]
    items = [("UV photon", "from a hot star\n(a few to 13.6 eV)", YELLOW),
             ("PAH absorbs it", "flat C/H molecule,\nheats to several\nhundred K", MUTED),
             ("IR emission", "cools by emitting at its\nvibration frequencies:\n3.3, 6.2, 7.7, 8.6,\n11.2, 12.7 µm bands", MAGENTA),
             ("JWST MIRI\nspectrum", "bands from millions of\nmolecules, + dust\ncontinuum, gas lines,\nnoise", VIOLET),
             ("PAHFIT", "decomposes the\nspectrum; returns\nband strengths", CC),
             ("Formula", "f ≈ c₁ + c₂ / (c₃ − r)\nr = 6.2/11.2 ratio\nf = ionised fraction", INK)]
    for x, (t, b, c) in zip(xs, items):
        s.box(x, 110, 140, 130, t, b, color=c)
    for a, b_ in zip(xs[:-1], xs[1:]):
        s.arrow(a + 142, 175, b_ - 2, 175, color=INK2)
    # brackets
    s.add(f'<path d="M{xs[0]},92 L{xs[0]},80 L{xs[3] + 140},80 L{xs[3] + 140},92" fill="none" stroke="{BLUE}" stroke-width="1.6"/>')
    s.text((xs[0] + xs[3] + 140) / 2, 50, "Nature (and our forward model): we can compute every step for a known population", 13, BLUE, weight="600")
    s.text((xs[0] + xs[3] + 140) / 2, 68, "PAHdb DFT spectra → cascade emission model → population mix → instrument model + noise", 11, INK2)
    s.add(f'<path d="M{xs[3]},258 L{xs[3]},270 L{xs[5] + 140},270 L{xs[5] + 140},258" fill="none" stroke="{ORANGE}" stroke-width="1.6"/>')
    s.text((xs[3] + xs[5] + 140) / 2, 292, "The inverse problem the project solves:", 13, ORANGE, weight="600")
    s.text((xs[3] + xs[5] + 140) / 2, 310, "from measured band strengths back to population properties (charge, size),", 11, INK2)
    s.text((xs[3] + xs[5] + 140) / 2, 325, "as a short readable formula found by symbolic regression (PySR)", 11, INK2)
    s.text(20, 300, "Each box is one stage in the\nT0b slice (Section 5).", 11, MUTED, anchor="start", italic=True)
    s.save("big_picture.svg")


# ------------------------------------------------------------------------------------------------ 2. roadmap
def roadmap():
    s = SVG(980, 330, "Where T0 and T0b sit in the project")
    s.text(20, 28, "now: first two months (this plan)", 12.5, INK, anchor="start", weight="600")
    s.text(640, 28, "later (parent proposal)", 12.5, MUTED, anchor="start", weight="600")
    s.add(f'<line x1="615" y1="15" x2="615" y2="320" stroke="{LINE}" stroke-dasharray="4 4"/>')
    s.box(20, 50, 250, 110, "T0: stack benchmark", "Is JAX (jax-md + mace-jax) > 5× faster\nthan PyTorch (ASE + MACE) per\nmolecule at batch 100?  GPU",
          color=CA)
    s.box(330, 50, 240, 110, "T0 memo", "picks the MD software stack\n(JAX / mixed / PyTorch)\nwith caveats", color=MUTED)
    s.box(20, 190, 250, 120, "T0b: vertical slice", "every pipeline stage once, on a tiny\ninput: 50 spectra → 2,000 mock\npopulations → PAHFIT → PySR\n→ Orion Bar.  CPU", color=CC)
    s.box(330, 190, 240, 120, "Slice report", "plumbing works end to end;\nclosure test: the search\nrediscovers a known tracer;\ngo / no-go for scale-up", color=MUTED)
    s.arrow(272, 105, 328, 105, INK2); s.arrow(272, 250, 328, 250, INK2)
    s.box(640, 50, 320, 110, "ML molecular dynamics at scale", "hot, anharmonic IR spectra from MD\n(MACE forces + a dipole model) for\nmany PAHs, on the chosen stack",
          color=CA, dashed=True)
    s.box(640, 190, 320, 120, "Medium-scale run, then the full study", "e.g. 500 species, 10,000 populations\n(plan §8); formulas for charge, size …\napplied to JWST spectra", color=CC, dashed=True)
    s.arrow(572, 105, 638, 105, INK2); s.arrow(572, 250, 638, 250, INK2)
    s.arrow(800, 162, 800, 188, MUTED, dashed=True, label="better spectra feed the library", ldx=90, ldy=4)
    s.save("roadmap.svg")


# ------------------------------------------------------------------------------------------------ 3. the T0b pipeline
def pipeline():
    s = SVG(980, 470, "The T0b vertical slice, stage by stage")
    stages = [("1  Species", "25 PAHs × (neutral,\ncation) = 50\nspecies.csv", "§5.1"),
              ("2  Spectra", "DFT band lists\n(cm⁻¹, km/mol)\nPAHdb v4.00", "§5.2"),
              ("3  Cascade", "6 eV photon →\nemission spectrum,\n15 cm⁻¹ Gaussian", "§5.3"),
              ("4  Populations", "2,000 mixes:\nsize (log-normal) ×\nionised fraction f", "§5.4"),
              ("5  Instrument", "MIRI MRS resolution,\n+ dust continuum,\n+ noise (S/N 50)", "§5.5"),
              ("6  PAHFIT", "fit all 2,000;\nband strengths,\nMonte Carlo errors", "§5.6"),
              ("7  Features", "ratios /11.2 or /total;\nrandom-forest\nreference, baselines", "§5.7"),
              ("8  PySR", "search formulas\nf = g(ratios);\nPareto front", "§5.8–5.9"),
              ("9  Orion Bar", "apply to 5 JWST\ntemplate spectra\n(PDRs4All)", "§5.10")]
    pkg = ["AmesPAHdbPythonSuite", "AmesPAHdbPythonSuite", "AmesPAHdbPythonSuite", "numpy / scipy",
           "PAHFIT jwst.yaml, astropy", "PAHFIT (classic.yaml)", "pandas, scikit-learn", "PySR → Julia", "PAHFIT + formula"]
    W, H, gx = 196, 104, 20
    pos = []
    for i, ((t, b, sec), p) in enumerate(zip(stages, pkg)):
        row = i // 5
        col = i if row == 0 else 4 - (i - 5)
        x = 12 + col * W
        y = 60 + row * 190
        pos.append((x, y))
        s.box(x, y, W - 20, H, t, b, color=CC, data=f"stage{i + 1}")
        s.text(x + (W - 20) / 2, y + H + 16, p, 10.5, MUTED, italic=True)
        s.text(x + W - 26, y + 16, sec, 10, MUTED, anchor="end")
    for i in range(8):
        (x1, y1), (x2, y2) = pos[i], pos[i + 1]
        if i < 4: s.arrow(x1 + W - 18, y1 + H / 2, x2 - 2, y2 + H / 2, INK2)
        elif i == 4: s.arrow(x1 + (W - 20) / 2, y1 + H + 24, x2 + (W - 20) / 2, y2 - 4, INK2)
        else: s.arrow(x1 - 2, y1 + H / 2, x2 + W - 18, y2 + H / 2, INK2)
    s.text(20, 30, "Container C (CPU only).  Every stage reads and writes one HDF5 file, slice.h5; `make slice` reruns the chain.", 12.5, INK, anchor="start", weight="600")
    s.box(12, 400, 300, 56, "slice.h5", "the single data file, with provenance:\nPAHdb version, commits, seeds, image hash", color=INK, tsize=12, bsize=10.5)
    s.text(340, 420, "Stages 1–4 are our forward model (synthetic data with known answers).\nStages 6–8 are the analysis we want to trust.  Stage 9 is the first contact with real data.",
           11, INK2, anchor="start")
    s.save("pipeline.svg")


# ------------------------------------------------------------------------------------------------ 4. what a MACE potential does
def mlip():
    s = SVG(980, 320, "What a machine-learned force field computes")
    # atoms with a cutoff circle
    import math
    cx, cy = 120, 150
    pts = [(cx + 34 * math.cos(a), cy + 34 * math.sin(a)) for a in [i * math.pi / 3 for i in range(6)]]
    pts += [(cx + 64 * math.cos(a), cy + 64 * math.sin(a)) for a in [i * math.pi / 3 for i in range(6)]]
    s.add(f'<circle cx="{pts[0][0]}" cy="{pts[0][1]}" r="62" fill="{TINT[BLUE]}" stroke="{BLUE}" stroke-dasharray="4 3"/>')
    for i in range(6):
        a, b = pts[i], pts[(i + 1) % 6]
        s.add(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" stroke="#3a3a38" stroke-width="2"/>')
        h = pts[6 + i]
        s.add(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{h[0]:.1f}" y2="{h[1]:.1f}" stroke="#a9a8a3" stroke-width="1.5"/>')
    for i, (x, y) in enumerate(pts):
        r, f = (7, "#3a3a38") if i < 6 else (5, "#ffffff")
        s.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{f}" stroke="#3a3a38" stroke-width="1.2"/>')
    s.text(120, 250, "atoms within the cutoff\n(dashed circle, ~5 Å) are neighbours", 10.5, INK2)
    s.text(120, 40, "positions xᵢ, elements Zᵢ", 12, INK, weight="600")
    s.arrow(205, 150, 255, 150, INK2)
    s.box(260, 85, 200, 130, "MACE network", "graph of neighbours →\nequivariant messages\n(\"rotate the molecule,\nthe outputs rotate too\")", color=CA)
    s.arrow(462, 150, 512, 150, INK2)
    s.box(515, 85, 170, 130, "Energy E", "sum of per-atom\nenergies:\nE = Σᵢ εᵢ", color=MUTED)
    s.arrow(687, 150, 737, 150, INK2)
    s.text(712, 240, "automatic\ndifferentiation", 10.5, INK2)
    s.arrow(712, 228, 712, 156, MUTED, head=False, dashed=True)
    s.box(740, 85, 220, 130, "Forces", "Fᵢ = −∂E/∂xᵢ\n→ Newton's equations\n(molecular dynamics)", color=MUTED)
    s.text(490, 305, "Trained on DFT calculations; ~DFT accuracy at a tiny fraction of the cost.  "
           "T0 uses one published model, MACE-OFF23 small (organic, neutral molecules only).", 11, MUTED, italic=True)
    s.save("mlip.svg")


# ------------------------------------------------------------------------------------------------ 5. MD loop: ASE vs JAX
def md_loop():
    s = SVG(980, 330, "Why a compiled MD loop can be faster: who waits for whom")
    def lane(y, label):
        s.text(115, y + 20, label, 11.5, INK2, anchor="end")
        s.add(f'<line x1="125" y1="{y + 32}" x2="960" y2="{y + 32}" stroke="{LINE}"/>')
    s.text(20, 28, "PyTorch + ASE (one molecule): every step goes CPU → GPU → CPU", 13, INK, anchor="start", weight="600")
    lane(45, "CPU (Python)"); lane(85, "GPU")
    x = 130
    for i in range(7):
        s.add(f'<rect x="{x}" y="{50}" width="62" height="24" rx="3" fill="{TINT[ORANGE]}" stroke="{ORANGE}"/>')
        s.add(f'<rect x="{x + 64}" y="{90}" width="40" height="24" rx="3" fill="{ORANGE}"/>')
        s.add(f'<line x1="{x + 62}" y1="74" x2="{x + 64}" y2="90" stroke="{MUTED}"/>')
        x += 118
    s.text(160, 65, "step", 9.5, INK2); s.text(214, 106, "MACE", 9.5, "#fff")
    s.text(545, 140, "orange boxes on the CPU lane: integrator, neighbour list and graph built in Python each step; the GPU idles in between",
           10.5, MUTED, italic=True)
    s.text(20, 185, "JAX (b molecules): 1,000 steps compiled into one GPU program (jax.lax.scan)", 13, INK, anchor="start", weight="600")
    lane(200, "CPU (Python)"); lane(240, "GPU")
    s.add(f'<rect x="130" y="205" width="40" height="24" rx="3" fill="{TINT[BLUE]}" stroke="{BLUE}"/>')
    s.text(150, 221, "launch", 9.5, INK2)
    s.add(f'<rect x="172" y="245" width="760" height="24" rx="3" fill="{BLUE}"/>')
    s.text(552, 261, "integrator + neighbour edges + MACE for all b molecules, step after step, no return to Python", 10.5, "#fff")
    s.text(545, 300, "the cost: the program is compiled once per array shape (timed separately), and all molecules must be padded to fixed sizes",
           10.5, MUTED, italic=True)
    s.save("md_loop.svg")


# ------------------------------------------------------------------------------------------------ 6. the four ways to run b molecules
def batching():
    s = SVG(980, 420, "Four ways to simulate b molecules on one GPU")
    W = 225
    xs = [15, 260, 505, 750]
    titles = [("Sequential", "ASE, one process,\nmolecules one after another"),
              ("MPS pool", "P processes share the GPU\nthrough the CUDA MPS daemon"),
              ("Batched PyTorch", "one MACE call for all b:\nTorchSim, or one big Atoms"),
              ("JAX padded graph", "all b in one fixed-size graph,\none compiled loop")]
    cols = [ORANGE, ORANGE, ORANGE, BLUE]
    for x, (t, b), c in zip(xs, titles, cols):
        s.box(x, 20, W, 64, t, b, color=c, tsize=13)
        s.add(f'<rect x="{x}" y="96" width="{W}" height="300" rx="8" fill="#fbfbfa" stroke="{LINE}"/>')
    def mol(x, y, r=11, c=INK2, fill="#fff"):
        import math
        p = " ".join(f"{x + r * math.cos(a * math.pi / 3 + math.pi / 6):.1f},{y + r * math.sin(a * math.pi / 3 + math.pi / 6):.1f}" for a in range(6))
        s.add(f'<polygon points="{p}" fill="{fill}" stroke="{c}" stroke-width="1.5"/>')
    # sequential
    x = xs[0]
    s.box(x + 20, 110, W - 40, 34, "1 Python process", "", color=ORANGE, tsize=11)
    for i in range(5):
        mol(x + 45 + i * 34, 175)
    s.text(x + W / 2, 205, "mol 1 → mol 2 → … → mol b", 10.5, INK2)
    s.box(x + 20, 320, W - 40, 50, "GPU", "busy only during each\nsmall MACE call", color=MUTED, tsize=11, bsize=10)
    s.arrow(x + W / 2, 215, x + W / 2, 318, ORANGE)
    s.text(x + W / 2, 260, "time ∝ b", 11, INK, weight="600")
    # MPS
    x = xs[1]
    s.box(x + 20, 110, W - 40, 30, "job queue: 1 … b", "", color=MUTED, tsize=11)
    for i in range(4):
        s.box(x + 16 + i * 50, 160, 44, 40, f"w{i + 1}", "", color=ORANGE, tsize=10.5)
        mol(x + 38 + i * 50, 222, 9)
        s.arrow(x + 38 + i * 50, 234, x + W / 2, 265, ORANGE, width=1.2)
    s.box(x + 20, 268, W - 40, 34, "MPS daemon", "", color=ORANGE, tsize=11)
    s.arrow(x + W / 2, 302, x + W / 2, 318, ORANGE)
    s.box(x + 20, 320, W - 40, 50, "GPU", "kernels from P processes\nrun side by side", color=MUTED, tsize=11, bsize=10)
    # batched torch
    x = xs[2]
    s.box(x + 20, 110, W - 40, 34, "1 Python process", "", color=ORANGE, tsize=11)
    for i in range(3):
        for j in range(3):
            mol(x + 65 + i * 48, 175 + j * 32, 10)
    s.text(x + W / 2, 285, "TorchSim: a batch of systems\nconcat: one Atoms, ≥ 30 Å apart", 10.5, INK2)
    s.box(x + 20, 320, W - 40, 50, "GPU", "one MACE call per step\nfor all b molecules", color=MUTED, tsize=11, bsize=10)
    # JAX
    x = xs[3]
    s.box(x + 20, 110, W - 40, 34, "1 compiled program", "", color=BLUE, tsize=11)
    for i in range(3):
        for j in range(3):
            mol(x + 65 + i * 48, 175 + j * 32, 10, c=BLUE)
    s.add(f'<rect x="{x + 40}" y="155" width="{W - 80}" height="104" rx="6" fill="none" stroke="{BLUE}" stroke-dasharray="4 3"/>')
    s.text(x + W / 2, 285, "fixed caps on atoms and edges;\nunused slots are padding", 10.5, INK2)
    s.box(x + 20, 320, W - 40, 50, "GPU", "the whole loop runs here\n(1,000 steps per launch)", color=MUTED, tsize=11, bsize=10)
    s.save("batching.svg")


# ------------------------------------------------------------------------------------------------ 7. the GH200 node
def hardware():
    s = SVG(980, 300, "The dual GH200 node and how T0 uses it")
    for k, (x, used) in enumerate(((20, True), (500, False))):
        c = BLUE if used else MUTED
        s.add(f'<rect x="{x}" y="40" width="460" height="230" rx="12" fill="{"#f7fafe" if used else "#fafaf9"}" stroke="{c}" stroke-width="{2 if used else 1.2}"/>')
        s.text(x + 230, 30, f"superchip {k}" + ("  ← T0 runs here" if used else "  (other users; keep idle)"), 12.5, c if used else MUTED, weight="600")
        s.box(x + 20, 70, 175, 170, f"Grace CPU {k}", "72 ARM cores (aarch64)\n~480 GB LPDDR5X\n" + (f"cores 0–71, NUMA node 0" if used else "cores 72–143, NUMA node 1"),
              color=c)
        s.box(x + 285, 70, 155, 170, f"H100 GPU {k}", "144 GB HBM3e\n" + ("CUDA_VISIBLE_DEVICES=0" if used else "GPU 1"), color=c)
        s.arrow(x + 197, 155, x + 283, 155, c, width=3, head=False)
        s.text(x + 240, 145, "NVLink-C2C", 10, c)
    s.text(490, 292, "Every T0 run: CUDA_VISIBLE_DEVICES=0  numactl --cpunodebind=0 --membind=0   (one GPU, its own CPU and memory)", 11.5, INK, weight="600")
    s.save("hardware.svg")


# ------------------------------------------------------------------------------------------------ 8. the three containers as layer cakes
def layers():
    s = SVG(980, 560, "What is inside each container, layer by layer")
    cols = [("A: JAX stack", CA, 110), ("B: PyTorch stack (+ TorchSim)", CB, 395), ("C: CPU tools", CC, 680)]
    rows = [
        ("our code", ["t0/md_jax.py\nweight converter (mace-jax-from-torch)", "t0/md_torch.py, run_mps.sh\nequivalence gates", "slice/ stages (planned)\nselect_species.py"]),
        ("science packages", ["mace-jax 0.2.0, jax-md 0.2.29\n(+ mace-torch, CPU torch: converter only)", "mace-torch 0.3.17, ase 3.29\ntorch-sim-atomistic 0.6.2", "pahfit (git ca599f9), AmesPAHdb\nsuite (git e8961b2), pysr 2.7.2"]),
        ("building blocks", ["e3nn-jax, jraph, flax, optax\ncuequivariance-jax 0.10.0", "e3nn 0.4.4, matscipy\ncuequivariance-torch 0.10.0", "astropy, specutils, scikit-learn\npandas, juliacall → Julia 1.13.3"]),
        ("numerical engine", ["JAX 0.11.2 / jaxlib (XLA compiler)", "PyTorch 2.15.0a0 (NVIDIA build)\ntriton", "numpy / scipy\nSymbolicRegression.jl (in Julia)"]),
        ("GPU libraries", ["pip wheels: nvidia-*-cu12 (CUDA 12.9)\nthe ONLY CUDA in the image", "NGC: CUDA 13.4 + compat libcuda\n615.71 (forward compatibility)", "none (CPU only)"]),
        ("Python", ["CPython 3.12.3 (Ubuntu), venv /opt/venv", "CPython 3.12.3 (NGC image), no venv", "CPython 3.12.14 from uv, OpenSSL 3.5\nvenv /opt/venv"]),
        ("base image", ["ubuntu:24.04 (7.1 GB image)", "nvcr.io/nvidia/pytorch:26.10-py3\n(12 GB image)", "ubuntu:24.04 (1.3 GB image)"]),
    ]
    rh, top = 62, 60
    for t, c, x in cols:
        s.text(x + 137, 30, t, 14, c, weight="700")
    for i, (lab, cells) in enumerate(rows):
        y = top + i * rh
        for (t, c, x), cell in zip(cols, cells):
            fill = "#ffffff" if i == 0 else TINT[c]
            s.box(x, y, 275, rh - 6, "", cell, color=c, fill=fill, bsize=10.5, dashed=(i == 0))
        s.text(100, y + rh / 2 + 2, lab, 10.5, MUTED, anchor="end")
    y = top + len(rows) * rh
    s.add(f'<rect x="20" y="{y}" width="935" height="44" rx="8" fill="#f1f0ed" stroke="{MUTED}"/>')
    s.text(490, y + 18, "shared host: Linux aarch64 (64 KiB pages) · NVIDIA driver 595.45 (CUDA 13.2) · Apptainer 1.5.5", 11.5, INK, weight="600")
    s.text(490, y + 35, "apptainer exec --nv binds the host driver into A and B; C never touches the GPU", 10.5, INK2)
    s.save("layers.svg")


# ------------------------------------------------------------------------------------------------ 9. dependency maps (curated from pip metadata)
def depmap(img, title, rows, color, name, notes=()):
    D = json.load(open(os.path.join(DATA, "deps.json")))[img]
    W = 980
    rh = 92
    s = SVG(W, 70 + rh * len(rows) + 30 * bool(notes), title)
    pos = {}
    for r, (lab, nodes) in enumerate(rows):
        y = 50 + r * rh
        s.text(W - 8, y + 26, lab, 10, MUTED, anchor="end")
        n = len(nodes)
        bw = min(150, (W - 120) / n - 10)
        x0 = (W - 100 - n * (bw + 10)) / 2 + 5
        for i, nd in enumerate(nodes):
            key, label = nd if isinstance(nd, tuple) else (nd, nd)
            if "\n" not in label and len(label) > 13 and "-" in label:
                cut = min((j for j, ch in enumerate(label) if ch == "-"), key=lambda j: abs(j - len(label) / 2))
                label = label[:cut + 1] + "\n" + label[cut + 1:]
            p = D.get(key)
            ver = p["version"].split("+")[0] if p else ""
            req = p and p.get("requested")
            ours = key.startswith("ours:")
            col = INK if ours else (color if req else MUTED)
            x = x0 + i * (bw + 10)
            s.box(x, y, bw, 52, label, "" if ours or not ver else ver[:18], color=col, fill="#fff" if not req else TINT.get(color),
                  tsize=10.8 if bw >= 95 else 9.6, bsize=9.5, dashed=ours, data=key, stroke_w=2 if req else 1.1)
            pos[key] = (x + bw / 2, y, y + 52)
    keys = set(pos)
    edges = []
    for k in keys:
        p = D.get(k)
        if p:
            for r in p["requires"]:
                if r in keys: edges.append((k, r))
    for k, deps in EXTRA_EDGES.get(img, {}).items():
        for r in deps: edges.append((k, r))
    for a, b in edges:
        (xa, ya0, ya1), (xb, yb0, yb1) = pos[a], pos[b]
        if yb0 > ya0:
            s.items.insert(0, f'<path d="M{xa:.1f},{ya1} C{xa:.1f},{ya1 + 30} {xb:.1f},{yb0 - 30} {xb:.1f},{yb0}" fill="none" '
                              f'stroke="{LINE}" stroke-width="1.1" data-edge="{a}>{b}"/>')
    s.items.insert(0, "")  # keep rect background first after save
    for i, n in enumerate(notes):
        s.text(20, 70 + rh * len(rows) + 4 + i * 15, n, 10.5, MUTED, anchor="start", italic=True)
    # move the background rect fix: save() puts the background before items, so edges are drawn under boxes.
    s.save(name)

EXTRA_EDGES = {
    "A_jax": {"ours:md_jax": ["mace-jax", "jax-md"], "ours:convert": ["mace-jax", "mace-torch"]},
    "B_torchsim": {"ours:md_torch": ["mace-torch", "ase", "torch-sim-atomistic"], "cuequivariance-torch": ["torch"],
                   "cuequivariance-ops-torch-cu13": ["torch"]},
    "C_cpu": {"ours:slice": ["amespahdbpythonsuite", "pahfit", "pysr", "scikit-learn", "h5py"],
              "juliacall": ["ours:julia"]},
}


def depmaps():
    depmap("A_jax", "Container A: who needs whom", [
        ("our code", [("ours:md_jax", "t0/md_jax.py"), ("ours:convert", "mace-jax-from-torch")]),
        ("MD + model", ["jax-md", "mace-jax", "mace-torch"]),
        ("layers", ["e3nn-jax", "jraph", "cuequivariance-jax", "cuequivariance", "flax", "optax", "chex", "e3nn", "matscipy", "ase"]),
        ("engines", ["jax", "jaxlib", "jax-cuda12-plugin", "cuequivariance-ops-jax-cu12", "torch"]),
        ("GPU libs", ["jax-cuda12-pjrt", "cuequivariance-ops-cu12", "nvidia-cublas-cu12", "nvidia-cudnn-cu12", "nvidia-cusparse-cu12", "nvidia-cuda-nvrtc-cu12", "nvidia-nccl-cu12"]),
    ], CA, "deps_A.svg", notes=["Thick outline: named in the recipe. Grey: pulled in as a dependency. Lines: 'requires' from the installed packages' metadata (pip show).", "numpy and scipy are left out: almost everything needs them."])
    depmap("B_torchsim", "Container B (+ TorchSim): who needs whom", [
        ("our code", [("ours:md_torch", "t0/md_torch.py")]),
        ("MD + model", ["ase", "mace-torch", "torch-sim-atomistic"]),
        ("layers", ["e3nn", "matscipy", "torch-ema", "torchmetrics", "cuequivariance-torch", "cuequivariance", "nvalchemi-toolkit-ops", "tables", "h5py"]),
        ("engines", ["torch", "opt-einsum-fx", "cuequivariance-ops-torch-cu13", "warp-lang"]),
        ("GPU libs", ["triton", "cuequivariance-ops-cu13", "nvidia-cublas", "nvidia-cuda-nvrtc"]),
    ], CB, "deps_B.svg", notes=["torch, triton and the CUDA libraries come from the NGC image (not pip); the recipe forbids pip from replacing them. numpy and scipy are left out."])
    depmap("C_cpu", "Container C: who needs whom", [
        ("our code", [("ours:slice", "slice/ stage scripts")]),
        ("science", ["amespahdbpythonsuite", "pahfit", "pysr", "scikit-learn"]),
        ("layers", ["specutils", "astropy", "lxml", "vtk", "fnnls", "juliacall", "pandas", "h5py", "matplotlib"]),
        ("engines", ["gwcs", "ndcube", "asdf", "juliapkg", ("ours:julia", "Julia 1.13.3 +\nSymbolicRegression.jl")]),
    ], CC, "deps_C.svg", notes=["Julia is not a Python package: juliapkg downloads it (and SymbolicRegression.jl) into /opt/julia_env at build time. numpy and scipy are left out."])


# ------------------------------------------------------------------------------------------------ 10. files between containers
def dataflow():
    s = SVG(980, 400, "Files passed between the containers in T0")
    # host files (middle column)
    s.text(490, 26, "host files (local NVMe and the repository)", 12.5, INK, weight="600")
    files = [(110, "MACE-OFF23_small.model", "PyTorch checkpoint, float64\n(sha256 165cce4c…)"),
             (205, "mace-jax bundles", "float64 master + float32 copies;\ncueq / conv_fusion variants"),
             (290, "t0/species_T0.csv, .xyz", "20 species, DFT geometries"),
             ]
    for y, t, b in files:
        s.box(370, y - 34, 240, 68, t, b, color=INK, fill="#ffffff", tsize=11.5, bsize=10)
    s.box(370, 338, 240, 50, "reports/t0_timing_*.csv", "one row per timed run", color=INK, fill="#ffffff", tsize=11.5, bsize=10)
    s.box(370, 40, 240, 34, "PAHdb v4.00 XML (503 MB)", "", color=INK, fill="#ffffff", tsize=11.5)
    # containers
    s.box(20, 60, 260, 300, "", "", color=CA, fill="#f7fafe")
    s.text(150, 82, "A: JAX", 13, CA, weight="700")
    s.box(40, 150, 220, 60, "mace-jax-from-torch", "run once, under JAX_ENABLE_X64=1", color=CA, tsize=11.5, bsize=10)
    s.box(40, 250, 220, 60, "t0/md_jax.py", "timed JAX runs", color=CA, tsize=11.5, bsize=10)
    s.box(700, 60, 260, 120, "", "", color=CB, fill="#fffaf7")
    s.text(830, 82, "B+: PyTorch (timing)", 13, CB, weight="700")
    s.box(720, 96, 220, 66, "t0/md_torch.py", "sequential, MPS pool,\nTorchSim, concat", color=CB, tsize=11.5, bsize=10)
    s.box(700, 210, 260, 110, "", "", color=CC, fill="#f6fcf9")
    s.text(830, 232, "C: CPU tools", 13, CC, weight="700")
    s.box(720, 246, 220, 60, "species selection", "AmesPAHdbPythonSuite", color=CC, tsize=11.5, bsize=10)
    # arrows
    s.arrow(370, 100, 262, 160, CA, label="reads", ldy=-4)
    s.arrow(262, 195, 368, 205, CA, label="writes", ldy=12)
    s.arrow(370, 215, 262, 268, CA)
    s.arrow(370, 300, 262, 290, CA)
    s.arrow(262, 300, 368, 350, CA)
    s.arrow(612, 100, 718, 120, CB, label="loads directly")
    s.arrow(612, 300, 718, 150, CB)
    s.arrow(718, 155, 612, 345, CB)
    s.arrow(612, 57, 718, 265, CC)
    s.arrow(718, 290, 612, 300, CC, label="writes", ldy=16)
    s.save("dataflow.svg")


if __name__ == "__main__":
    dataflow()
    os.makedirs(FIG, exist_ok=True)
    for f in (big_picture, roadmap, pipeline, mlip, md_loop, batching, hardware, layers, depmaps):
        f()
