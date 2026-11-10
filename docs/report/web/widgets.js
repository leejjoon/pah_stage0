/* Interactive figures for the HTML build. Each <div class="figure" data-widget="..."> keeps its static image and
   caption (used by print, EPUB and PDF); on screen the widget is inserted above the caption and the image hidden.
   Data: window.REPORT_DATA (web/report-data.js, written by scripts/make_figures.py). No external libraries. */
(function () {
  "use strict";
  const D = window.REPORT_DATA;
  const NS = "http://www.w3.org/2000/svg";
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  function el(tag, attrs, parent, text) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs || {}) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function h(tag, attrs, parent, text) {
    const e = document.createElement(tag);
    for (const k in attrs || {}) { if (k === "class") e.className = attrs[k]; else e.setAttribute(k, attrs[k]); }
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  const tip = h("div", { class: "tooltip", role: "status" }, document.body);
  function showTip(evt, html) {
    tip.innerHTML = html; tip.style.display = "block";
    const r = tip.getBoundingClientRect();
    let x = evt.clientX + 14, y = evt.clientY + 14;
    if (x + r.width > window.innerWidth - 8) x = evt.clientX - r.width - 14;
    if (y + r.height > window.innerHeight - 8) y = evt.clientY - r.height - 14;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  }
  const hideTip = () => { tip.style.display = "none"; };
  function seg(parent, options, value, onChange) {
    const s = h("span", { class: "seg", role: "group" }, parent);
    const btns = options.map(([v, label]) => {
      const b = h("button", { type: "button", "aria-pressed": String(v === value) }, s, label);
      b.addEventListener("click", () => { btns.forEach((x) => x.setAttribute("aria-pressed", "false")); b.setAttribute("aria-pressed", "true"); onChange(v); });
      return b;
    });
    return s;
  }
  function mount(fig, build) {
    const box = h("div", { class: "widget" });
    const figure = fig.querySelector("figure");
    figure.insertBefore(box, figure.querySelector("figcaption"));
    if (fig.getAttribute("data-widget") === "deps") h("div", { class: "note" }, box, "Package explorer: every installed package, searchable; click a name to follow the dependency links.").style.marginBottom = "6px";
    try { build(box, fig); } catch (e) { console.error(e); box.remove(); return; }
    if (fig.getAttribute("data-widget") === "deps") { box.style.marginTop = "10px"; return; }   // keep the curated static map too
    fig.classList.add("has-widget");
    const t = h("button", { type: "button", class: "static-toggle" }, null, "show the static (print) version");
    figure.insertBefore(t, figure.querySelector("figcaption"));
    t.addEventListener("click", () => { fig.classList.toggle("show-static"); t.textContent = fig.classList.contains("show-static") ? "hide the static version" : "show the static (print) version"; });
  }

  /* ------------------------------------------------------------------ molecule viewer */
  const HCLS = { 1: ["solo", "--s1", "circle"], 2: ["duo", "--s2", "square"], 3: ["trio", "--s3", "triangle"], 4: ["quartet", "--s5", "diamond"] };
  function marker(g, shape, x, y, r, color) {
    const a = { fill: color, stroke: css("--surface"), "stroke-width": 0.06 };
    if (shape === "circle") return el("circle", { cx: x, cy: y, r: r, ...a }, g);
    if (shape === "square") return el("rect", { x: x - r * 0.9, y: y - r * 0.9, width: r * 1.8, height: r * 1.8, ...a }, g);
    if (shape === "triangle") return el("polygon", { points: `${x},${y - r * 1.15} ${x + r},${y + r * 0.7} ${x - r},${y + r * 0.7}`, ...a }, g);
    return el("polygon", { points: `${x},${y - r * 1.2} ${x + r},${y} ${x},${y + r * 1.2} ${x - r},${y}`, ...a }, g);
  }
  function molecules(box) {
    const mols = D.molecules.slice().sort((a, b) => a.n_c - b.n_c || a.formula.localeCompare(b.formula));
    const ctr = h("div", { class: "controls" }, box);
    h("span", {}, ctr, "Species:");
    const sel = h("select", { "aria-label": "species" }, ctr);
    mols.forEach((m, i) => h("option", { value: i }, sel, `${m.formula} — ${m.n_c} C, ${m.class} (PAHdb UID ${m.uid})`));
    sel.value = String(mols.findIndex((m) => m.formula === "C24H12"));
    const prev = h("button", { type: "button", class: "seg" }, ctr, "◀"); const next = h("button", { type: "button", class: "seg" }, ctr, "▶");
    [prev, next].forEach((b) => { b.style.padding = "3px 9px"; b.style.cursor = "pointer"; b.style.background = "var(--surface-2)"; b.style.color = "var(--ink)"; });
    prev.onclick = () => { sel.value = String((+sel.value + mols.length - 1) % mols.length); draw(); };
    next.onclick = () => { sel.value = String((+sel.value + 1) % mols.length); draw(); };
    const svg = el("svg", { role: "img", "aria-label": "molecule drawing" }, box);
    const legend = h("div", { class: "controls" }, box);
    const note = h("div", { class: "note" }, box);
    sel.addEventListener("change", draw);
    function draw() {
      const m = mols[+sel.value];
      svg.innerHTML = "";
      const xs = m.xy.map((p) => p[0]), ys = m.xy.map((p) => -p[1]);
      const pad = 1.4, x0 = Math.min(...xs) - pad, x1 = Math.max(...xs) + pad, y0 = Math.min(...ys) - pad, y1 = Math.max(...ys) + pad;
      const span = Math.max(x1 - x0, (y1 - y0) * 1.6, 12);
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
      svg.setAttribute("viewBox", `${cx - span / 2} ${cy - span / 3.2} ${span} ${span / 1.6}`);
      const g = el("g", {}, svg);
      m.bonds.forEach(([i, j]) => {
        const cc = m.Z[i] === 6 && m.Z[j] === 6;
        el("line", { x1: xs[i], y1: ys[i], x2: xs[j], y2: ys[j], stroke: cc ? css("--ink-2") : css("--muted"), "stroke-width": cc ? 0.11 : 0.07, "stroke-linecap": "round" }, g);
      });
      Object.entries(m.hclass).forEach(([hIdx, c]) => {
        const [name, col, shape] = HCLS[Math.min(c, 4)];
        const mk = marker(g, shape, xs[+hIdx], ys[+hIdx], 0.36, css(col));
        mk.style.cursor = "help";
        mk.addEventListener("mousemove", (e) => showTip(e, `<b>${name} H</b><br>${c} adjacent H-carrying carbon${c > 1 ? "s" : ""} on this stretch of edge`));
        mk.addEventListener("mouseleave", hideTip);
      });
      legend.innerHTML = "";
      Object.values(HCLS).forEach(([name, col]) => {
        const n = +m["n_" + name];
        const s = h("span", {}, legend); h("span", { class: "swatch", style: `background:${css(col)}` }, s); s.append(`${name} H: ${n}`);
        if (!n) s.style.opacity = 0.45;
      });
      note.textContent = `${m.formula}: ${m.n_c} carbon, ${m.n_h} hydrogen atoms, ${3 * (+m.n_c + +m.n_h) - 6} vibrational modes. ` +
        (m.planarity_A > 0.3 ? `Not quite flat: atoms sit up to ${m.planarity_A.toFixed(1)} Å out of the best-fit plane (crowded bays), so this is a projection. ` : "") +
        "Hover over an H for its edge group.";
    }
    draw();
  }

  /* ------------------------------------------------------------------ spectrum explorer */
  const PAIR_NAMES = { coronene: "coronene C24H12", circumcoronene: "circumcoronene C54H18", C26H16: "hexacene C26H16", circumcircumcoronene: "circumcircumcoronene C96H24" };
  function axes(svg, W, H, M, xs, ys, xt, yt, xlabel, ylabel, fmtx, fmty) {
    const g = el("g", { class: "axis" }, svg);
    yt.forEach((v) => { el("line", { x1: M.l, x2: W - M.r, y1: ys(v), y2: ys(v), class: "gridline" }, g); el("text", { x: M.l - 6, y: ys(v) + 4, "text-anchor": "end", "font-size": 11 }, g, fmty(v)); });
    xt.forEach((v) => { el("line", { x1: xs(v), x2: xs(v), y1: H - M.b, y2: H - M.b + 4 }, g); el("text", { x: xs(v), y: H - M.b + 17, "text-anchor": "middle", "font-size": 11 }, g, fmtx(v)); });
    el("line", { x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b }, g);
    el("text", { x: (M.l + W - M.r) / 2, y: H - 4, "text-anchor": "middle", "font-size": 12 }, g, xlabel);
    el("text", { x: 14, y: (M.t + H - M.b) / 2, "text-anchor": "middle", "font-size": 12, transform: `rotate(-90 14 ${(M.t + H - M.b) / 2})` }, g, ylabel);
  }
  function spectra(box) {
    const S = D.spectra;
    let pair = "coronene", mode = "emission", on = { n: true, c: true };
    const ctr = h("div", { class: "controls" }, box);
    const sel = h("select", { "aria-label": "molecule pair" }, ctr);
    Object.keys(S.pairs).forEach((k) => h("option", { value: k }, sel, PAIR_NAMES[k] || k));
    sel.onchange = () => { pair = sel.value; draw(); };
    seg(ctr, [["emission", "Emission after a 6 eV photon"], ["sticks", "Cold band list (km/mol)"]], mode, (v) => { mode = v; draw(); });
    [["n", "neutral", "--s1"], ["c", "cation", "--s2"]].forEach(([k, label, col]) => {
      const l = h("label", {}, ctr); const cb = h("input", { type: "checkbox" }, l); cb.checked = true;
      h("span", { class: "swatch", style: `background:${css(col)}` }, l); l.append(label);
      cb.onchange = () => { on[k] = cb.checked; draw(); };
    });
    const W = 820, H = 340, M = { l: 54, r: 14, t: 22, b: 40 };
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "spectrum" }, box);
    const note = h("div", { class: "note" }, box);
    const um = S.grid_cm.map((c) => 1e4 / c);
    const X0 = 2.8, X1 = 20, xs = (v) => M.l + (v - X0) / (X1 - X0) * (W - M.l - M.r);
    function draw() {
      svg.innerHTML = "";
      const [nu, cu] = S.pairs[pair].map(String);
      let ymax = 1.1;
      if (mode === "sticks") ymax = Math.max(...[nu, cu].filter((u, i) => on[i ? "c" : "n"]).flatMap((u) => S.sticks[u].filter((s) => 1e4 / s[0] > X0).map((s) => s[1])), 1) * 1.1;
      const ys = (v) => H - M.b - v / ymax * (H - M.t - M.b);
      const yt = mode === "sticks" ? niceTicks(0, ymax, 5) : [0, 0.25, 0.5, 0.75, 1];
      axes(svg, W, H, M, xs, ys, [4, 6, 8, 10, 12, 14, 16, 18, 20], yt, "wavelength (µm)", mode === "sticks" ? "intensity (km/mol)" : "emission (peak = 1)", (v) => v, (v) => +v.toFixed(2));
      [3.3, 6.2, 7.7, 8.6, 11.2, 12.7].forEach((b) => { el("line", { x1: xs(b), x2: xs(b), y1: M.t, y2: H - M.b, stroke: css("--grid"), "stroke-dasharray": "3 3" }, svg); el("text", { x: xs(b), y: M.t - 6, "text-anchor": "middle", "font-size": 10 }, svg, b); });
      [[nu, "n", "--s1"], [cu, "c", "--s2"]].forEach(([u, k, col]) => {
        if (!on[k]) return;
        if (mode === "emission") {
          const e = S.emission[u]; let d = "";
          um.forEach((x, i) => { if (x >= X0 && x <= X1) d += (d ? "L" : "M") + xs(x).toFixed(1) + "," + ys(e[i]).toFixed(1); });
          el("path", { d, fill: "none", stroke: css(col), "stroke-width": 2, "stroke-linejoin": "round" }, svg);
        } else {
          S.sticks[u].forEach(([c, I]) => { const x = 1e4 / c; if (x >= X0 && x <= X1) el("line", { x1: xs(x), x2: xs(x), y1: ys(0), y2: ys(I), stroke: css(col), "stroke-width": 2, opacity: 0.9 }, svg); });
        }
      });
      const cross = el("line", { y1: M.t, y2: H - M.b, stroke: css("--muted"), "stroke-width": 1, visibility: "hidden" }, svg);
      const hit = el("rect", { x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b, fill: "transparent" }, svg);
      hit.addEventListener("mousemove", (evt) => {
        const p = svg.createSVGPoint(); p.x = evt.clientX; p.y = evt.clientY;
        const q = p.matrixTransform(svg.getScreenCTM().inverse());
        const x = X0 + (q.x - M.l) / (W - M.l - M.r) * (X1 - X0);
        cross.setAttribute("x1", q.x); cross.setAttribute("x2", q.x); cross.setAttribute("visibility", "visible");
        let html = `<b>${x.toFixed(2)} µm</b> = ${(1e4 / x).toFixed(0)} cm⁻¹`;
        [[nu, "n", "neutral"], [cu, "c", "cation"]].forEach(([u, k, label]) => {
          if (!on[k]) return;
          if (mode === "emission") {
            let best = 0, bd = 1e9; um.forEach((v, i) => { const d = Math.abs(v - x); if (d < bd) { bd = d; best = i; } });
            html += `<br>${label}: ${S.emission[u][best].toFixed(3)}`;
          } else {
            const near = S.sticks[u].filter(([c]) => Math.abs(1e4 / c - x) < 0.08);
            if (near.length) html += `<br>${label}: ` + near.map(([c, I]) => `${c.toFixed(0)} cm⁻¹, ${I.toFixed(1)} km/mol`).join("; ");
          }
        });
        showTip(evt, html);
      });
      hit.addEventListener("mouseleave", () => { hideTip(); cross.setAttribute("visibility", "hidden"); });
      note.textContent = `PAHdb v4.00, UIDs ${nu} (neutral) and ${cu} (cation). ` + (mode === "emission"
        ? "Each spectrum is normalised to its own peak, so compare shapes, not heights between the two."
        : "One line per normal mode; harmonic, scaled DFT frequencies, intensities in km/mol.");
    }
    draw();
  }
  function niceTicks(a, b, n) {
    const step0 = (b - a) / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= step0);
    const t = []; for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) t.push(+v.toFixed(10)); return t;
  }

  /* ------------------------------------------------------------------ timing explorer */
  function family(s) {
    if (s.startsWith("JAX")) return ["JAX", "--s1"];
    if (s.startsWith("TorchSim")) return ["TorchSim", "--s2"];
    if (s.startsWith("ASE concatenated")) return ["concatenated Atoms", "--s3"];
    if (s.startsWith("ASE MPS")) return ["MPS pool", "--s7"];
    return ["sequential", "--muted"];
  }
  const isOn = (s) => /cueq (on|conv_fusion)/.test(s);
  function timing(box) {
    const T = D.timing;
    const sets = [["calib", "Calibration pass (1 repeat, 400 steps)"]];
    const done = T.full && T.full.length >= 31;
    if (T.full && T.full.length) sets.push(["full", done ? "Full grid (3 repeats, 10 ps runs)" : "Full grid so far (10 ps runs)"]);
    let set = done ? "full" : "calib", fam = { JAX: true, TorchSim: true, "concatenated Atoms": true, "MPS pool": true, sequential: true }, cq = { on: true, off: true }, tableView = false;
    const ctr = h("div", { class: "controls" }, box);
    seg(ctr, sets, set, (v) => { set = v; draw(); });
    const ctr2 = h("div", { class: "controls" }, box);
    Object.keys(fam).forEach((f) => {
      const l = h("label", {}, ctr2); const cb = h("input", { type: "checkbox" }, l); cb.checked = true;
      h("span", { class: "swatch", style: `background:${css(family(f === "JAX" ? "JAX" : f === "TorchSim" ? "TorchSim" : f === "concatenated Atoms" ? "ASE concatenated" : f === "MPS pool" ? "ASE MPS" : "seq")[1])}` }, l);
      l.append(f); cb.onchange = () => { fam[f] = cb.checked; draw(); };
    });
    [["on", "cueq on (solid)"], ["off", "cueq off (dashed)"]].forEach(([k, label]) => {
      const l = h("label", {}, ctr2); const cb = h("input", { type: "checkbox" }, l); cb.checked = true; l.append(label);
      cb.onchange = () => { cq[k] = cb.checked; draw(); };
    });
    const tb = h("label", {}, ctr2); const tcb = h("input", { type: "checkbox" }, tb); tb.append("table view");
    tcb.onchange = () => { tableView = tcb.checked; draw(); };
    const W = 820, H = 420, M = { l: 62, r: 16, t: 14, b: 44 };
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "timing plot" }, box);
    const tableDiv = h("div", {}, box);
    const note = h("div", { class: "note" }, box);
    function draw() {
      svg.innerHTML = ""; tableDiv.innerHTML = "";
      const rows = (T[set] || []).filter((r) => fam[family(r.series)[0]] && cq[isOn(r.series) ? "on" : "off"]);
      const series = [...new Set(rows.map((r) => r.series))];
      const best = {}; (T[set] || []).filter((r) => r.series.startsWith("JAX")).forEach((r) => { best[r.batch] = Math.min(best[r.batch] || 1e9, r.median); });
      svg.style.display = tableView ? "none" : "block";
      if (tableView) {
        const batches = [...new Set((T[set] || []).map((r) => r.batch))].sort((a, b) => a - b);
        const t = h("table", {}, tableDiv); const hr = h("tr", {}, h("thead", {}, t));
        ["series (s per molecule-ps, median)", ...batches.map((b) => "batch " + b)].forEach((x) => h("th", {}, hr, x));
        const tbody = h("tbody", {}, t);
        series.forEach((s) => { const tr = h("tr", {}, tbody); h("td", {}, tr, s); batches.forEach((b) => { const r = rows.find((x) => x.series === s && x.batch === b); h("td", {}, tr, r ? r.median.toPrecision(3) + (r.n > 1 ? ` (${r.min.toPrecision(3)}–${r.max.toPrecision(3)})` : "") : "—"); }); });
      } else {
        const vals = rows.flatMap((r) => [r.min, r.max]);
        const lo = Math.pow(10, Math.floor(Math.log10(Math.min(...vals, 0.1)))), hi = Math.pow(10, Math.ceil(Math.log10(Math.max(...vals, 1))));
        const xs = (b) => M.l + (Math.log10(b) - Math.log10(0.7)) / (Math.log10(150) - Math.log10(0.7)) * (W - M.l - M.r);
        const ys = (v) => H - M.b - (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) * (H - M.t - M.b);
        const yt = []; for (let v = lo; v <= hi * 1.001; v *= 10) yt.push(v);
        axes(svg, W, H, M, xs, ys, [1, 10, 100], yt, "batch: molecules simulated together", "s per molecule-ps (lower is faster)", (v) => v, (v) => (v >= 1 ? v.toFixed(0) : v.toString()));
        if (best[100]) {
          el("rect", { x: xs(80), width: xs(125) - xs(80), y: ys(best[100] * 5), height: ys(best[100] * 2) - ys(best[100] * 5), fill: css("--grid") }, svg);
          el("text", { x: xs(80) - 6, y: ys(best[100] * 3.2), "text-anchor": "end", "font-size": 10.5 }, svg, "2×–5× JAX");
        }
        const pts = [];
        series.forEach((s) => {
          const r = rows.filter((x) => x.series === s).sort((a, b) => a.batch - b.batch);
          const col = css(family(s)[1]);
          const d = r.map((x, i) => (i ? "L" : "M") + xs(x.batch).toFixed(1) + "," + ys(x.median).toFixed(1)).join("");
          el("path", { d, fill: "none", stroke: col, "stroke-width": 2, "stroke-dasharray": isOn(s) ? "" : "5 4" }, svg);
          r.forEach((x) => {
            if (x.n > 1) el("line", { x1: xs(x.batch), x2: xs(x.batch), y1: ys(x.min), y2: ys(x.max), stroke: col, "stroke-width": 1.5 }, svg);
            el("circle", { cx: xs(x.batch), cy: ys(x.median), r: 4.5, fill: col, stroke: css("--surface"), "stroke-width": 2 }, svg);
            pts.push([xs(x.batch), ys(x.median), x]);
          });
        });
        const hit = el("rect", { x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b, fill: "transparent" }, svg);
        hit.addEventListener("mousemove", (evt) => {
          const p = svg.createSVGPoint(); p.x = evt.clientX; p.y = evt.clientY; const q = p.matrixTransform(svg.getScreenCTM().inverse());
          let bp = null, bd = 18; pts.forEach((pt) => { const d = Math.hypot(pt[0] - q.x, pt[1] - q.y); if (d < bd) { bd = d; bp = pt; } });
          if (!bp) return hideTip();
          const x = bp[2];
          showTip(evt, `<b>${x.series}</b><br>batch ${x.batch}: ${x.median.toPrecision(3)} s per molecule-ps` + (x.n > 1 ? `<br>range ${x.min.toPrecision(3)}–${x.max.toPrecision(3)} over ${x.n} repeats` : `<br>${x.n} run`) +
            (best[x.batch] && !x.series.startsWith("JAX") ? `<br>${(x.median / best[x.batch]).toFixed(1)}× the best JAX time` : ""));
        });
        hit.addEventListener("mouseleave", hideTip);
      }
      note.textContent = (set === "calib" ? "Calibration only: one repeat of 100 + 400 steps per cell. " : (done ? "Medians of 3 repeats; bars span min to max. " : "Medians over the repeats finished so far; bars span min to max. ")) +
        (D.timing.bad ? "Excluded: " + Object.entries(D.timing.bad).map(([k, v]) => `${k} (${v})`).join("; ") + ". " : "") +
        (set === "full" && done ? "" : "Not the benchmark result until all three repeats have finished.");
    }
    draw();
  }

  /* ------------------------------------------------------------------ package explorer */
  const MAIN = { A_jax: "mace-jax", B_torchsim: "torch-sim-atomistic", C_cpu: "pysr" };
  function deps(box, fig) {
    const img = fig.getAttribute("data-image");
    const P = D.deps[img];
    let filter = "all", q = "", cur = MAIN[img];
    const ctr = h("div", { class: "controls" }, box);
    const search = h("input", { type: "search", placeholder: "search packages…", "aria-label": "search packages" }, ctr);
    const opts = [["all", `All (${Object.keys(P).length})`], ["req", "Named in the recipe"], ["top", "Nothing requires it"]];
    if (img === "B_torchsim") opts.splice(2, 0, ["added", "Added by our recipes"]);
    seg(ctr, opts, filter, (v) => { filter = v; list(); });
    const grid = h("div", { class: "deps-grid" }, box);
    const lst = h("div", { class: "deps-list", role: "list" }, grid);
    const det = h("div", { class: "deps-detail" }, grid);
    search.addEventListener("input", () => { q = search.value.toLowerCase(); list(); });
    function list() {
      lst.innerHTML = "";
      Object.keys(P).sort().forEach((k) => {
        const p = P[k];
        if (filter === "req" && !p.requested) return;
        if (filter === "added" && p.origin !== "added") return;
        if (filter === "top" && p.required_by.length) return;
        if (q && !(k.includes(q) || (p.summary || "").toLowerCase().includes(q))) return;
        const b = h("button", { type: "button", "aria-current": String(k === cur) }, lst);
        h("span", {}, b, p.name + (p.requested ? " ●" : ""));
        h("span", { class: "v" }, b, p.version.split("+")[0].slice(0, 14));
        b.onclick = () => select(k);
      });
    }
    function chips(parent, keys) {
      const c = h("div", { class: "chips" }, parent);
      if (!keys.length) h("span", { class: "note" }, c, "none");
      keys.forEach((k) => { const b = h("button", { type: "button" }, c, k); if (P[k]) b.onclick = () => select(k); else b.disabled = true; });
    }
    function select(k) {
      cur = k; det.innerHTML = "";
      const p = P[k];
      const t = h("h4", {}, det, p.name);
      h("span", { class: "tag" }, t, p.version);
      if (p.requested) h("span", { class: "tag" }, t, "named in recipe");
      if (p.origin) h("span", { class: "tag" }, t, p.origin === "added" ? "added by our recipes" : "from the NGC image");
      if (p.role) { const r = h("p", {}, det); h("b", {}, r, "Role here: "); r.append(p.role); }
      h("p", { class: "note" }, det, "Package's own summary: " + (p.summary || "(none in its metadata)"));
      // neighbourhood graph: required-by on the left, requires on the right
      const L = p.required_by.slice(0, 10), R = p.requires.slice(0, 10);
      const rows = Math.max(L.length, R.length, 1), W = 520, H = 30 + rows * 26;
      const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "dependency neighbourhood" }, det);
      const cy = H / 2;
      const node = (x, y, label, anchor, key, strong) => {
        const tx = el("text", { x, y: y + 4, "text-anchor": anchor, "font-size": 11.5 }, svg, label);
        if (strong) { tx.setAttribute("font-weight", 700); tx.style.fill = "var(--ink)"; }
        if (key && P[key]) { tx.style.cursor = "pointer"; tx.addEventListener("click", () => select(key)); }
      };
      L.forEach((k2, i) => { const y = 18 + i * 26 + (rows - L.length) * 13; el("path", { d: `M150,${y} C200,${y} 210,${cy} 255,${cy}`, fill: "none", stroke: css("--border"), "stroke-width": 1.4 }, svg); node(145, y, k2, "end", k2); });
      R.forEach((k2, i) => { const y = 18 + i * 26 + (rows - R.length) * 13; el("path", { d: `M265,${cy} C310,${cy} 320,${y} 370,${y}`, fill: "none", stroke: css("--border"), "stroke-width": 1.4 }, svg); node(375, y, k2, "start", k2); });
      el("circle", { cx: 260, cy, r: 6, fill: css("--s1") }, svg);
      node(260, cy - 14, p.name, "middle", null, true);
      el("text", { x: 10, y: 12, "font-size": 10 }, svg, "required by ←");
      el("text", { x: W - 10, y: 12, "font-size": 10, "text-anchor": "end" }, svg, "→ requires");
      h("div", { class: "note" }, det, "Required by" + (p.required_by.length > 10 ? ` (${p.required_by.length}, first 10 drawn)` : "") + ":");
      chips(det, p.required_by);
      h("div", { class: "note" }, det, "Requires:");
      chips(det, p.requires);
      list();
    }
    list(); select(cur);
  }

  const BUILDERS = { molecules, spectra, timing, deps };
  document.querySelectorAll(".figure[data-widget]").forEach((fig) => {
    const b = BUILDERS[fig.getAttribute("data-widget")];
    if (b && D) mount(fig, b);
  });

  /* theme toggle: auto → light → dark */
  const btn = h("button", { type: "button", class: "theme-toggle" }, document.body);
  const modes = ["auto", "light", "dark"];
  let mode = "auto";
  try { mode = localStorage.getItem("report-theme") || "auto"; } catch (e) {}
  function applyTheme() {
    if (mode === "auto") document.documentElement.removeAttribute("data-theme"); else document.documentElement.setAttribute("data-theme", mode);
    btn.textContent = "theme: " + mode;
  }
  btn.onclick = () => { mode = modes[(modes.indexOf(mode) + 1) % 3]; try { localStorage.setItem("report-theme", mode); } catch (e) {} applyTheme(); location.hash || null; rerender(); };
  function rerender() {  // colours are read from CSS tokens at draw time, so redraw on theme change
    document.querySelectorAll(".figure[data-widget]").forEach((fig) => {
      const w = fig.querySelector(".widget"); if (!w) return;
      w.remove(); const t = fig.querySelector(".static-toggle"); if (t) t.remove();
      fig.classList.remove("has-widget", "show-static"); mount(fig, BUILDERS[fig.getAttribute("data-widget")]);
    });
  }
  applyTheme();
  if (window.matchMedia) window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (mode === "auto") rerender(); });
})();
