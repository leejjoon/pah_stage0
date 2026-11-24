# Peak vibrational temperature of PAHs after absorbing one UV photon, from PAHdb v4 harmonic frequencies, and the share
# of the emitted IR energy that comes out above a temperature (docs/plan/charged_mlff_plan.md, decision 2026-11-16).
#   python d1/peak_temperature.py /var/tmp/$USER-pah/data/pahdb-complete-theoretical-v4.00.xml
# - Quantum harmonic internal energy U(T) = sum h nu / (exp(h nu / kT) - 1) (zero-point excluded), pure CxHy, charge
#   0/+1; T_peak solves U(T_peak) = E for a cold start (E = 6/8/10/13.6 eV).
# - Share of the emitted energy above T_cut = (E - U(T_cut)) / E (energy conservation along the cascade, cold start).
# - For comparison, the classical temperature for the same energy, E / ((3N - 6) k), and with the zero-point energy
#   added (not a convention we use; it shows why the classical and quantum temperatures can differ by ~2x).
# Plain Python 3 with numpy; ~20 s.
import re, sys, xml.etree.ElementTree as ET
import numpy as np

NS = "{http://www.astrochemistry.org/pahdb/theoretical}"
CM_EV = 1.239841984e-4; K_EV = 8.617333262e-5
rows = []
for ev, el in ET.iterparse(sys.argv[1], events=("end",)):
    if el.tag != NS + "specie":
        continue
    f = el.findtext(NS + "formula"); q = int(el.findtext(NS + "charge"))
    m = re.fullmatch(r"C(\d+)H(\d+)[+-]*", f or "")
    if m and q in (0, 1):
        nu = np.array([float(x.text) for x in el.iter(NS + "frequency")])
        nu = nu[nu > 0]
        rows.append((int(el.get("uid")), f, q, int(m[1]), int(m[2]), nu))
    el.clear()


def U(nu, T):
    x = nu * CM_EV / (K_EV * T)
    return float(np.sum(nu * CM_EV / np.expm1(np.minimum(x, 700))))


def tpeak(nu, E):
    lo, hi = 10.0, 20000.0
    for _ in range(60):
        mid = (lo * hi) ** 0.5
        lo, hi = (mid, hi) if U(nu, mid) < E else (lo, mid)
    return (lo * hi) ** 0.5


bins = [(10, 20), (21, 30), (31, 40), (41, 50), (51, 80), (81, 100), (101, 150)]
print(f"{len(rows)} pure CxHy species (q 0/+1)")
print("nC bin  q  N   | Tpeak 8 / 10 / 13.6 eV (median, max) | frac >900K 8/10/13.6 (median) | T_classical 10 eV (median)")
for lo, hi in bins:
    for q in (0, 1):
        sel = [r for r in rows if lo <= r[3] <= hi and r[2] == q]
        if not sel:
            continue
        out = []
        for E in (8, 10, 13.6):
            t = np.array([tpeak(r[5], E) for r in sel]); out.append((np.median(t), t.max()))
        fr = [np.median([max(0.0, (E - U(r[5], 900.0)) / E) for r in sel]) for E in (8, 10, 13.6)]
        tcl = np.median([10 / ((3 * (r[3] + r[4]) - 6) * K_EV) for r in sel])
        tzp = np.median([(10 + 0.5 * r[5].sum() * CM_EV) / ((3 * (r[3] + r[4]) - 6) * K_EV) for r in sel])
        print(f"{lo:>3}-{hi:<3} {q} {len(sel):>4} | " + " / ".join(f"{a:5.0f} ({b:4.0f})" for a, b in out)
              + " | " + " / ".join(f"{x:.2f}" for x in fr) + f" | {tcl:5.0f} / +ZPE {tzp:5.0f}")

print("\nShare of emitted energy above T_cut = (E - U(T_cut)) / E (median; mean of the hottest decile), pure CxHy, q 0+1")
print("nC bin   N   | E eV: >1200 K  >1500 K  >1800 K")
for lo, hi in [(10, 20), (21, 30), (31, 40), (41, 50), (51, 80)]:
    sel = [r for r in rows if lo <= r[3] <= hi]
    for E in (6, 8, 10, 13.6):
        cells = []
        for Tc in (1200, 1500, 1800):
            fr = np.sort([max(0.0, (E - U(r[5], Tc)) / E) for r in sel])
            cells.append(f"{np.median(fr):.2f} ({fr[-max(1, len(fr) // 10):].mean():.2f})")
        print(f"{lo:>3}-{hi:<3} {len(sel):>4} | {E:>4}: " + "  ".join(cells))
