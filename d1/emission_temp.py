# How much of a PAH's IR emission comes out above a given temperature (docs/plan/d1_hangang_new_species.md,
# "Emission temperatures")? Thermal approximation, as in PAHdb's cascade: after absorbing a photon of energy E the
# molecule has U(T_max) = E, U(T) = sum_i h nu_i / (exp(h nu_i / kT) - 1) over its harmonic modes; it then cools by IR
# emission alone, so the energy emitted while its temperature is above T_c is U(T_max) - U(T_c). The emitted fraction
# above T_c is therefore 1 - U(T_c)/E (0 if T_max < T_c). Frequencies from PAHdb v4.00 (all 3N-6 modes, as given).
#   python emission_temp.py <pahdb xml.gz> <species csv with a species column named <formula>_v4u<uid>> <out csv>
import csv, gzip, sys
import xml.etree.ElementTree as ET
import numpy as np

NS = "{http://www.astrochemistry.org/pahdb/theoretical}"
H_C_K = 1.4387769            # h c / k in cm K
CM_EV = 1.239841984e-4       # 1 cm^-1 in eV
ENERGIES = (3.0, 6.0, 8.0, 10.0, 13.6)
TCUTS = (600.0, 900.0, 1200.0)


def U(nu, T):
    return float((nu * CM_EV / np.expm1(H_C_K * nu / T)).sum())


def t_max(nu, E):
    lo, hi = 10.0, 1e5
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if U(nu, mid) < E else (lo, mid)
    return np.sqrt(lo * hi)


def main():
    xml, species_csv, out = sys.argv[1:4]
    want = {int(r["species"].split("_v4u")[1]): r["species"] for r in csv.DictReader(open(species_csv))}
    rows = []
    with gzip.open(xml) as f:
        for ev, el in ET.iterparse(f, events=("end",)):
            if el.tag != NS + "specie":
                continue
            uid = int(el.get("uid"))
            if uid in want:
                nu = np.array([float(m.find(NS + "frequency").text) for m in el.find(NS + "transitions").findall(NS + "mode")])
                nat = len(el.find(NS + "geometry").findall(NS + "atom"))
                nu = nu[nu > 0]
                sp = want[uid]; row = dict(species=sp, nC=int(sp[1:sp.index("H")]), n_modes=len(nu), n_expected=3 * nat - 6)
                for E in ENERGIES:
                    row[f"Tmax_{E:g}eV"] = round(t_max(nu, E))
                    for Tc in TCUTS:
                        row[f"frac_above_{Tc:.0f}K_{E:g}eV"] = round(max(0.0, 1 - U(nu, Tc) / E), 4)
                rows.append(row)
            el.clear()
    with open(out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    bad = [r for r in rows if r["n_modes"] != r["n_expected"]]
    print(f"{len(rows)} of {len(want)} species -> {out}; mode count != 3N-6 for {len(bad)}")
    bins = [(0, 30), (30, 50), (50, 80), (80, 120), (120, 999)]
    for Tc in TCUTS:
        print(f"\nShare of emitted IR energy above {Tc:.0f} K (median per Nc bin); T_max in brackets:")
        print("  Nc        " + "".join(f"{E:>14g} eV" for E in ENERGIES))
        for lo, hi in bins:
            R = [r for r in rows if lo <= r["nC"] < hi]
            if R:
                print(f"  {lo:3d}-{hi:<4d}({len(R):3d})" + "".join(
                    f"{np.median([r[f'frac_above_{Tc:.0f}K_{E:g}eV'] for r in R]):7.2f} [{np.median([r[f'Tmax_{E:g}eV'] for r in R]):5.0f}]"
                    for E in ENERGIES))


if __name__ == "__main__":
    main()
