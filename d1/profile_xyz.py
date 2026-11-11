"""Structure profile of PAH xyz files (standard library only; Python >= 3.6, so it runs on a login node):
Nc, NH, N, O, sp3 C, CH2, bare edge C, pentagons (chordless 5-cycles), solo/duo/trio/quartet H counts (runs of bonded
aromatic CH carbons; runs longer than 4 count as quartet). Usage: python3 d1/profile_xyz.py XYZ_DIR OUT.csv"""
import sys, os, math, csv, collections

R = {'C': 0.76, 'H': 0.31, 'N': 0.71, 'O': 0.66}


def read(path):
    with open(path) as f:
        n = int(f.readline()); f.readline()
        at = []
        for _ in range(n):
            p = f.readline().split()
            at.append((p[0], float(p[1]), float(p[2]), float(p[3])))
    return at


def bonds(at):
    nb = [[] for _ in at]
    for i in range(len(at)):
        si, xi, yi, zi = at[i]
        for j in range(i + 1, len(at)):
            sj, xj, yj, zj = at[j]
            if si == 'H' and sj == 'H':
                continue
            d = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)
            if d < 1.15 * (R[si] + R[sj]):
                nb[i].append(j); nb[j].append(i)
    return nb


def five_rings(at, nb):
    heavy = [i for i, a in enumerate(at) if a[0] != 'H']
    rings = set()
    for s in heavy:
        stack = [(s, [s])]
        while stack:
            v, path = stack.pop()
            if len(path) == 5:
                if s in nb[v]:
                    rings.add(frozenset(path))
                continue
            for w in nb[v]:
                if at[w][0] != 'H' and w not in path and w > s:
                    stack.append((w, path + [w]))
    # keep only chordless 5-cycles (true rings, not cuts through fused 6-rings)
    out = 0
    for r in rings:
        rl = list(r)
        chords = sum(1 for a in rl for b in nb[a] if b in r) // 2
        if chords == 5:
            out += 1
    return out


def profile(path):
    return profile_atoms(read(path), os.path.basename(path))


def components(at, nb):
    seen, n = set(), 0
    for i in range(len(at)):
        if i in seen:
            continue
        n += 1; st = [i]; seen.add(i)
        while st:
            v = st.pop()
            for w in nb[v]:
                if w not in seen:
                    seen.add(w); st.append(w)
    return n


def profile_atoms(at, name):
    """at: list of (symbol, x, y, z); name: file name, '<formula>_<uid>.xyz'."""
    nb = bonds(at)
    el = collections.Counter(a[0] for a in at)
    nC, nH = el['C'], el['H']
    sp3 = sum(1 for i, a in enumerate(at) if a[0] == 'C' and len(nb[i]) == 4)
    nHonC = lambda i: sum(1 for j in nb[i] if at[j][0] == 'H')
    # aromatic CH carbons: C with 3 neighbours, one of them H
    ch = set(i for i, a in enumerate(at) if a[0] == 'C' and len(nb[i]) == 3 and nHonC(i) == 1)
    seen, groups = set(), collections.Counter()
    for i in ch:
        if i in seen:
            continue
        comp, st = 0, [i]; seen.add(i)
        while st:
            v = st.pop(); comp += 1
            for w in nb[v]:
                if w in ch and w not in seen:
                    seen.add(w); st.append(w)
        groups[min(comp, 4)] += comp          # count H atoms in each adjacency class
    bare = sum(1 for i, a in enumerate(at) if a[0] == 'C' and len(nb[i]) == 2)   # dehydrogenated edge C
    ch2 = sum(1 for i, a in enumerate(at) if a[0] == 'C' and nHonC(i) == 2)
    tot = sum(groups.values()) or 1
    return dict(file=name, uid=name.split('_')[-1].split('.')[0], frags=components(at, nb),
                nC=nC, nH=nH, nN=el['N'], nO=el['O'], other=sum(v for k, v in el.items() if k not in R),
                sp3C=sp3, CH2=ch2, bareC=bare, pent=five_rings(at, nb),
                solo=groups[1], duo=groups[2], trio=groups[3], quartet=groups[4],
                f_solo=round(groups[1] / tot, 3), f_duo=round(groups[2] / tot, 3),
                f_trio=round(groups[3] / tot, 3), f_quart=round(groups[4] / tot, 3))


if __name__ == '__main__':
    d, out = sys.argv[1], sys.argv[2]
    rows = [profile(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.endswith('.xyz')]
    with open(out, 'w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(len(rows), 'species ->', out)
