import csv,glob,re,subprocess,sys
D='<dir with profile_fits_v1_c{0,1,2}.csv>'
scr={r['uid']:r for r in csv.DictReader(open('d1/pahdb_v4_screen_spin_v1.csv'))}
old={r['uid'] for r in csv.DictReader(open('d1/pahdb_v4_exclude_nonkekule_v1.csv'))}
uid=lambda s: re.search(r'v4u(\d+)',s).group(1)
keep={'all':lambda u:True,'drop4':lambda u: scr[u]['verdict']=='ok' or u in old,'dropbelow':lambda u: scr[u]['verdict']=='ok',
      'closed':lambda u: scr[u]['verdict']=='ok' and scr[u]['mult']=='1'}
for k,fn in keep.items():
    fs=[]
    for f in sorted(glob.glob(D+'/profile_fits_v1_c?.csv')):
        R=list(csv.DictReader(open(f))); K=[r for r in R if fn(uid(r['species']))]
        o=f.replace('.csv',f'_{k}.csv'); w=csv.DictWriter(open(o,'w',newline=''),fieldnames=list(R[0])); w.writeheader(); w.writerows(K); fs.append(o)
    subprocess.run([sys.executable,'d1/profile_fit.py','rules',f'{D}/rules_{k}.csv',*fs],stdout=subprocess.DEVNULL,check=True)
T={k:{(r['band'],r['quantity']):r for r in csv.DictReader(open(f'{D}/rules_{k}.csv'))} for k in keep}
v1={(r['band'],r['quantity']):r for r in csv.DictReader(open('d1/tdep_rules_v1.csv'))}
print('reproduces v1:', all(abs(float(T['all'][k]['at_Nc50'])-float(v1[k]['at_Nc50']))<1e-9 and abs(float(T['all'][k]['per_decade_Nc'])-float(v1[k]['per_decade_Nc']))<1e-9 for k in v1), len(v1), len(T['all']))
print('n species:',{k:T[k][('3.3','position')]['n_species'] for k in keep})
print(f"{'band':>5} {'quantity':9} {'v1':>9} {'se':>7} {'scat':>7} | d(drop4) d(dropbelow) d(closed) in units of se | decade v1, se, d4, dbelow, dclosed")
for k in v1:
    a=T['all'][k]; v=float(a['at_Nc50']); se=float(a['se']) or 1e-9; sd=float(a['se_decade']) or 1e-9; pd=float(a['per_decade_Nc'])
    d=[float(T[x][k]['at_Nc50'])-v for x in ('drop4','dropbelow','closed')]; dd=[float(T[x][k]['per_decade_Nc'])-pd for x in ('drop4','dropbelow','closed')]
    print(f"{k[0]:>5} {k[1]:9} {v:+9.4f} {se:7.4f} {float(a['scatter']):7.4f} | "+' '.join(f'{x:+8.4f}({x/se:+5.1f})' for x in d)+f" | {pd:+8.3f} {sd:6.3f} "+' '.join(f'{x:+7.3f}' for x in dd))
