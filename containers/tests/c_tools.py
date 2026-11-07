import os, time, numpy as np
t=time.time(); import pysr; print("pysr", pysr.__version__, "import", round(time.time()-t,1), "s")
from pysr import PySRRegressor
rng=np.random.default_rng(0); X=rng.normal(size=(100,2)); y=2.5*np.cos(X[:,1])+X[:,0]**2-0.5
m=PySRRegressor(niterations=40, binary_operators=["+","*","-","/"], unary_operators=["cos"], random_state=0, deterministic=True, parallelism="serial", progress=False, verbosity=0); m.fit(X,y); print("serial:", m.sympy())
m=PySRRegressor(niterations=20, binary_operators=["+","*"], unary_operators=["cos"], parallelism="multiprocessing", procs=4, progress=False, verbosity=0); m.fit(X,y); print("multiprocessing:", m.sympy())
import pahfit; from pahfit.model import Model
r=os.path.dirname(pahfit.__file__); print("pahfit", pahfit.__version__, [os.path.exists(f"{r}/packs/{p}") for p in ("science/classic.yaml","instrument/jwst.yaml")])
print("classic features:", len(Model.from_yaml("classic.yaml").features))
import amespahdbpythonsuite as a; print("suite ok")
