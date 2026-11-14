# Container D: GPU4PySCF on the GH200 (aarch64 + H100)

GPU4PySCF 1.9.2 for the B3LYP/6-31G\* training data of the charged MLFF (`docs/research/pyscf_gpu_jax.md`,
`docs/plan/handoff_gpu4pyscf_gh200.md`). Built and tested on the project node `gh200` on 2026-11-14.

## Contents

| Package | Version | Source |
|---|---|---|
| base image | `nvidia/cuda:12.9.1-devel-ubuntu24.04` (arm64) | Docker Hub; CUDA 12.9.86, gcc 13.3 |
| Python | 3.11 (uv's CPython, venv `/opt/venv`) | uv 0.12.12 |
| gpu4pyscf-cuda12x | 1.9.2 | built from git tag `v1.9.2` (commit `8536a80c`), sm_90 only |
| gpu4pyscf-libxc-cuda12x | 0.9.0 (libxc 7.1.2, CUDA) | built from the same tree (libxc fetched from gitlab.com by CMake) |
| cupy-cuda12x | 14.2.0 | **rebuilt from the sdist** with cuTENSOR (see below), sm_90 |
| cutensor-cu12 | 2.8.1 | PyPI wheel |
| pyscf | 2.15.0 | PyPI wheel |

The full list is `containers/requirements-D_gpu4pyscf-container.txt`. Image:
`/var/tmp/$USER-pah/sif/D_gpu4pyscf.sif`, sha256 `09afb47a44974c8b4575ba18b8664e6eb0f681bd3264c47c858be0d5ae20893b`
In myhub as a public ORAS artifact (same digest): `apptainer pull D_gpu4pyscf.sif oras://myhub.ksc.re.kr/pah/d_gpu4pyscf:gh200-2026-11-14`.
(6.1 GB).

## Why it is built from source

- **No aarch64 GPU4PySCF wheels.** `gpu4pyscf-cuda12x` and `gpu4pyscf-libxc-cuda12x` exist on PyPI only for x86-64, and
  conda-forge has neither (checked 2026-11-14). `D_gpu4pyscf_post.sh` follows upstream's `builder/build_libxc.sh` and
  `setup.py`: one CMake build of `gpu4pyscf/lib` with `BUILD_LIBXC=ON`, then the two wheels are packaged from it, so
  the distribution names match the x86 wheels and `pip check` is clean.
- **The CMake project declares Fortran** (`project(gpu4pyscf C CXX Fortran)`), so `gfortran` is needed even though
  nothing Fortran is compiled for the GPU part.
- **The aarch64 `cupy-cuda12x` 14.2.0 wheel has no cuTENSOR backend** (`cupy_backends/cuda/libs/cutensor` is in the x86
  wheel only), so GPU4PySCF warns "using cupy as the tensor contraction engine" and uses `cupy.einsum`. The recipe
  rebuilds CuPy 14.2.0 from the sdist (sha256 checked) against the `cutensor-cu12` wheel. Two details: the wheel ships
  only `libcutensor*.so.2`, so unversioned links are added for the linker (CuPy's detection silently drops cuTENSOR
  when `-lcutensorMg` fails); and `pyproject.toml`'s name is set to `cupy-cuda12x` so the gpu4pyscf requirement holds.
  Check: `gpu4pyscf.lib.cutensor.contract_engine` is `None` (= cuTENSOR).
- **`import gpu4pyscf` needs a GPU** (it counts devices on import), so the build only checks files; test with `--nv`.
- Build time on the GH200 (48 jobs): about 7 minutes in all (CuPy ~2.5 min, GPU4PySCF + libxc ~3 min).

## Build

```bash
export APPTAINER_CACHEDIR=/var/tmp/$USER-pah/cache/apptainer APPTAINER_TMPDIR=/var/tmp/$USER-pah/tmp
apptainer build --fakeroot /var/tmp/$USER-pah/sif/D_gpu4pyscf.sif containers/D_gpu4pyscf.def
```

The recipe copies `containers/D_gpu4pyscf_post.sh` in and runs it as `%post`; the same script can be run in a writable
sandbox (`apptainer exec --fakeroot --writable <sandbox> bash D_gpu4pyscf_post.sh`) to debug a build step.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 \
  apptainer exec --nv --bind $PWD:/r --env CUPY_CACHE_DIR=/var/tmp/$USER-pah/cache/cupy \
  /var/tmp/$USER-pah/sif/D_gpu4pyscf.sif python /r/d1/pyscf_match.py ...
```

- GPU 0 belongs to cores 0–71 (NUMA node 0), GPU 1 to cores 72–143 (node 1).
- The image's CUDA 12.9 runs on the host driver (595.45, CUDA 13.2) directly, without forward compatibility (cupy
  reports runtime 12090, driver 13020), so a **host** MPS daemon serves its clients, unlike container B. Under MPS a
  client's `CUDA_VISIBLE_DEVICES` indexes the devices the daemon sees (`d1/h100_bench.sh batch_mps`).
- `d1/h100_bench.sh` runs the validation, timing, noise-floor, stability and batching tests.

## Tests (2026-11-14)

Against CPU PySCF 2.15.0 (env `dft`, Grace cores), naphthalene and coronene, neutral and cation, displaced 0.05 Å
(seed 0), production setting (`99x590_t9g6`): see `docs/research/pyscf_gpu_jax.md` §H100.
