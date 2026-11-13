#!/bin/bash
# Container D build steps (GPU4PySCF 1.9.2 for aarch64 + H100). Run as %post of D_gpu4pyscf.def; the same file was
# used to debug the build in a writable sandbox. No aarch64 wheels of gpu4pyscf-cuda12x / gpu4pyscf-libxc-cuda12x
# exist (PyPI, conda-forge; checked 2026-11-14), so both are built from the v1.9.2 tag, the way upstream's
# builder/build_libxc.sh and setup.py do it, for sm_90 only. One CMake build compiles everything (libxc 7.1.2 as
# an ExternalProject); the two wheels are then packaged from it, so the package names match the x86 wheels.
set -ex
export DEBIAN_FRONTEND=noninteractive
NJOBS=${NJOBS:-32}
GPU4PYSCF_TAG=v1.9.2
GPU4PYSCF_COMMIT=8536a80c1e03260aab030e108ef1d181a3709057

apt-get update && apt-get install -y --no-install-recommends git ca-certificates curl patch cmake make gfortran
curl -LsSf https://astral.sh/uv/0.12.12/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
export UV_PYTHON_INSTALL_DIR=/opt/python
uv python install 3.11
rm -rf /opt/venv /opt/src /opt/wheels
uv venv --seed -p 3.11 /opt/venv && . /opt/venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "setuptools<81" wheel
# Versions validated on wasp33a (docs/research/pyscf_gpu_jax.md, Setup)
python -m pip install numpy pyscf==2.15.0 cupy-cuda12x==14.2.0 cutensor-cu12==2.8.1 geometric packaging ase
export CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH

# CuPy: the aarch64 cupy-cuda12x 14.2.0 wheel has no cuTENSOR backend (cupy_backends/cuda/libs/cutensor; the x86
# wheel has it), so GPU4PySCF would fall back to cupy.einsum. Rebuild the same version from the sdist against the
# cutensor-cu12 wheel, keeping the distribution name cupy-cuda12x (gpu4pyscf-cuda12x requires it).
CT=/opt/venv/lib/python3.11/site-packages/cutensor
for l in cutensor cutensorMg cutensorMp; do ln -sf lib$l.so.2 $CT/lib/lib$l.so; done   # the wheel has only .so.2
mkdir -p /opt/src/cupy /opt/wheels && cd /opt/src/cupy
curl -sLO https://files.pythonhosted.org/packages/dc/10/059a92d40b161ea24607c9e41a75098812ca7021f2b565524f87beba15c9/cupy-14.2.0.tar.gz
echo "cfed725e612178cfbdbf62f4a3a598b60edecbbca718c0d4580c21a16ae4a62c  cupy-14.2.0.tar.gz" | sha256sum -c
tar xzf cupy-14.2.0.tar.gz && cd cupy-14.2.0
sed -i 's/^name = "cupy"$/name = "cupy-cuda12x"/' pyproject.toml && grep -q '^name = "cupy-cuda12x"' pyproject.toml
CUTENSOR_PATH=$CT CUPY_NUM_BUILD_JOBS=${NJOBS} CUPY_NVCC_GENERATE_CODE="arch=compute_90,code=sm_90" \
  python -m pip wheel --no-deps -w /opt/wheels .
python -m pip uninstall -y cupy-cuda12x && python -m pip install --no-deps /opt/wheels/cupy_cuda12x-14.2.0-*.whl
cd / && python -c "import cupy_backends.cuda.libs.cutensor"      # fails if the cuTENSOR backend was skipped
rm -rf /opt/src/cupy

mkdir -p /opt/src && cd /opt/src
git clone --depth 1 --branch ${GPU4PYSCF_TAG} https://github.com/pyscf/gpu4pyscf.git
cd gpu4pyscf
test "$(git rev-parse HEAD)" = "${GPU4PYSCF_COMMIT}"

# 1. Compile everything, libxc included, into the build dir setup.py uses (build/temp.<plat>/gpu4pyscf), so its
#    own cmake pass below finds nothing to rebuild.
PLAT=$(python -c "from setuptools._distutils.util import get_platform; print(get_platform())")
BDIR=build/temp.${PLAT}/gpu4pyscf
cmake -S gpu4pyscf/lib -B ${BDIR} -DCUDA_ARCHITECTURES=90-real -DBUILD_LIBXC=ON -DCMAKE_MESSAGE_LOG_LEVEL=WARNING
cmake --build ${BDIR} -j ${NJOBS}

# 2. gpu4pyscf-libxc-cuda12x wheel (as builder/build_libxc.sh, without auditwheel)
mkdir -p /opt/src/libxc_whl/gpu4pyscf/lib/deps/lib
cp gpu4pyscf/lib/deps/lib/libxc.so /opt/src/libxc_whl/gpu4pyscf/lib/deps/lib/
(cd /opt/src/libxc_whl && GPU4PYSCF_BUILD_CUDA_WHEEL=1 python /opt/src/gpu4pyscf/builder/setup_libxc.py bdist_wheel)
rm -rf gpu4pyscf/lib/deps

# 3. gpu4pyscf-cuda12x wheel (setup.py re-runs cmake with BUILD_LIBXC=OFF; the objects are up to date)
GPU4PYSCF_BUILD_CUDA_WHEEL=1 CMAKE_CONFIGURE_ARGS="-DCUDA_ARCHITECTURES=90-real" CMAKE_BUILD_ARGS="-j ${NJOBS}" \
  python setup.py bdist_wheel
cp /opt/src/libxc_whl/dist/*.whl dist/*.whl /opt/wheels/
python -m pip install --no-deps /opt/wheels/*.whl
python -m pip check

# import gpu4pyscf needs a GPU (it counts devices on import), which a build has not: check the files instead
cd / && python -c "import importlib.util as u; [print(m, u.find_spec(m).origin) for m in ('gpu4pyscf', 'pyscf', 'cupy', 'cupy_backends.cuda.libs.cutensor')]"
ls /opt/venv/lib/python3.11/site-packages/gpu4pyscf/lib/deps/lib/libxc.so /opt/venv/lib/python3.11/site-packages/gpu4pyscf/lib/libgdft.so
rm -rf /opt/src/gpu4pyscf/build /root/.cache
python -m pip list --format=freeze > /opt/requirements-frozen.txt
