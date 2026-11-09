## Container A (`A_jax.sif`)

151 distributions. **R** marks those the recipe names.

| Package | Version | | Summary (from the package) | Required by |
| --- | --- | --- | --- | --- |
| absl-py | 2.5.1 |  | Abseil Python Common Libraries, see https://github.com/abseil/abseil-py. | chex, jax-md, ml-collections, optax, orbax-checkpoint |
| aiofiles | 25.1.0 |  | File support for asyncio. | orbax-checkpoint |
| aiohappyeyeballs | 2.7.1 |  | Happy Eyeballs for asyncio | aiohttp |
| aiohttp | 3.14.5 |  | Async http client/server framework (asyncio) | gcsfs |
| aiosignal | 1.4.0 |  | aiosignal: a list of registered asynchronous callbacks | aiohttp |
| ase | 3.29.0 | R | Atomic Simulation Environment | mace-torch, mace-jax, matscipy |
| attrs | 26.1.0 |  | Classes Without Boilerplate | aiohttp, e3nn-jax |
| bibtexparser | 1.4.4 |  | Bibtex parser for python 3 | pymatgen-core |
| certifi | 2026.10.23 |  | Python package for providing Mozilla's CA Bundle. | requests |
| cffi | 2.1.1 |  | Foreign Function Interface for Python calling C code. | cryptography |
| charset-normalizer | 3.5.2 |  | The Real First Universal Charset Detector. Open, modern and actively maintained alternative to Chardet. | requests |
| chex | 0.1.92 |  | Chex: Testing made fun, in JAX! | mace-jax |
| cloudpickle | 3.1.2 |  | Pickler class to extend the standard pickle.Pickler functionality | joblib |
| ConfigArgParse | 1.9.0 |  | A drop-in replacement for argparse that allows options to also be set via config files and/or environment vari | mace-torch, mace-jax |
| contourpy | 1.4.0 |  | Python library for calculating contours of 2D quadrilateral grids | matplotlib |
| cryptography | 50.0.2 |  | cryptography is a package which provides cryptographic recipes and primitives to Python developers. | google-auth |
| cuequivariance | 0.10.0 |  | CUDA accelerated equivariant operations | cuequivariance-jax, mace-jax |
| cuequivariance-jax | 0.10.0 |  | CUDA accelerated equivariant operations | mace-jax |
| cuequivariance-ops-cu12 | 0.10.0 |  | cuequivariance-ops - GPU Accelerated Extensions for Equivariant Primitives | cuequivariance-ops-jax-cu12 |
| cuequivariance-ops-jax-cu12 | 0.10.0 |  | cuequivariance-ops-jax - GPU Accelerated JAX Extensions for Equivariant Primitives |  |
| cycler | 0.12.1 |  | Composable style cycles | matplotlib |
| decorator | 5.3.1 |  | Decorators for Humans | gcsfs |
| e3nn | 0.4.4 |  | Equivariant convolutional neural networks for the group E(3) of 3 dimensional rotations, translations, and mir | mace-torch, mace-jax |
| e3nn-jax | 0.21.0 |  | Equivariant convolutional neural networks for the group E(3) of 3 dimensional rotations, translations, and mir | jax-md, mace-jax |
| einops | 0.8.2 |  | A new flavour of deep learning operations | cuequivariance-jax, jax-md |
| etils | 1.15.0 |  | Collection of common python utils | orbax-checkpoint |
| filelock | 4.2.4 |  | A platform independent file lock. | torch |
| flax | 0.12.12 | R | Flax: A neural network library for JAX designed for flexibility | jax-md, mace-jax |
| fonttools | 4.68.0 |  | Tools to manipulate font files | matplotlib |
| frozendict | 2.5.1 |  | A simple immutable dictionary | jax-md |
| frozenlist | 1.8.0 |  | A list-like structure which implements collections.abc.MutableSequence | aiohttp, aiosignal |
| fsspec | 2026.9.0 |  | File-system specification | gcsfs, torch |
| gcsfs | 2026.10.1 |  | Convenient Filesystem interface over GCS |  |
| gin-config | 0.5.0 |  | Gin-Config: A lightweight configuration library for Python | mace-jax |
| gitdb | 4.0.12 |  | Git Object Database | gitpython |
| GitPython | 3.2.1 |  | GitPython is a Python library used to interact with Git repositories | mace-torch |
| google-api-core | 2.44.0 |  | Google API client core library | google-cloud-core, google-cloud-storage, google-cloud-storage-control |
| google-auth | 2.62.1 |  | Google Authentication Library | gcsfs, google-api-core, google-auth-oauthlib, google-cloud-core, google-cloud-storage, google-cloud-storage-control |
| google-auth-oauthlib | 1.5.0 |  | Google Authentication Library | gcsfs |
| google-cloud-core | 2.8.0 |  | Google Cloud API client core library | google-cloud-storage |
| google-cloud-storage | 3.17.1 |  | Google Cloud Storage API client library | gcsfs |
| google-cloud-storage-control | 1.17.0 |  | Google Cloud Storage Control API client library | gcsfs |
| google-crc32c | 1.9.0 |  | A python wrapper of the C library 'Google CRC32C' | google-cloud-storage, google-resumable-media |
| google-resumable-media | 2.11.0 |  | Utilities for Google Media Downloads and Resumable Uploads | google-cloud-storage |
| googleapis-common-protos | 1.76.0 |  | Common protobufs used in Google APIs | google-api-core, grpc-google-iam-v1, grpcio-status |
| grpc-google-iam-v1 | 0.14.6 |  | IAM API client library | google-cloud-storage-control |
| grpcio | 1.85.1 |  | HTTP/2-based RPC framework | google-cloud-storage-control, grpc-google-iam-v1, grpcio-status |
| grpcio-status | 1.85.1 |  | Status proto mapping for gRPC |  |
| h5py | 3.16.0 | R | Read and write HDF5 files from Python | mace-torch, mace-jax |
| humanize | 4.16.0 |  | Python humanize utilities | orbax-checkpoint |
| idna | 3.20 |  | Internationalized Domain Names in Applications (IDNA) | requests, yarl |
| jax | 0.11.2 |  | Differentiate, compile, and transform Numpy code. | chex, cuequivariance-jax, e3nn-jax, flax, jax-md, jraph … |
| jax-cuda12-pjrt | 0.11.2 |  | JAX XLA PJRT Plugin for NVIDIA GPUs | jax-cuda12-plugin |
| jax-cuda12-plugin | 0.11.2 |  | JAX Plugin for NVIDIA GPUs |  |
| jax-md | 0.2.29 | R | Differentiable, Hardware Accelerated, Molecular Dynamics |  |
| jaxlib | 0.11.2 |  | XLA library for JAX | chex, e3nn-jax, jax, jax-md, jraph, optax |
| Jinja2 | 3.1.6 |  | A very fast and expressive template engine. | torch |
| joblib | 1.6.0 |  | Lightweight pipelining with Python functions | pymatgen-core |
| jraph | 0.0.6.dev0 |  | Jraph: A library for Graph Neural Networks in Jax | jax-md, mace-jax |
| kiwisolver | 1.5.1 |  | A fast implementation of the Cassowary constraint solver | matplotlib |
| lightning-utilities | 0.15.3 |  | Lightning toolbox for across the our ecosystem. | torchmetrics |
| lmdb | 3.0.0 |  | Universal Python binding for the LMDB 'Lightning' Database | mace-torch |
| lxml | 6.1.3 |  | Powerful and Pythonic XML processing library combining libxml2/libxslt with the ElementTree API. | pymatgen-core |
| mace\_jax | 0.2.0 | R |  |  |
| mace-torch | 0.3.17 | R |  |  |
| markdown-it-py | 4.2.0 |  | Python port of markdown-it. Markdown parsing, done right! | rich |
| MarkupSafe | 3.0.3 |  | Safely add untrusted strings to HTML/XML markup. | jinja2 |
| matplotlib | 3.11.2 | R | Python plotting package | ase, mace-torch, pymatgen-core |
| matscipy | 1.3.1 |  | Generic Python Materials Science tools | mace-torch, mace-jax |
| mdurl | 0.1.2 |  | Markdown URL utilities | markdown-it-py |
| ml\_collections | 1.1.0 |  | ML Collections is a library of Python collections designed for ML usecases. | jax-md |
| ml\_dtypes | 0.6.0 |  | ml\_dtypes is a stand-alone implementation of several NumPy dtype extensions used in machine learning. | jax, jaxlib, tensorstore |
| monty | 2026.10.21 |  | Monty is the missing complement to Python. | pymatgen-core |
| mpmath | 1.3.0 |  | Python library for arbitrary-precision floating-point arithmetic | sympy |
| msgpack | 1.2.4 |  | MessagePack serializer | flax, orbax-checkpoint |
| multidict | 7.2.1 |  | multidict implementation | aiohttp, yarl |
| narwhals | 2.28.0 |  | Extremely lightweight compatibility layer between dataframe libraries | plotly |
| networkx | 3.7 |  | Python package for creating and manipulating graphs and networks | cuequivariance, pymatgen-core, torch |
| numpy | 2.5.4 | R | Fundamental package for array computing in Python | ase, chex, contourpy, cuequivariance, e3nn-jax, flax … |
| nvidia-cublas-cu12 | 12.9.3.13 |  | CUBLAS native runtime libraries | cuequivariance-ops-cu12, nvidia-cudnn-cu12, nvidia-cusolver-cu12 |
| nvidia-cuda-cccl-cu12 | 12.9.27 |  | CUDA CCCL | nvidia-nvshmem-cu12 |
| nvidia-cuda-cupti-cu12 | 12.9.79 |  | CUDA profiling tools runtime libs. |  |
| nvidia-cuda-nvcc-cu12 | 12.9.86 |  | CUDA nvcc |  |
| nvidia-cuda-nvrtc-cu12 | 12.9.86 |  | NVRTC native runtime libraries | nvidia-cublas-cu12 |
| nvidia-cuda-runtime-cu12 | 12.9.79 |  | CUDA Runtime native Libraries |  |
| nvidia-cudnn-cu12 | 9.28.0.42 |  | cuDNN runtime libraries |  |
| nvidia-cufft-cu12 | 11.4.1.4 |  | CUFFT native runtime libraries |  |
| nvidia-cusolver-cu12 | 11.7.5.82 |  | CUDA solver native runtime libraries |  |
| nvidia-cusparse-cu12 | 12.5.10.65 |  | CUSPARSE native runtime libraries | nvidia-cusolver-cu12 |
| nvidia-ml-py | 13.615.71 |  | Python Bindings for the NVIDIA Management Library | cuequivariance-ops-cu12 |
| nvidia-nccl-cu12 | 2.32.3 |  | NVIDIA Collective Communication Library (NCCL) Runtime |  |
| nvidia-nvjitlink-cu12 | 12.9.86 |  | Nvidia JIT LTO Library | nvidia-cufft-cu12, nvidia-cusolver-cu12, nvidia-cusparse-cu12 |
| nvidia-nvshmem-cu12 | 3.8.0 |  | NVSHMEM creates a global address space that provides efficient and scalable communication for NVIDIA GPU clust |  |
| oauthlib | 4.0.1 |  | A generic, spec-compliant, thorough implementation of the OAuth request-signing logic | requests-oauthlib |
| opentelemetry-api | 1.45.2 |  | OpenTelemetry Python API | google-api-core |
| opt\_einsum | 3.4.0 |  | Path optimization of einsum functions. | cuequivariance, jax, mace-torch, opt-einsum-fx |
| opt-einsum-fx | 0.1.4 |  | Einsum optimization using opt\_einsum and PyTorch FX | e3nn |
| optax | 0.2.8 | R | A gradient processing and optimization library in JAX. | flax, jax-md, mace-jax |
| orbax-checkpoint | 0.12.9 |  | Orbax Checkpoint | flax |
| orjson | 3.13.0 |  | Fast, correct Python JSON library supporting dataclasses, datetimes, and numpy | mace-torch, pymatgen-core |
| packaging | 26.3 |  | Core utilities for Python packages | cuequivariance-jax, lightning-utilities, matplotlib, matscipy, opt-einsum-fx, plotly … |
| palettable | 3.3.3 |  | Color palettes for Python | pymatgen-core |
| pandas | 3.1.1 |  | Powerful data structures for data analysis, time series, and statistics | mace-torch, pymatgen-core |
| pillow | 13.0.0 |  | Python Imaging Library (fork) | matplotlib |
| pip | 26.3 |  | The PyPA recommended tool for installing Python packages. |  |
| platformdirs | 4.14.5 |  | A small Python package for determining appropriate platform-specific dirs, e.g. a `user data dir`. | cuequivariance-ops-cu12 |
| plotly | 7.1.0 |  | An open-source interactive data visualization library for Python | pymatgen-core |
| prettytable | 3.18.0 |  | A simple Python library for easily displaying tabular data in a visually appealing ASCII table format | mace-torch |
| prometheus\_client | 0.26.0 |  | Python client for the Prometheus monitoring system. | orbax-checkpoint |
| propcache | 0.5.4 |  | Accelerated property cache | aiohttp, yarl |
| proto-plus | 1.29.0 |  | Beautiful, Pythonic protocol buffers | google-api-core, google-cloud-storage-control |
| protobuf | 7.36.2 |  |  | google-api-core, google-cloud-storage-control, googleapis-common-protos, grpc-google-iam-v1, grpcio-status, orbax-checkpoint … |
| psutil | 7.2.2 |  | Cross-platform lib for process and system monitoring. | orbax-checkpoint |
| pyasn1 | 0.6.5 |  | Pure-Python implementation of ASN.1 types and DER/BER/CER codecs (X.208) | pyasn1-modules |
| pyasn1\_modules | 0.4.2 |  | A collection of ASN.1-based protocols modules | google-auth |
| pycparser | 3.12 |  | C parser in Python | cffi |
| Pygments | 2.21.0 |  | Pygments is a syntax highlighting package written in Python. | rich |
| pymatgen | 2026.10.24 |  | Python Materials Genomics is a robust materials analysis code that defines core object representations for str | jax-md |
| pymatgen-core | 2026.10.31 |  | Python Materials Genomics is a robust materials analysis code that defines core object representations for str | pymatgen |
| pyparsing | 3.3.3 |  | pyparsing - Classes and methods to define and execute parsing grammars | bibtexparser, matplotlib |
| python-dateutil | 4.0.0 |  | Extensions to the standard Python datetime module | matplotlib, pandas |
| python\_hostlist | 2.3.0 |  | Python module for hostlist handling | mace-torch |
| PyYAML | 6.0.3 |  | YAML parser and emitter for Python | flax, mace-torch, mace-jax, ml-collections, orbax-checkpoint |
| requests | 2.34.2 |  | Python HTTP for Humans. | gcsfs, google-api-core, google-cloud-storage, pymatgen-core, requests-oauthlib |
| requests-oauthlib | 2.0.0 |  | OAuthlib authentication support for Requests. | google-auth-oauthlib |
| rich | 15.0.0 |  | Render rich text, tables, progress bars, syntax highlighting, markdown and more to the terminal | flax |
| roundmantissa | 0.2.1 |  | Round mantissa of a float number | mace-jax |
| ruamel.yaml | 0.19.1 |  | ruamel.yaml is a YAML parser/emitter that supports roundtrip preservation of comments, seq/map flow style, and | monty |
| scipy | 1.18.1 | R | Fundamental algorithms for scientific computing in Python | ase, cuequivariance, e3nn, jax, jaxlib, matscipy … |
| setuptools | 78.1.0 |  | Easily download, build, install, upgrade, and uninstall Python packages | torch |
| simplejson | 4.2.0 |  | Simple, fast, extensible JSON encoder/decoder for Python | orbax-checkpoint |
| smmap | 5.0.3 |  | A pure Python implementation of a sliding window memory map manager | gitdb |
| spglib | 2.8.0 |  | This is the spglib module. | pymatgen-core |
| sympy | 1.14.0 |  | Computer algebra system (CAS) in Python | cuequivariance, e3nn, e3nn-jax, pymatgen-core, torch |
| tabulate | 0.10.0 |  | Pretty-print tabular data | pymatgen-core |
| tensorstore | 0.1.86 |  | Read and write large, multi-dimensional arrays | flax, orbax-checkpoint |
| toolz | 1.2.0 |  | List processing tools and functional utilities | chex |
| torch | 2.15.0+cpu | R | Tensors and Dynamic neural networks in Python with strong GPU acceleration | e3nn, mace-torch, opt-einsum-fx, torch-ema, torchmetrics |
| torch-ema | 0.3 |  | PyTorch library for computing moving averages of model parameters. | mace-torch |
| torchmetrics | 1.9.0 |  | PyTorch native Metrics | mace-torch |
| tqdm | 4.70.1 |  | Fast, Extensible Progress Meter | cuequivariance-ops-cu12, mace-torch, mace-jax, pymatgen-core |
| treescope | 0.1.10 |  | Treescope: An interactive HTML pretty-printer for ML research in IPython notebooks. | flax |
| typing\_extensions | 4.16.0 |  | Backported and Experimental Type Hints for Python 3.9+ | aiohttp, aiosignal, ase, chex, flax, grpcio … |
| tzdata | 2026.6 |  | Provider of IANA time zone data | python-dateutil |
| uncertainties | 3.2.3 |  | calculations with values with uncertainties, error propagation | pymatgen-core |
| unique-names-generator | 1.0.2 |  | Generate unique and memorable name strings. | mace-jax |
| urllib3 | 2.8.0 |  | HTTP library with thread-safe connection pooling, file post, and more. | requests |
| uvloop | 0.23.0 |  | Fast implementation of asyncio event loop on top of libuv | orbax-checkpoint |
| wcwidth | 0.9.4 |  | Measures the displayed width of unicode strings in a terminal | prettytable |
| yarl | 1.25.1 |  | Yet another URL library | aiohttp |
| zipp | 4.1.2 |  | Backport of pathlib-compatible object wrapper for zip files |  |

## Container B + TorchSim (`B_torchsim.sif`)

**B** marks the 36 distributions our recipes added (`B_torch.def`, `B_torchsim.def`); the others came with the NGC 26.10 image. `B_torch.sif` is the same list without torch-sim-atomistic, nvalchemi-toolkit-ops, warp-lang, tables and their own dependencies.

296 distributions. **R** marks those the recipe names.

| Package | Version | | Summary (from the package) | Required by |
| --- | --- | --- | --- | --- |
| absl-py | 2.5.1 |  | Abseil Python Common Libraries, see https://github.com/abseil/abseil-py. | array-record, grain, tensorboard |
| aiohappyeyeballs | 2.7.1 |  | Happy Eyeballs for asyncio | aiohttp |
| aiohttp | 3.14.4 |  | Async http client/server framework (asyncio) |  |
| aiosignal | 1.4.0 |  | aiosignal: a list of registered asynchronous callbacks | aiohttp |
| annotated-doc | 0.0.5 |  | Document parameters, class attributes, return types, and variables inline, with Annotated. | fastapi |
| annotated-types | 0.8.0 |  | Reusable constraint types to use with typing.Annotated | pydantic |
| anyio | 4.15.1 |  | High-level concurrency and networking framework on top of asyncio or Trio | httpx, jupyter-server, starlette, watchfiles |
| apache-tvm-ffi | 0.1.14.post1 |  | tvm ffi | flash-attn-4, nvidia-cudnn-frontend, nvidia-cutlass-operators, quack-kernels |
| apex | 0.1 |  | PyTorch Extensions written by NVIDIA |  |
| argon2-cffi | 25.1.0 |  | Argon2 for Python | jupyter-server |
| argon2-cffi-bindings | 26.1.0 |  | Low-level CFFI bindings for Argon2 | argon2-cffi |
| array\_record | 0.8.4 |  | A file format that achieves a new frontier of IO efficiency | grain |
| arrow | 1.4.0 |  | Better dates & times for Python | isoduration |
| ase | 3.29.0 | R | Atomic Simulation Environment | mace-torch, matscipy |
| asttokens | 3.0.2 |  | Annotate AST trees with source code positions | stack-data |
| astunparse | 1.6.3 |  | An AST unparser for Python | nvidia-dali-cuda130 |
| async-lru | 2.3.0 |  | Simple LRU cache for asyncio | jupyterlab |
| attn\_gym | 0.0.8 |  | Helpful tools and examples for working with flex-attention | torchtitan |
| attrs | 26.1.0 |  | Classes Without Boilerplate | aiohttp, jsonschema, referencing |
| babel | 2.18.0 |  | Internationalization utilities | jupyterlab-server |
| beautifulsoup4 | 4.15.0 |  | Screen-scraping library | nbconvert |
| black | 26.10.0 |  | The uncompromising code formatter. |  |
| bleach | 6.4.0 |  | An easy safelist-based HTML-sanitizing tool. | nbconvert |
| blosc2 | 4.14.1 | B | A fast & compressed ndarray library with a flexible compute engine. | tables |
| certifi | 2026.7.22 |  | Python package for providing Mozilla's CA Bundle. | httpcore, httpx, requests |
| cffi | 2.1.1 |  | Foreign Function Interface for Python calling C code. | argon2-cffi-bindings |
| charset-normalizer | 3.5.2 |  | The Real First Universal Charset Detector. Open, modern and actively maintained alternative to Chardet. | requests |
| click | 8.5.0 |  | Composable command line interface toolkit | black, huggingface-hub, uvicorn, wandb |
| cloudpickle | 3.1.2 |  | Pickler class to extend the standard pickle.Pickler functionality | grain |
| cmake | 3.31.6 |  | CMake is an open-source, cross-platform family of tools designed to build, test and package software |  |
| comm | 0.2.3 |  | Jupyter Python Comm implementation, for usage in ipykernel, xeus-python etc. | ipykernel |
| ConfigArgParse | 1.9.0 | B | A drop-in replacement for argparse that allows options to also be set via config files and/or environment vari | mace-torch |
| contourpy | 1.4.0 |  | Python library for calculating contours of 2D quadrilateral grids | matplotlib |
| cuda-bindings | 13.4.3 |  | Python bindings for CUDA | cupti-python, nvidia-cutlass-dsl-libs-base, nvidia-cutlass-dsl-libs-core, nvidia-cutlass-dsl-libs-cu12, nvidia-cutlass-dsl-libs-cu13 |
| cuda-pathfinder | 1.8.3 |  | Pathfinder for CUDA components | cuda-bindings, cupti-python |
| cuda-tile | 1.6.0 |  | CUDA Tile Compiler |  |
| cuequivariance | 0.10.0 | R | CUDA accelerated equivariant operations | cuequivariance-torch |
| cuequivariance-ops-cu13 | 0.10.0 | B | cuequivariance-ops - GPU Accelerated Extensions for Equivariant Primitives | cuequivariance-ops-torch-cu13 |
| cuequivariance-ops-torch-cu13 | 0.10.0 | R | cuequivariance-ops-torch - GPU Accelerated Torch Extensions for Equivariant Primitives |  |
| cuequivariance-torch | 0.10.0 | R | CUDA accelerated equivariant operations |  |
| cupti-python | 13.4.0 |  | NVIDIA CUPTI Python Library |  |
| cycler | 0.12.1 |  | Composable style cycles | matplotlib |
| Cython | 3.3.0 |  | The Cython compiler for writing C extensions in the Python language. | pycocotools |
| datasets | 4.7.0 |  | HuggingFace community-driven open-source library of datasets | torchtitan |
| debugpy | 1.8.22 |  | An implementation of the Debug Adapter Protocol for Python | ipykernel |
| defusedxml | 0.7.1 |  | XML bomb protection for Python stdlib modules | nbconvert, nvidia-resiliency-ext |
| Deprecated | 3.0.0 |  | Python @deprecated decorator to deprecate old python classes, functions or methods. | limits |
| dill | 0.4.0 |  | serialize all of Python | datasets, multiprocess |
| dllist | 2.0.0 |  | List the shared libraries loaded by the current process. | torch-tensorrt |
| docstring\_parser | 0.18.0 |  | Parse Python docstrings in reST, Google and Numpydoc format | tyro |
| dragon4 |  |  |  |  |
| e3nn | 0.4.4 | B | Equivariant convolutional neural networks for the group E(3) of 3 dimensional rotations, translations, and mir | mace-torch |
| einops | 0.8.2 |  | A new flavour of deep learning operations | flash-attn-4, flash-attn, quack-kernels, torchtitan, transformer-engine |
| etils | 1.14.0 |  | Collection of common python utils | array-record, grain |
| executing | 2.2.1 |  | Get the currently executing AST node of a frame, and other information | stack-data |
| fastapi | 0.143.0 |  | FastAPI framework, high performance, easy to learn, fast to code, ready for production | nvidia-resiliency-ext |
| fastjsonschema | 2.22.2 |  | Fastest Python implementation of JSON schema | nbformat |
| filelock | 4.0.12 |  | A platform independent file lock. | datasets, huggingface-hub, torch, torch-tensorrt |
| flash\_attn | 2.8.3.post1+git91ffc2ac7 |  | Flash Attention: Fast and Memory-Efficient Exact Attention |  |
| flash-attn-4 | 4.0.0b24 |  | Flash Attention CUTE (CUDA Template Engine) implementation |  |
| fonttools | 4.66.1 |  | Tools to manipulate font files | matplotlib |
| fqdn | 1.6.0 |  | Validates fully-qualified domain names against RFC 1123, so that they are acceptable to modern browsers |  |
| frozenlist | 1.8.0 |  | A list-like structure which implements collections.abc.MutableSequence | aiohttp, aiosignal |
| fsspec | 2026.2.0 |  | File-system specification | datasets, huggingface-hub, torch |
| gast | 0.7.0 |  | Python AST that abstracts the underlying Python version | nvidia-dali-cuda130 |
| GCC runtime library |  |  |  |  |
| gitdb | 4.0.12 | B | Git Object Database | gitpython |
| GitPython | 3.2.1 | B | GitPython is a Python library used to interact with Git repositories | mace-torch |
| googleapis-common-protos | 1.75.5 |  | Common protobufs used in Google APIs | opentelemetry-exporter-otlp-proto-http |
| grain | 0.2.18 |  | Grain: A library for loading and transforming data for ML training. | torchtitan |
| grpcio | 1.84.0 |  | HTTP/2-based RPC framework | grpcio-tools, nvidia-resiliency-ext, tensorboard |
| grpcio-tools | 1.81.1 |  | Protobuf code generator for gRPC | nvidia-resiliency-ext |
| h11 | 0.16.0 |  | A pure-Python, bring-your-own-I/O implementation of HTTP/1.1 | httpcore, uvicorn |
| h2 | 4.4.1 | B | Pure-Python HTTP/2 protocol implementation |  |
| h5py | 3.16.0 | R | Read and write HDF5 files from Python | mace-torch, torch-sim-atomistic |
| hf-xet | 1.7.0 |  | Fast transfer of large files with the Hugging Face Hub. | huggingface-hub |
| hpack | 4.2.0 | B | Pure-Python HPACK header encoding | h2 |
| httpcore | 1.0.9 |  | A minimal low-level HTTP client. | httpx |
| httptools | 0.8.0 |  | A collection of framework independent HTTP protocol utils. |  |
| httpx | 0.28.1 |  | The next generation HTTP client. | blosc2, datasets, huggingface-hub, jupyterlab, nvidia-resiliency-ext |
| huggingface\_hub | 1.33.0 |  | Client library to download and publish models, datasets and other repos on the huggingface.co hub | datasets, tokenizers |
| hyperframe | 6.1.0 | B | Pure-Python HTTP/2 framing | h2 |
| idna | 3.20 |  | Internationalized Domain Names in Applications (IDNA) | anyio, httpx, requests, yarl |
| importlib\_metadata | 9.0.1 |  | Read metadata from Python packages | transformer-engine |
| ipykernel | 7.4.0 |  | IPython Kernel for Jupyter | jupyterlab |
| ipython | 9.17.1 |  | IPython: Productive Interactive Computing | ipykernel |
| ipython\_pygments\_lexers | 1.1.1 |  | Defines a variety of Pygments lexers for highlighting IPython code. | ipython |
| isoduration | 20.11.0 |  | Operations with ISO 8601 durations |  |
| isort | 9.0.2 |  | A Python utility / library to sort Python imports. |  |
| jedi | 0.20.0 |  | An autocompletion tool for Python that can be used for text editors. | ipython |
| Jinja2 | 3.1.6 |  | A very fast and expressive template engine. | jupyter-server, jupyterlab, jupyterlab-server, nbconvert, torch |
| json5 | 0.16.0 |  | A Python implementation of the JSON5 data format. | jupyterlab-server |
| jsonpointer | 3.2.0 |  | Identify specific nodes in a JSON document (RFC 6901)  |  |
| jsonschema | 4.26.0 |  | An implementation of JSON Schema validation for Python | jupyter-events, jupyterlab-server, nbformat |
| jsonschema-specifications | 2025.9.1 |  | The JSON Schema meta-schemas and vocabularies, exposed as a Registry | jsonschema |
| jupyter\_builder | 1.2.2 |  | JupyterLab build tools | jupyterlab, notebook |
| jupyter\_client | 8.10.0 |  | Jupyter protocol implementation and client libraries | ipykernel, jupyter-server, nbclient |
| jupyter\_core | 5.9.1 |  | Jupyter core package. A base package on which Jupyter projects rely. | ipykernel, jupyter-builder, jupyter-client, jupyter-server, jupyterlab, nbclient … |
| jupyter-events | 0.12.1 |  | Jupyter Event System library | jupyter-server |
| jupyter-lsp | 2.3.1 |  | Multi-Language Server WebSocket proxy for Jupyter Notebook/Lab server | jupyterlab |
| jupyter\_server | 2.21.1 |  | The backend—i.e. core services, APIs, and REST endpoints—to Jupyter web applications. | jupyter-lsp, jupyterlab, jupyterlab-code-formatter, jupyterlab-server, notebook, notebook-shim |
| jupyter\_server\_terminals | 0.5.4 |  | A Jupyter Server Extension Providing Terminals. | jupyter-server |
| jupyterlab | 4.6.4 |  | JupyterLab computational environment | jupyterlab-tensorboard-pro, notebook |
| jupyterlab\_code\_formatter | 3.1.0 |  | A JupyterLab plugin to facilitate invocation of code formatters. |  |
| jupyterlab\_pygments | 0.3.0 |  | Pygments theme using JupyterLab CSS variables | nbconvert |
| jupyterlab\_server | 2.28.1 |  | A set of server components for JupyterLab and JupyterLab like applications. | jupyterlab, notebook |
| jupyterlab\_tensorboard\_pro | 4.0.0 |  | A JupyterLab extension for tensorboard. |  |
| jupytext | 1.19.6 |  | Jupyter notebooks as Markdown documents, Julia, Python or R scripts |  |
| kiwisolver | 1.5.1 |  | A fast implementation of the Cassowary constraint solver | matplotlib |
| LAPACK |  |  |  |  |
| lapack-lite |  |  |  |  |
| lark | 1.3.1 |  | a modern parsing library | rfc3987-syntax |
| libcst | 1.9.0 |  | A concrete syntax tree with AST-like properties for Python 3.0 through 3.15 programs. | nvidia-dali-cuda130 |
| libdivide |  |  |  |  |
| libquadmath |  |  |  | ase, blosc2, contourpy, cuequivariance, cupti-python, datasets … |
| lightning-utilities | 0.15.3 | B | Lightning toolbox for across the our ecosystem. | torchmetrics |
| limits | 5.8.0 |  | Rate limiting utilities | slowapi |
| llvmlite | 0.46.0 |  | lightweight wrapper around basic LLVM functionality | numba |
| lmdb | 3.0.0 | B | Universal Python binding for the LMDB 'Lightning' Database | mace-torch |
| mace-torch | 0.3.17 | R |  |  |
| makefun | 1.16.0 |  | Small library to dynamically create python functions. | nvidia-dali-cuda130 |
| Markdown | 3.11 |  | Python implementation of John Gruber's Markdown. | tensorboard |
| markdown-it-py | 4.2.0 |  | Python port of markdown-it. Markdown parsing, done right! | jupytext, mdit-py-plugins, rich |
| MarkupSafe | 3.0.4 |  | Safely add untrusted strings to HTML/XML markup. | jinja2, nbconvert, werkzeug |
| matplotlib | 3.11.2 |  | Python plotting package | ase, mace-torch, pycocotools |
| matplotlib-inline | 0.2.2 |  | Inline Matplotlib backend for Jupyter | ipykernel, ipython |
| matscipy | 1.3.1 | B | Generic Python Materials Science tools | mace-torch |
| mdit-py-plugins | 0.6.1 |  | Collection of plugins for markdown-it-py | jupytext |
| mdurl | 0.1.2 |  | Markdown URL utilities | markdown-it-py |
| Meson |  |  |  |  |
| mistune | 3.3.4 |  | A sane and fast Markdown parser with useful plugins and renderers | nbconvert |
| ml\_dtypes | 0.6.0 |  | ml\_dtypes is a stand-alone implementation of several NumPy dtype extensions used in machine learning. | onnx, onnx-ir, onnxscript |
| mpmath | 1.3.0 |  | Python library for arbitrary-precision floating-point arithmetic | sympy |
| msgpack | 1.2.4 | B | MessagePack serializer | blosc2 |
| multidict | 6.9.1 |  | multidict implementation | aiohttp, yarl |
| multiprocess | 0.70.18 |  | better multiprocessing and multithreading in Python | datasets |
| mypy\_extensions | 1.1.0 |  | Type system extensions for programs checked with the mypy type checker. | black, isort |
| nbclient | 0.11.0 |  | A client library for executing notebooks. Formerly nbconvert's ExecutePreprocessor. | nbconvert |
| nbconvert | 7.17.1 |  | Convert Jupyter Notebooks (.ipynb files) to other formats. | jupyter-server |
| nbformat | 5.11.1 |  | The Jupyter Notebook format | jupyter-server, jupytext, nbclient, nbconvert |
| ndindex | 1.10.1 | B | A Python library for manipulating indices of ndarrays. | blosc2 |
| nest-asyncio2 | 1.7.4 |  | Patch asyncio to allow nested event loops | ipykernel |
| networkx | 3.7 |  | Python package for creating and manipulating graphs and networks | cuequivariance, nvidia-cutlass-operators, torch |
| ninja | 1.13.2 |  | Ninja is a small build system with a focus on speed | nvidia-modelopt |
| notebook | 7.6.3 |  | Jupyter Notebook - A web-based notebook environment for interactive computing |  |
| notebook\_shim | 0.2.4 |  | A shim layer for notebook traits and config | jupyterlab, notebook |
| numba | 0.64.0 |  | compiling Python code using LLVM |  |
| numexpr | 2.14.2 | B | Fast numerical expression evaluator for NumPy | blosc2, tables |
| numpy | 2.1.0 |  | Fundamental package for array computing in Python |  |
| nvalchemi-toolkit-ops | 0.5.0 | B | High-performance NVIDIA Warp primitives for GPU-enabled computational chemistry and atomistic simulation workf | torch-sim-atomistic |
| nvdlfw\_inspect | 0.2.2 |  | Facilitates debugging convergence issues and testing new algorithms/recipes for training LLMs using Nvidia lib | transformer-engine |
| nvidia-cublas | 13.10.0.20 | B | CUBLAS native runtime libraries | cuequivariance-ops-cu13 |
| nvidia-cuda-nvrtc | 13.5.37 | B | NVRTC native runtime libraries | nvidia-cublas |
| nvidia-cudnn-frontend | 1.30.0 |  | NVIDIA cuDNN Frontend — Python and C++ Graph API with SOTA attention (SDPA / Flash Attention), MoE grouped GEM | transformer-engine |
| nvidia-cutlass-dsl | 4.8.0 |  | NVIDIA CUTLASS Python DSL | flash-attn-4, nvidia-cudnn-frontend, nvidia-cutlass-operators, quack-kernels |
| nvidia-cutlass-dsl-libs-base | 4.8.0 |  | NVIDIA CUTLASS Python DSL | nvidia-cutlass-dsl, nvidia-cutlass-dsl-libs-cu12, nvidia-cutlass-dsl-libs-cu13 |
| nvidia-cutlass-dsl-libs-core | 4.8.0 |  | NVIDIA CUTLASS Python DSL | nvidia-cutlass-dsl-libs-base |
| nvidia-cutlass-dsl-libs-cu12 | 4.8.0 |  | NVIDIA CUTLASS Python DSL | nvidia-cutlass-dsl |
| nvidia-cutlass-dsl-libs-cu13 | 4.8.0 |  | NVIDIA CUTLASS Python DSL |  |
| nvidia-cutlass-operators | 0.3.0 |  | High-level discover/compile/execute API for CUTLASS Python kernels. |  |
| nvidia-dali-cuda130 | 2.3.0 |  | NVIDIA DALI  for CUDA 13.0. Git SHA: 68b40ed8bd2f77985c4cf114e50fcca065cb66d7 |  |
| nvidia-libnvcomp-cu13 | 5.3.0.16 |  | NVIDIA nvcomp for CUDA 13 | nvidia-dali-cuda130 |
| nvidia-ml-py | 13.615.71 |  | Python Bindings for the NVIDIA Management Library | cuequivariance-ops-cu13, nvidia-modelopt, nvidia-resiliency-ext |
| nvidia-modelopt | 0.47.0 |  | Nvidia Model Optimizer: A unified library of SOTA model optimization techniques like quantization, pruning, Ne |  |
| nvidia-nvimgcodec-cu13 | 0.9.0.20 |  | NVIDIA nvimgcodec  for CUDA 13. | nvidia-dali-cuda130 |
| nvidia-nvjpeg | 13.2.3.58 |  | NVJPEG native runtime libraries |  |
| nvidia-nvjpeg2k-cu13 | 0.11.0.51 |  | NVIDIA nvJPEG2000 native runtime libraries |  |
| nvidia-nvtiff-cu13 | 0.8.0.82 |  | NVIDIA nvTIFF native runtime libraries |  |
| nvidia-resiliency-ext | 0.7.0 |  | NVIDIA Resiliency Package |  |
| nvtx | 0.2.16 |  | Python NVTX - Python code annotation library | nvidia-dali-cuda130 |
| omegaconf | 2.4.0 |  | A flexible configuration library | nvidia-modelopt |
| onnx | 1.21.0 |  | Open Neural Network Exchange | onnx-ir, onnxscript, transformer-engine |
| onnx-ir | 1.0.0 |  | Efficient in-memory representation for ONNX | onnxscript |
| onnxscript | 0.7.2 |  | Naturally author ONNX functions and models using a subset of Python | transformer-engine |
| OpenBLAS |  |  |  |  |
| opentelemetry-api | 1.45.1 |  | OpenTelemetry Python API | fastapi, opentelemetry-exporter-http-transport, opentelemetry-exporter-otlp-proto-http, opentelemetry-sdk, opentelemetry-semantic-conventions, wandb |
| opentelemetry-exporter-http-transport | 0.66b1 |  | OpenTelemetry Exporters HTTP transport | opentelemetry-exporter-otlp-proto-http |
| opentelemetry-exporter-otlp-common | 0.66b1 |  | OpenTelemetry OTLP HTTP export utilities | opentelemetry-exporter-otlp-proto-http |
| opentelemetry-exporter-otlp-proto-common | 1.45.1 |  | OpenTelemetry Protobuf encoding | opentelemetry-exporter-otlp-proto-http |
| opentelemetry-exporter-otlp-proto-http | 1.45.1 |  | OpenTelemetry Collector Protobuf over HTTP Exporter | wandb |
| opentelemetry-proto | 1.45.1 |  | OpenTelemetry Python Proto | opentelemetry-exporter-otlp-proto-common, opentelemetry-exporter-otlp-proto-http |
| opentelemetry-sdk | 1.45.1 |  | OpenTelemetry Python SDK | opentelemetry-exporter-otlp-common, opentelemetry-exporter-otlp-proto-http, wandb |
| opentelemetry-semantic-conventions | 0.66b1 |  | OpenTelemetry Semantic Conventions | opentelemetry-sdk |
| opt\_einsum | 3.4.0 | B | Path optimization of einsum functions. | cuequivariance, mace-torch, opt-einsum-fx |
| opt-einsum-fx | 0.1.4 | B | Einsum optimization using opt\_einsum and PyTorch FX | e3nn |
| optree | 0.20.0 |  | Optimized PyTree Utilities. | nvidia-dali-cuda130 |
| orjson | 3.13.0 | B | Fast, correct Python JSON library supporting dataclasses, datetimes, and numpy | mace-torch |
| packaging | 26.2 |  | Core utilities for Python packages | apex, black, datasets, huggingface-hub, ipykernel, jupyter-events … |
| pandas | 3.0.6 |  | Powerful data structures for data analysis, time series, and statistics | datasets, mace-torch |
| pandocfilters | 1.5.1 |  | Utilities for writing pandoc filters in python | nbconvert |
| parso | 0.8.7 |  | A Python Parser | jedi |
| pathspec | 1.1.1 |  | Utility library for gitignore style pattern matching of file paths. | black |
| pexpect | 4.9.0 |  | Pexpect allows easy control of interactive console applications. | ipython |
| pillow | 12.3.0 |  | Python Imaging Library (fork) | matplotlib, tensorboard, torchtitan, torchvision |
| pip | 26.2.1 |  | The PyPA recommended tool for installing Python packages. |  |
| platformdirs | 4.12.4 |  | A small Python package for determining appropriate platform-specific dirs, e.g. a `user data dir`. | black, cuequivariance-ops-cu13, jupyter-core, wandb |
| polygraphy | 0.53.5 |  | Polygraphy: A Deep Learning Inference Prototyping and Debugging Toolkit |  |
| portpicker | 1.6.0 |  | A library to choose unique available network ports. | grain |
| prettytable | 3.18.0 | B | A simple Python library for easily displaying tabular data in a visually appealing ASCII table format | mace-torch |
| prometheus\_client | 0.26.0 |  | Python client for the Prometheus monitoring system. | jupyter-server |
| prompt\_toolkit | 3.0.53 |  | Library for building powerful interactive command lines in Python | ipython |
| propcache | 0.5.4 |  | Accelerated property cache | aiohttp, yarl |
| protobuf | 6.33.6 |  |  | googleapis-common-protos, grain, grpcio-tools, nvidia-cutlass-dsl-libs-base, nvidia-cutlass-dsl-libs-core, nvidia-cutlass-dsl-libs-cu12 … |
| psutil | 7.2.2 |  | Cross-platform lib for process and system monitoring. | ipython, nvidia-resiliency-ext, portpicker, torch-tensorrt |
| ptyprocess | 0.7.0 |  | Run a subprocess in a pseudo terminal | pexpect, terminado |
| PuLP | 3.3.2 |  | PuLP is an LP modeler written in python. PuLP can generate MPS or LP files and call GLPK, COIN CLP/CBC, CPLEX, | nvidia-modelopt |
| pure\_eval | 0.2.4 |  | Safely evaluate AST nodes without side effects | stack-data |
| py-cpuinfo | 9.0.0 | B | Get CPU info with pure Python | tables |
| pyarrow | 26.0.0 |  | Python library for Apache Arrow | datasets |
| pybind11 | 3.1.0 |  | Seamless operability between C++11 and Python |  |
| pycocotools | 2.0+nv0.8.1 |  |  |  |
| pycparser | 3.11 |  | C parser in Python | cffi |
| pydantic | 2.14.0 |  | Data validation using Python type hints | blosc2, fastapi, nvidia-modelopt, nvidia-resiliency-ext, pydantic-settings, transformer-engine … |
| pydantic\_core | 2.50.0 |  | Core functionality for Pydantic validation and serialization | pydantic |
| pydantic-settings | 2.15.0 |  | Settings management using Pydantic | nvidia-resiliency-ext |
| pyelftools | 0.33 |  | Library for analyzing ELF files and DWARF debugging information | cupti-python |
| Pygments | 2.17.2 |  | Pygments is a syntax highlighting package written in Python. | ipython, ipython-pygments-lexers, nbconvert, rich |
| pyparsing | 3.3.3 |  | pyparsing - Classes and methods to define and execute parsing grammars | matplotlib |
| python-dateutil | 2.9.0.post0 |  | Extensions to the standard Python datetime module | arrow, jupyter-client, matplotlib, pandas |
| python-dotenv | 1.2.4 |  | Read key-value pairs from a .env file and set them as environment variables | pydantic-settings |
| python\_hostlist | 2.3.0 |  | Python module for hostlist handling | mace-torch |
| python-json-logger | 4.2.0 |  | JSON Log Formatter for the Python Logging Package | jupyter-events |
| pytokens | 0.4.1 |  | A Fast, spec compliant Python 3.14+ tokenizer that runs on older Pythons. | black |
| PyYAML | 6.0.1 |  | YAML parser and emitter for Python | datasets, huggingface-hub, jupyter-events, jupytext, libcst, mace-torch … |
| pyzmq | 27.2.0 |  | Python bindings for 0MQ | ipykernel, jupyter-client, jupyter-server |
| quack-kernels | 0.6.2 |  |  | flash-attn-4 |
| referencing | 0.37.0 |  | JSON Referencing + Python | jsonschema, jsonschema-specifications, jupyter-events |
| regex | 2026.9.29 |  | Alternative regular expression module, to replace re. | nvidia-modelopt |
| requests | 2.34.2 |  | Python HTTP for Humans. | datasets, jupyterlab-server, opentelemetry-exporter-otlp-proto-http, wandb |
| rfc3339-validator | 0.1.4 |  | A pure python RFC3339 validator | jupyter-events |
| rfc3986-validator | 0.1.1 |  | Pure python rfc3986 validator | jupyter-events |
| rfc3987-syntax | 1.1.0 |  | Helper functions to syntactically validate strings according to RFC 3987. |  |
| rich | 15.0.0 |  | Render rich text, tables, progress bars, syntax highlighting, markdown and more to the terminal | blosc2, nvidia-modelopt |
| rpds-py | 2026.9.1 |  | Python bindings to Rust's persistent data structures (rpds) | jsonschema, referencing |
| safetensors | 0.8.0 |  |  | nvidia-modelopt, torch-checkpointing, torchtitan |
| scipy | 1.18.1 |  | Fundamental algorithms for scientific computing in Python | ase, cuequivariance, cuequivariance-ops-torch-cu13, e3nn, matscipy, nvidia-modelopt |
| Send2Trash | 2.1.0 |  | Send file to trash natively under Mac OS X, Windows and Linux | jupyter-server |
| setuptools | 81.0.0 |  | Easily download, build, install, upgrade, and uninstall Python packages | grpcio-tools, nvidia-modelopt, pycocotools, tensorboard, torch |
| six | 1.16.0 |  | Python 2 and 3 compatibility utilities | astunparse, nvidia-dali-cuda130, python-dateutil, rfc3339-validator |
| slowapi | 0.1.10 |  | A rate limiting extension for Starlette and Fastapi | nvidia-resiliency-ext |
| smmap | 5.0.3 | B | A pure Python implementation of a sliding window memory map manager | gitdb |
| soupsieve | 2.10 |  | A modern CSS selector implementation for Beautiful Soup. | beautifulsoup4 |
| spin |  |  |  |  |
| spmd\_types | 0.2.5 |  | A type system for distributed (SPMD) tensor computations in PyTorch | torchtitan |
| stack-data | 0.6.3 |  | Extract data from python stack frames and tracebacks for informative displays | ipython |
| starlette | 1.7.0 |  | The little ASGI library that shines. | fastapi |
| sympy | 1.14.0 |  | Computer algebra system (CAS) in Python | cuequivariance, e3nn, onnx-ir, torch |
| tables | 3.11.1 | B | Hierarchical datasets for Python | torch-sim-atomistic |
| tabulate | 0.10.0 |  | Pretty-print tabular data |  |
| tensorboard | 2.20.0 |  | TensorBoard lets you watch Tensors Flow | torchtitan |
| tensorboard-data-server | 0.7.2 |  | Fast data loading for TensorBoard | tensorboard |
| tensorrt | 11.4.0.106 |  | A high performance deep learning inference library |  |
| terminado | 0.18.1 |  | Tornado websocket backend for the Xterm.js Javascript terminal emulator library. | jupyter-server, jupyter-server-terminals |
| threadpoolctl | 3.8.0 | B | threadpoolctl | blosc2 |
| tinycss2 | 1.5.1 |  | A tiny CSS parser |  |
| tokenizers | 0.23.3 |  |  | torchtitan |
| torch | 2.15.0a0+875d815502.nv26 |  | Tensors and Dynamic neural networks in Python with strong GPU acceleration | attn-gym, e3nn, flash-attn-4, flash-attn, mace-torch, nvdlfw-inspect … |
| torch\_c\_dlpack\_ext | 0.1.5 |  | torch c dlpack ext | flash-attn-4, quack-kernels |
| torch\_checkpointing | 0.1.0 |  | A high-performance, distributed-training-ready checkpointing library for PyTorch models | torchtitan |
| torch-ema | 0.3 | B | PyTorch library for computing moving averages of model parameters. | mace-torch |
| torch\_remat | 0.2.0 |  | Explicit activation rematerialization helpers for PyTorch | torchtitan |
| torch-sim-atomistic | 0.6.2 | R | A pytorch toolkit for calculating material properties using MLIPs |  |
| torch\_tensorrt | 2.15.0a0 |  | Torch-TensorRT is a package which allows users to automatically compile PyTorch and TorchScript modules to Ten |  |
| torchao | 0.19.0+git6792133e |  | Package for applying ao techniques to GPU models |  |
| torchmetrics | 1.9.0 | B | PyTorch native Metrics | mace-torch |
| torchtitan | 0.2.2+git65ba8a69 |  | A PyTorch native platform for training generative AI models |  |
| torchvision | 0.30.0a0+ac8d215f.nv26.1 |  | image and video datasets and models for torch deep learning |  |
| tornado | 6.5.10 |  | Tornado is a Python web framework and asynchronous networking library, originally developed at FriendFeed. | ipykernel, jupyter-client, jupyter-server, jupyterlab, jupyterlab-tensorboard-pro, notebook … |
| tqdm | 4.70.1 |  | Fast, Extensible Progress Meter | cuequivariance-ops-cu13, datasets, huggingface-hub, mace-torch, nvidia-modelopt, torch-sim-atomistic |
| traitlets | 5.16.1 |  | Traitlets Python configuration system | ipykernel, ipython, jupyter-events, jupyter-builder, jupyter-client, jupyter-core … |
| transformer\_engine | 2.20.2+6ea2a74a |  | Transformer acceleration library |  |
| triton | 3.8.0+gitc01b6774.nv26.1 |  | A language and compiler for custom Deep Learning operations |  |
| triton\_kernels | 1.0.0+gitc01b6774.nv26.1 |  |  |  |
| typeguard | 4.6.0 |  | Run-time type checker for Python | tyro |
| typing\_extensions | 4.16.0 |  | Backported and Experimental Type Hints for Python 3.9+ | aiohttp, aiosignal, anyio, apache-tvm-ffi, ase, beautifulsoup4 … |
| typing-inspection | 0.4.4 |  | Runtime typing introspection tools | fastapi, pydantic, pydantic-settings |
| tyro | 1.0.16 |  | CLI interfaces & config objects, from types | torchtitan |
| tzdata | 2026.5 |  | Provider of IANA time zone data | arrow |
| uri-template | 1.3.0 |  | RFC 6570 URI Template Processor |  |
| urllib3 | 2.8.0 |  | HTTP library with thread-safe connection pooling, file post, and more. | requests |
| uvicorn | 0.54.0 |  | The lightning-fast ASGI server. | nvidia-resiliency-ext |
| uvloop | 0.23.0 |  | Fast implementation of asyncio event loop on top of libuv |  |
| wandb | 0.30.0 |  | A CLI and library for interacting with the Weights & Biases API. | torchtitan |
| warp-lang | 1.19.0 | B | A Python framework for high-performance simulation and graphics programming | nvalchemi-toolkit-ops |
| watchfiles | 1.3.0 |  | Simple, modern and high performance file watching and code reload in python. |  |
| wcwidth | 0.9.2 |  | Measures the displayed width of unicode strings in a terminal | prettytable, prompt-toolkit |
| webcolors | 25.10.0 |  | A library for working with the color formats defined by HTML and CSS. |  |
| webencodings | 0.6.1 |  | Character encoding aliases for legacy web content | bleach, tinycss2 |
| websocket-client | 1.9.2 |  | WebSocket client for Python with low level API options | jupyter-server |
| websockets | 17.2 |  | An implementation of the WebSocket Protocol (RFC 6455 & 7692) |  |
| Werkzeug | 3.1.9 |  | The comprehensive WSGI web application library. | tensorboard |
| wheel | 0.48.0 |  | Command line tool for manipulating wheel files | astunparse |
| wrapt | 2.5.0 |  | Module for decorators, wrappers and monkey patching. | deprecated |
| xxhash | 4.0.1 |  | Python binding for xxHash | datasets, wandb |
| yarl | 1.25.1 |  | Yet another URL library | aiohttp |
| zipp | 4.1.1 |  | Backport of pathlib-compatible object wrapper for zip files | importlib-metadata |

## Container C (`C_cpu.sif`)

147 distributions. **R** marks those the recipe names.

| Package | Version | | Summary (from the package) | Required by |
| --- | --- | --- | --- | --- |
| amespahdbpythonsuite | 0.5.0.post96+ge8961b2 | R | A Python module to work with a downloaded PAHdb XML-file. |  |
| anyio | 4.16.0 |  | High-level concurrency and networking framework on top of asyncio or Trio | httpx, jupyter-server |
| argon2-cffi | 25.1.0 |  | Argon2 for Python | jupyter-server |
| argon2-cffi-bindings | 26.1.0 |  | Low-level CFFI bindings for Argon2 | argon2-cffi |
| arrow | 1.4.0 |  | Better dates & times for Python | isoduration |
| asdf | 6.0.0 |  | Python implementation of the ASDF Standard | asdf-astropy, asdf-coordinates-schemas, gwcs, specutils |
| asdf-astropy | 0.11.0 |  | ASDF serialization support for astropy | gwcs, specutils |
| asdf\_coordinates\_schemas | 0.5.1 |  | ASDF schemas for coordinates | asdf-astropy, asdf-wcs-schemas |
| asdf\_standard | 1.5.0 |  | The ASDF Standard schemas | asdf, asdf-astropy, asdf-coordinates-schemas, asdf-transform-schemas, asdf-wcs-schemas |
| asdf\_transform\_schemas | 0.6.0 |  | ASDF schemas for transforms | asdf-astropy, asdf-wcs-schemas |
| asdf\_wcs\_schemas | 0.5.0 |  | ASDF WCS schemas | gwcs |
| astropy | 8.0.2 | R | Astronomy and astrophysics core library | amespahdbpythonsuite, asdf-astropy, gwcs, ndcube, pahfit, specutils |
| astropy-iers-data | 0.2026.11.2.0.34.55 |  | IERS Earth Rotation and Leap Second tables for the astropy core package | astropy |
| asttokens | 3.0.2 |  | Annotate AST trees with source code positions | stack-data |
| async-lru | 2.4.0 |  | Simple LRU cache for asyncio | jupyterlab |
| attrs | 26.1.0 |  | Classes Without Boilerplate | asdf, jsonschema, referencing |
| babel | 2.18.0 |  | Internationalization utilities | jupyterlab-server |
| beautifulsoup4 | 4.15.0 |  | Screen-scraping library | nbconvert |
| bleach | 6.4.0 |  | An easy safelist-based HTML-sanitizing tool. | nbconvert |
| certifi | 2026.10.23 |  | Python package for providing Mozilla's CA Bundle. | httpcore, httpx, requests |
| cffi | 2.1.1 |  | Foreign Function Interface for Python calling C code. | argon2-cffi-bindings |
| charset-normalizer | 3.5.2 |  | The Real First Universal Charset Detector. Open, modern and actively maintained alternative to Chardet. | requests |
| click | 8.5.1 |  | Composable command line interface toolkit | pysr |
| cloudpickle | 3.1.2 |  | Pickler class to extend the standard pickle.Pickler functionality | joblib |
| comm | 0.2.3 |  | Jupyter Python Comm implementation, for usage in ipykernel, xeus-python etc. | ipykernel, ipywidgets |
| contourpy | 1.4.0 |  | Python library for calculating contours of 2D quadrilateral grids | matplotlib |
| cycler | 0.12.1 |  | Composable style cycles | matplotlib |
| debugpy | 1.8.22 |  | An implementation of the Debug Adapter Protocol for Python | ipykernel |
| defusedxml | 0.7.1 |  | XML bomb protection for Python stdlib modules | nbconvert |
| executing | 2.3.0 |  | Get the currently executing AST node of a frame, and other information | stack-data |
| fastjsonschema | 2.22.3 |  | Fastest Python implementation of JSON schema | nbformat |
| filelock | 4.2.9 |  | A platform independent file lock. | juliapkg |
| fnnls | 1.0.0 |  | Fast Nonnegative Least Squares | amespahdbpythonsuite |
| fonttools | 4.68.0 |  | Tools to manipulate font files | matplotlib |
| fqdn | 1.6.0 |  | Validates fully-qualified domain names against RFC 1123, so that they are acceptable to modern browsers |  |
| gwcs | 1.0.3 |  | Generalized World Coordinate System | ndcube, specutils |
| h11 | 0.16.0 |  | A pure-Python, bring-your-own-I/O implementation of HTTP/1.1 | httpcore |
| h5py | 3.16.0 | R | Read and write HDF5 files from Python |  |
| httpcore | 1.0.9 |  | A minimal low-level HTTP client. | httpx |
| httpx | 0.28.1 |  | The next generation HTTP client. | jupyterlab |
| idna | 3.20 |  | Internationalized Domain Names in Applications (IDNA) | anyio, httpx, requests |
| importlib\_resources | 7.1.0 |  | Read resources from Python packages | amespahdbpythonsuite |
| iniconfig | 2.3.1 |  | brain-dead simple config-ini parsing | pytest |
| ipykernel | 7.4.0 |  | IPython Kernel for Jupyter | jupyter, jupyter-console, jupyterlab |
| ipython | 9.18.1 |  | IPython: Productive Interactive Computing | ipykernel, ipywidgets, jupyter-console |
| ipython\_pygments\_lexers | 1.1.1 |  | Defines a variety of Pygments lexers for highlighting IPython code. | ipython |
| ipywidgets | 8.1.9 |  | Jupyter interactive widgets | jupyter |
| isoduration | 20.11.0 |  | Operations with ISO 8601 durations |  |
| jedi | 0.20.1 |  | An autocompletion tool for Python that can be used for text editors. | ipython |
| Jinja2 | 3.1.6 |  | A very fast and expressive template engine. | jupyter-server, jupyterlab, jupyterlab-server, nbconvert |
| jmespath | 1.1.0 |  | JSON Matching Expressions | asdf |
| joblib | 1.6.0 |  | Lightweight pipelining with Python functions | scikit-learn |
| json5 | 0.17.3 |  | A Python implementation of the JSON5 data format. | jupyterlab-server |
| jsonpointer | 3.10 |  | Identify specific nodes in a JSON document (RFC 6901)  |  |
| jsonschema | 4.26.0 |  | An implementation of JSON Schema validation for Python | jupyter-events, jupyterlab-server, nbformat |
| jsonschema-specifications | 2025.9.1 |  | The JSON Schema meta-schemas and vocabularies, exposed as a Registry | jsonschema |
| juliacall | 0.9.36 |  | Julia and Python in seamless harmony | pysr |
| juliapkg | 0.1.28 |  | Julia version manager and package manager | juliacall |
| jupyter | 1.1.1 | R | Jupyter metapackage. Install all the Jupyter components in one go. |  |
| jupyter\_builder | 1.2.4 |  | JupyterLab build tools | jupyterlab, notebook |
| jupyter\_client | 8.10.0 |  | Jupyter protocol implementation and client libraries | ipykernel, jupyter-console, jupyter-server, nbclient |
| jupyter-console | 6.6.3 |  | Jupyter terminal console | jupyter |
| jupyter\_core | 5.9.1 |  | Jupyter core package. A base package on which Jupyter projects rely. | ipykernel, jupyter-console, jupyter-builder, jupyter-client, jupyter-server, jupyterlab … |
| jupyter-events | 0.12.1 |  | Jupyter Event System library | jupyter-server |
| jupyter-lsp | 2.3.1 |  | Multi-Language Server WebSocket proxy for Jupyter Notebook/Lab server | jupyterlab |
| jupyter\_server | 2.21.1 |  | The backend—i.e. core services, APIs, and REST endpoints—to Jupyter web applications. | jupyter-lsp, jupyterlab, jupyterlab-server, notebook, notebook-shim |
| jupyter\_server\_terminals | 0.5.4 |  | A Jupyter Server Extension Providing Terminals. | jupyter-server |
| jupyterlab | 4.6.6 |  | JupyterLab computational environment | jupyter, notebook |
| jupyterlab\_pygments | 0.3.0 |  | Pygments theme using JupyterLab CSS variables | nbconvert |
| jupyterlab\_server | 2.28.1 |  | A set of server components for JupyterLab and JupyterLab like applications. | jupyterlab, notebook |
| jupyterlab\_widgets | 3.0.17 |  | Jupyter interactive widgets for JupyterLab | ipywidgets |
| kiwisolver | 1.5.1 |  | A fast implementation of the Cassowary constraint solver | matplotlib |
| lark | 1.3.1 |  | a modern parsing library | rfc3987-syntax |
| lxml | 6.1.3 |  | Powerful and Pythonic XML processing library combining libxml2/libxslt with the ElementTree API. | amespahdbpythonsuite |
| MarkupSafe | 3.0.4 |  | Safely add untrusted strings to HTML/XML markup. | jinja2, nbconvert |
| matplotlib | 3.11.2 | R | Python plotting package | amespahdbpythonsuite, pahfit, vtk |
| matplotlib-inline | 0.2.2 |  | Inline Matplotlib backend for Jupyter | ipykernel, ipython |
| mistune | 3.4.0 |  | A sane and fast Markdown parser with useful plugins and renderers | nbconvert |
| mpmath | 1.3.0 |  | Python library for arbitrary-precision floating-point arithmetic | sympy |
| narwhals | 2.28.0 |  | Extremely lightweight compatibility layer between dataframe libraries | scikit-learn |
| nbclient | 0.11.0 |  | A client library for executing notebooks. Formerly nbconvert's ExecutePreprocessor. | nbconvert |
| nbconvert | 7.17.2 |  | Convert Jupyter Notebooks (.ipynb files) to other formats. | jupyter, jupyter-server |
| nbformat | 5.11.1 |  | The Jupyter Notebook format | jupyter-server, nbclient, nbconvert |
| ndcube | 2.4.3 |  | A package for multi-dimensional contiguous and non-contiguous coordinate aware arrays. | specutils |
| nest-asyncio2 | 1.7.4 |  | Patch asyncio to allow nested event loops | ipykernel |
| notebook | 7.6.5 |  | Jupyter Notebook - A web-based notebook environment for interactive computing | jupyter |
| notebook\_shim | 0.2.4 |  | A shim layer for notebook traits and config | jupyterlab, notebook |
| numpy | 2.5.4 |  | Fundamental package for array computing in Python | amespahdbpythonsuite, asdf, asdf-astropy, astropy, contourpy, fnnls … |
| packaging | 26.3 |  | Core utilities for Python packages | amespahdbpythonsuite, asdf, asdf-astropy, astropy, ipykernel, jupyter-events … |
| pahfit | 2.6.dev93+gca599f9a1 | R | fit the features in near- and mid-IR astronomical spectra |  |
| pandas | 3.1.1 | R | Powerful data structures for data analysis, time series, and statistics | pysr |
| pandocfilters | 1.5.1 |  | Utilities for writing pandoc filters in python | nbconvert |
| parso | 0.8.7 |  | A Python Parser | jedi |
| pexpect | 4.9.0 |  | Pexpect allows easy control of interactive console applications. | ipython |
| pillow | 13.0.0 |  | Python Imaging Library (fork) | matplotlib |
| pip | 26.3 |  | The PyPA recommended tool for installing Python packages. |  |
| platformdirs | 4.14.5 |  | A small Python package for determining appropriate platform-specific dirs, e.g. a `user data dir`. | jupyter-core |
| pluggy | 1.7.0 |  | plugin and hook calling mechanisms for python | pytest |
| prometheus\_client | 0.26.0 |  | Python client for the Prometheus monitoring system. | jupyter-server |
| prompt\_toolkit | 3.0.53 |  | Library for building powerful interactive command lines in Python | ipython, jupyter-console |
| psutil | 7.2.2 |  | Cross-platform lib for process and system monitoring. | ipython |
| ptyprocess | 0.7.0 |  | Run a subprocess in a pseudo terminal | pexpect, terminado |
| pure\_eval | 0.2.4 |  | Safely evaluate AST nodes without side effects | stack-data |
| pycparser | 3.12 |  | C parser in Python | cffi |
| pyerfa | 2.0.1.5 |  | Python bindings for ERFA | astropy |
| Pygments | 2.21.0 |  | Pygments is a syntax highlighting package written in Python. | ipython, ipython-pygments-lexers, jupyter-console, nbconvert, pytest |
| pyparsing | 3.3.3 |  | pyparsing - Classes and methods to define and execute parsing grammars | matplotlib |
| pysr | 2.7.2 | R | Simple and efficient symbolic regression |  |
| pytest | 9.2.0 |  | pytest: simple powerful testing with Python | fnnls |
| python-dateutil | 4.0.0 |  | Extensions to the standard Python datetime module | arrow, jupyter-client, matplotlib, pandas |
| python-json-logger | 4.2.0 |  | JSON Log Formatter for the Python Logging Package | jupyter-events |
| PyYAML | 6.0.3 |  | YAML parser and emitter for Python | asdf, astropy, jupyter-events |
| pyzmq | 27.3.0 |  | Python bindings for 0MQ | ipykernel, jupyter-console, jupyter-client, jupyter-server |
| referencing | 0.37.0 |  | JSON Referencing + Python | jsonschema, jsonschema-specifications, jupyter-events |
| requests | 2.34.2 |  | Python HTTP for Humans. | jupyterlab-server |
| rfc3339-validator | 0.1.4 |  | A pure python RFC3339 validator | jupyter-events |
| rfc3986-validator | 0.1.1 |  | Pure python rfc3986 validator | jupyter-events |
| rfc3987-syntax | 1.1.0 |  | Helper functions to syntactically validate strings according to RFC 3987. |  |
| rpds-py | 2026.9.1 |  | Python bindings to Rust's persistent data structures (rpds) | jsonschema, referencing |
| scikit-learn | 1.9.1 | R | A set of python modules for machine learning and data mining | pysr |
| scipy | 1.18.1 |  | Fundamental algorithms for scientific computing in Python | amespahdbpythonsuite, gwcs, ndcube, pahfit, scikit-learn, specutils |
| semantic-version | 2.10.0 |  | A library implementing the 'SemVer' scheme. | asdf |
| semver | 3.1.0 |  | Python helper for Semantic Versioning (https://semver.org) | juliapkg |
| Send2Trash | 2.1.0 |  | Send file to trash natively under Mac OS X, Windows and Linux | jupyter-server |
| six | 1.17.0 |  | Python 2 and 3 compatibility utilities | rfc3339-validator |
| soupsieve | 4.0.1 |  | A modern CSS selector implementation for Beautiful Soup. | beautifulsoup4 |
| specutils | 2.4.0 | R | Package for spectroscopic astronomical data | amespahdbpythonsuite, pahfit |
| stack-data | 0.6.3 |  | Extract data from python stack frames and tracebacks for informative displays | ipython |
| sympy | 1.14.0 |  | Computer algebra system (CAS) in Python | pysr |
| terminado | 0.18.1 |  | Tornado websocket backend for the Xterm.js Javascript terminal emulator library. | jupyter-server, jupyter-server-terminals |
| threadpoolctl | 3.8.0 |  | threadpoolctl | scikit-learn |
| tinycss2 | 1.5.1 |  | A tiny CSS parser |  |
| tomli | 2.5.0 |  | A lil' TOML parser | juliapkg |
| tomlkit | 0.15.2 |  | Style preserving TOML library | juliapkg |
| tornado | 6.6 |  | Tornado is a Python web framework and asynchronous networking library, originally developed at FriendFeed. | ipykernel, jupyter-client, jupyter-server, jupyterlab, notebook, terminado |
| tqdm | 4.70.1 |  | Fast, Extensible Progress Meter | amespahdbpythonsuite |
| traitlets | 5.16.1 |  | Traitlets Python configuration system | ipykernel, ipython, ipywidgets, jupyter-console, jupyter-events, jupyter-builder … |
| typing\_extensions | 4.16.0 |  | Backported and Experimental Type Hints for Python 3.9+ | anyio, asdf, beautifulsoup4, jupyter-client, pysr, referencing |
| tzdata | 2026.6 |  | Provider of IANA time zone data | arrow, python-dateutil |
| uri-template | 1.3.0 |  | RFC 6570 URI Template Processor |  |
| urllib3 | 2.8.0 |  | HTTP library with thread-safe connection pooling, file post, and more. | requests |
| vtk | 9.7.2 |  | VTK is an open-source toolkit for 3D computer graphics, image processing, and visualization | amespahdbpythonsuite |
| wcwidth | 0.9.4 |  | Measures the displayed width of unicode strings in a terminal | prompt-toolkit |
| webcolors | 25.10.0 |  | A library for working with the color formats defined by HTML and CSS. |  |
| webencodings | 0.6.1 |  | Character encoding aliases for legacy web content | bleach, tinycss2 |
| websocket-client | 1.9.2 |  | WebSocket client for Python with low level API options | jupyter-server |
| widgetsnbextension | 4.0.16 |  | Jupyter interactive widgets for Jupyter Notebook | ipywidgets |

