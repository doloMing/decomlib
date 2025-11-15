# decomlib
decomlib is a high-performance, modular library designed for modern matrix and tensor decomposition tasks supporting a wide spectrum of algorithms and hardware platforms. At its core, decomlib provides efficient implementations of fundamental decompositions (such as singular value decomposition, truncated low-rank approximations, and structured matrix factorisations) and is built with extension in mind --- so advanced tensor decompositions (e.g., CP, Tucker, hierarchical formats) can be added seamlessly in the future.

The library architecture separates algorithmic logic, matrix/tensor data representations (dense, sparse, implicit operators), and execution back-ends (CPU multi-core, Intel/AMD/NVIDIA GPUs). Users can specify whether they need a full decomposition or only the top-k components, select matrix density/sparsity/structure, and choose among hardware back-ends or let the library pick the optimal path automatically.

For performance and flexibility, decomlib uses a modern Fortran core for numerically intensive routines, with a clean C-binding interface for Python wrappers so you can leverage it in data-science pipelines and research workflows. Memory and compute tasks are abstracted so that dense operations, sparse matrix-vector multiplies and structured operator routines all plug into the same high-level API. Runtime instrumentation is built-in, so you can monitor iteration counts, residuals, error bounds and convergence behaviour.

Whether the task is performing large-scale low-rank approximations, exploring tensor network factorizations, or accelerating structured data decompositions on heterogeneous hardware, decomlib is crafted as a unified framework for the linear and multilinear algebra needs --- with simplicity in usage and depth in capability.

# Project framework
```
decomlib/
├── CMakeLists.txt                      
├── README.md                           
├── LICENSE                            
├── version.txt                         
├── src/
│   ├── api/
│   │   └── backend_api.f90             # Fortran module: defines the abstract interface for hardware back-end implementations (CPU, GPU, etc.).
│   ├── backend/
│   │   ├── cpu_mod.f90                 # Fortran module: the CPU back-end implementation (multi-core/BLAS/LAPACK path).
│   │   ├── nvidia_mod.f90              # Fortran module: NVIDIA GPU back-end implementation (CUDA/Fortran or via offload) .
│   │   ├── amd_mod.f90                 # Fortran module: AMD GPU back-end implementation (ROCm/HIP/Fortran) .
│   │   └── intel_mod.f90               # Fortran module: Intel GPU back-end implementation (oneAPI/DPC++/Fortran offload) .
│   ├── matrix/
│   │   ├── matrix_types.f90            # Fortran module: definitions of matrix/tensor types (dense, sparse, implicit) and common data structures.
│   │   ├── sparse_support.f90           # Fortran module: support routines for sparse matrix formats and operations (CSR/CSC).
│   │   └── implicit_matrix.f90          # Fortran module: interface for "implicit" matrix forms (user-provided A×x routines) and wrappers.
│   ├── algorithms/
│   │   ├── svd/
│   │   │   ├── randomized.f90          # Fortran module: implementation of randomized SVD algorithm (low-rank sketching path) .
│   │   │   ├── krylov.f90               # Fortran module: implementation of Krylov / block-Lanczos / subspace-iteration SVD path .
│   │   │   ├── hybrid_structured.f90    # Fortran module: hybrid and structure-exploiting SVD path (e.g., kernel, Hankel) .
│   │   │   └── full_svd.f90             # Fortran module: fallback path computing all singular values/vectors (dense SVD via LAPACK/solver) .
│   │   └── (future_algorithms/)         # Directory placeholder for future algorithms (e.g., NMF, tensor decompositions) .
│   ├── utils/
│   │   ├── memory_helpers.f90           # Fortran module: memory allocation, host-to-device transfers, pinned memory, alignment helpers.
│   │   └── logging.f90                  # Fortran module: logging, performance timers, iteration counts, residual monitors.
│   └── c_bindings/
│       └── c_interface.f90              # Fortran module with ISO_C_BINDING: exposes C-callable wrappers for Python (and other) bindings.
└── python/
    ├── setup.py                         # Python packaging/install script: defines metadata, builds/links library, installs Python module.
    └── decomlib/
        ├── __init__.py                  # Python package init: defines `__version__`, imports core interfaces.
        ├── core.py                      # Python module: general interface (setting back-end, logging level, meta info).
        ├── backend_selector.py            # Python module: logic for selecting hardware back-end, checking availability, runtime dispatch.
        ├── matrix.py                      # Python module: wrappers for matrix/tensor types (DenseMatrix, SparseMatrix, ImplicitMatrix) for user convenience.
        ├── svd/
        │   ├── __init__.py               # Python sub-package init for SVD algorithms.
        │   └── compute.py                  # Python module: high-level user API `compute_svd(...)`, accepts matrices, parameters, returns U, S, Vt.
        └── tensor/
            ├── __init__.py                # Python sub-package init for tensor decomposition algorithms.
            ├── cp.py                      # Python module: placeholder for CP decomposition API (future).
            └── tucker.py                  # Python module: placeholder for Tucker decomposition API (future).
```

# Project framework

 - The backend_api.f90 module defines a standardized interface so that algorithms and matrix types call generic routines like matmul, qr_decompose, svd_decompose, etc., without knowing which hardware back-end is active.
 - Each *_mod.f90 in backend/ implements that interface for a specific hardware type; compilation of each is controlled by build flags (in CMakeLists.txt) so you don't need separate "Fortran variants" visible to user --- user selects back-end, library picks correct implementation.
 - The matrix_types, sparse_support, and implicit_matrix modules allow abstraction of matrix representation such that algorithms can operate on a "matrix_t" type that may represent dense, sparse or implicit operator; this avoids rewriting algorithm logic for each storage.
 - The algorithms/svd folder holds the distinct algorithmic paths. Over time other decomposition algorithms can go into algorithms/nmf, algorithms/tensor, etc.
 - The utils modules provide needed support for memory management (especially important for GPU: host/device transfers, pinned memory) and logging/performance monitoring (since you emphasize performance).
 - The c_bindings folder ensures that Fortran routines expose C-linkable entry points (via ISO_C_BINDING) so Python (via ctypes/cffi or a small C wrapper) can load them transparently.
 - On the Python side:
    - core.py: e.g., functions like set_backend('nvidia_gpu'), get_available_backends(), set_log_level().
    - backend_selector.py: logic to detect available devices/back-ends at runtime, select or fallback.
    - matrix.py: classes exposing NumPy arrays, SciPy sparse, or user callback to Fortran library.
    - svd/compute.py: exposes compute_svd(A, k=..., algorithm='auto', hardware='auto', ...).
    - tensor/ sub-package reserved for future decompositions.
 - The CMakeLists.txt includes options like ENABLE_NVIDIA_GPU, ENABLE_AMD_GPU, ENABLE_INTEL_GPU, and conditionally compile modules accordingly.
