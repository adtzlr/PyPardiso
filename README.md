[![pypardiso-tests](https://github.com/haasad/PyPardisoProject/actions/workflows/tests.yaml/badge.svg?branch=master)](https://github.com/haasad/PyPardisoProject/actions/workflows/tests.yaml)
# PyPardiso

PyPardiso is a python package to solve large sparse linear systems of equations with the [Intel oneAPI Math Kernel Library PARDISO solver](https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-fortran/top/sparse-solver-routines/onemkl-pardiso-parallel-direct-sparse-solver-iface.html), a shared-memory multiprocessing parallel direct sparse solver.

PyPardiso provides the same functionality as SciPy's [scipy.sparse.linalg.spsolve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html#scipy.sparse.linalg.spsolve) for solving the sparse linear system `Ax=b`. However in many cases it is significantly faster than SciPy's built-in single-threaded SuperLU solver.

PyPardiso is not a python interface to the PARDISO Solver from the [PARDISO 7.2 Solver Project](https://www.pardiso-project.org/) and it also doesn't currently support complex numbers. Check out [JuliaSparse/Pardiso.jl](https://github.com/JuliaSparse/Pardiso.jl/) for these more advanced use cases. For macOS users we recommend [scikit-umfpack](https://github.com/scikit-umfpack/scikit-umfpack) as an alternative fast solver, since MKL is not available on Apple silicon.

## Installation

PyPardiso runs on Linux and Windows. It can be installed with __conda__ or __pip__. It is recommended to install PyPardiso using a virtual environment.

conda-forge | PyPI
:---:|:---:
[![conda-forge version](https://anaconda.org/conda-forge/pypardiso/badges/version.svg)](https://anaconda.org/conda-forge/pypardiso) | [![PyPI version](https://badge.fury.io/py/pypardiso.svg)](https://pypi.org/project/pypardiso/)
`conda install -c conda-forge pypardiso` | `pip install pypardiso`


## Basic usage

How to solve the sparse linear system `Ax=b` for `x`, where `A` is a square, sparse matrix in CSR (or CSC) format and `b` is a vector (or matrix):
```python
In [1]: import pypardiso

In [2]: import numpy as np

In [3]: import scipy.sparse as sp

In [4]: A = sp.rand(10, 10, density=0.5, format='csr')

In [5]: A
Out[5]:
<10x10 sparse matrix of type '<class 'numpy.float64'>'
	with 50 stored elements in Compressed Sparse Row format>

In [6]: b = np.random.rand(10)

In [7]: x = pypardiso.spsolve(A, b)

In [8]: x
Out[8]:
array([ 0.02918389,  0.59629935,  0.33407289, -0.48788966,  3.44508841,
        0.52565687, -0.48420646,  0.22136413, -0.95464127,  0.58297397])
```

## Single precision (float32)

Pass the matrix `A` as `float32` to run PARDISO in single precision (`iparm(28)=1`). The factorization then needs about
half the memory and is faster, the solution `x` is returned as `float32` (a `float64` right-hand side is cast silently).

```python
x = pypardiso.spsolve(A.astype(np.float32), b)
```

The accuracy of `x` is roughly `cond(A) * 1e-7`. Inside a Newton-Raphson loop with residuals evaluated in double
precision this is usually sufficient (inexact Newton), for ill-conditioned systems (`cond(A) > 1e6`) it is not.
Setting `iparm(8)` to a negative value (e.g. `-2`) accumulates the residuals of the iterative refinement in extended
precision, which helps somewhat.

## Triangular systems

`pypardiso.spsolve_triangular` mimics `scipy.sparse.linalg.spsolve_triangular`, but uses the triangular solver of the
MKL Sparse BLAS (`mkl_sparse_?_trsv` / `mkl_sparse_?_trsm`) instead of SuperLU. PARDISO itself is not used.

```python
x = pypardiso.spsolve_triangular(L, b)               # L lower triangular (CSR or CSC)
x = pypardiso.spsolve_triangular(U, b, lower=False)  # U upper triangular
```

Only the triangle given by `lower` is used, entries in the other triangle are ignored. `float32` and `float64` are
supported, the dtype of `x` follows the same promotion rules as in SciPy.
