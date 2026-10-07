# coding: utf-8
import ctypes
import warnings

import numpy as np
import scipy.sparse as sp
from numpy.linalg import LinAlgError
from scipy.sparse import SparseEfficiencyWarning

from .scipy_aliases import pypardiso_solver

# constants of the MKL Inspector-Executor Sparse BLAS, see mkl_spblas.h
_STATUS_SUCCESS = 0
_STATUS_EXECUTION_FAILED = 4
_STATUS_NAMES = {1: 'not initialized', 2: 'allocation failed', 3: 'invalid value', 4: 'execution failed',
                 5: 'internal error', 6: 'not supported'}
_OPERATION_NON_TRANSPOSE = 10
_OPERATION_TRANSPOSE = 11
_MATRIX_TYPE_TRIANGULAR = 23
_INDEX_BASE_ZERO = 0
_FILL_MODE_LOWER = 40
_FILL_MODE_UPPER = 41
_DIAG_NON_UNIT = 50
_DIAG_UNIT = 51
_LAYOUT_ROW_MAJOR = 101


class _MatrixDescr(ctypes.Structure):
    """struct matrix_descr of mkl_spblas.h (all members are C enums)"""
    _fields_ = [('type', ctypes.c_int), ('mode', ctypes.c_int), ('diag', ctypes.c_int)]


class _SparseBLAS:
    """ctypes interface to the triangular solvers of the MKL Inspector-Executor Sparse BLAS (32-bit integers, as
    used for Pardiso)."""

    def __init__(self, libmkl):
        int_ = ctypes.c_int32
        enum = ctypes.c_int
        ptr = ctypes.c_void_p

        self.create_csr, self.trsv, self.trsm = {}, {}, {}

        for prefix, dtype, real in (('s', np.float32, ctypes.c_float), ('d', np.float64, ctypes.c_double)):

            create_csr = getattr(libmkl, 'mkl_sparse_{}_create_csr'.format(prefix))
            create_csr.argtypes = [ctypes.POINTER(ptr),  # sparse_matrix_t *A
                                   enum,                 # indexing
                                   int_, int_,           # rows, cols
                                   ptr, ptr, ptr,        # rows_start, rows_end, col_indx
                                   ptr]                  # values
            create_csr.restype = enum

            trsv = getattr(libmkl, 'mkl_sparse_{}_trsv'.format(prefix))
            trsv.argtypes = [enum, real, ptr, _MatrixDescr,  # operation, alpha, A, descr
                             ptr, ptr]                       # x (right-hand side), y (solution)
            trsv.restype = enum

            trsm = getattr(libmkl, 'mkl_sparse_{}_trsm'.format(prefix))
            trsm.argtypes = [enum, real, ptr, _MatrixDescr, enum,  # operation, alpha, A, descr, layout
                             ptr, int_, int_,                      # x (right-hand sides), columns, ldx
                             ptr, int_]                            # y (solutions), ldy
            trsm.restype = enum

            self.create_csr[dtype], self.trsv[dtype], self.trsm[dtype] = create_csr, trsv, trsm

        self.destroy = libmkl.mkl_sparse_destroy
        self.destroy.argtypes = [ptr]
        self.destroy.restype = enum


_sparse_blas = None


def _get_sparse_blas():
    global _sparse_blas
    if _sparse_blas is None:
        _sparse_blas = _SparseBLAS(pypardiso_solver.libmkl)
    return _sparse_blas


def _check_status(status, routine):
    if status == _STATUS_EXECUTION_FAILED:
        raise LinAlgError('A is singular, MKL {} failed.'.format(routine))
    if status != _STATUS_SUCCESS:
        raise RuntimeError('MKL {} failed with status {} ({}).'.format(routine, status,
                                                                       _STATUS_NAMES.get(status, 'unknown')))


def spsolve_triangular(A, b, lower=True, overwrite_A=False, overwrite_b=False, unit_diagonal=False):
    """
    This function mimics scipy.sparse.linalg.spsolve_triangular, but uses the triangular solver of the Intel MKL
    Sparse BLAS (mkl_sparse_?_trsv / mkl_sparse_?_trsm) instead of SuperLU.

        solve Ax=b for x, where A is a lower or upper triangular matrix

        --- Parameters ---
        A: sparse square CSR or CSC matrix
           only the triangle given by `lower` is used, entries in the other triangle are ignored
        b: numpy ndarray, shape (n,) or (n, k)
           right-hand side(s)
        lower: boolean, default True
               whether A is a lower or an upper triangular matrix
        overwrite_A: boolean, default False
                     allow sorting the indices of A and summing up duplicate entries in place
        overwrite_b: boolean, default False
                     only for compatibility with scipy, b is never changed
        unit_diagonal: boolean, default False
                       if True, the diagonal elements of A are assumed to be 1 and are not used

        --- Returns ---
        x: numpy ndarray
           solution of the system of linear equations, same shape as b

        --- Notes ---
        The dtype of x follows scipy: the dtypes of A, b and float32 are promoted to a common dtype, which must be
        float32 or float64. For float32 the single precision routines of MKL are used.
        A CSC matrix is not converted: its arrays are the CSR arrays of the transpose of A, which is solved with
        the transposed operation.
    """

    if not (sp.issparse(A) and A.format in ('csr', 'csc')):
        warnings.warn('CSR or CSC matrix format is required. Converting to CSR matrix.', SparseEfficiencyWarning,
                      stacklevel=2)
        A = sp.csr_array(A)

    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError('A must be a square matrix but its shape is {}.'.format(A.shape))

    b = np.asanyarray(b)
    if b.ndim not in (1, 2):
        raise ValueError('b must have 1 or 2 dims but its shape is {}.'.format(b.shape))
    if b.shape[0] != n:
        raise ValueError('The size of the dimensions of A must be equal to the size of the first dimension of b '
                         'but the shape of A is {} and the shape of b is {}.'.format(A.shape, b.shape))

    dtype = np.promote_types(np.promote_types(A.dtype, np.float32), b.dtype)
    if dtype not in (np.float32, np.float64):
        raise TypeError('spsolve_triangular supports float32 and float64, but the dtypes of A and b are {} and {}.'
                        .format(A.dtype, b.dtype))
    dtype = np.dtype(dtype).type

    if n == 0 or b.size == 0:
        return np.zeros(b.shape, dtype=dtype)

    # sorted indices without duplicates
    if not A.has_canonical_format:
        if not overwrite_A:
            A = A.copy()
        A.sum_duplicates()

    if A.nnz > np.iinfo(np.int32).max:
        raise ValueError('A has too many non-zero entries for 32-bit indices.')

    if not unit_diagonal and np.any(A.diagonal() == 0):
        raise LinAlgError('A is singular: zero entry on diagonal.')

    # the arrays of a CSC matrix A are the CSR arrays of A^T: solve A^T^T x = b with the other triangle
    if A.format == 'csr':
        operation = _OPERATION_NON_TRANSPOSE
        mode = _FILL_MODE_LOWER if lower else _FILL_MODE_UPPER
    else:
        operation = _OPERATION_TRANSPOSE
        mode = _FILL_MODE_UPPER if lower else _FILL_MODE_LOWER

    descr = _MatrixDescr(_MATRIX_TYPE_TRIANGULAR, mode, _DIAG_UNIT if unit_diagonal else _DIAG_NON_UNIT)

    data = np.ascontiguousarray(A.data, dtype=dtype)
    indptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
    indices = np.ascontiguousarray(A.indices, dtype=np.int32)

    rhs = np.ascontiguousarray(b.reshape(n, -1), dtype=dtype)  # row-major (n, k)
    k = rhs.shape[1]
    x = np.empty_like(rhs)

    blas = _get_sparse_blas()
    handle = ctypes.c_void_p()
    status = blas.create_csr[dtype](ctypes.byref(handle), _INDEX_BASE_ZERO, n, n,
                                    indptr.ctypes.data,                         # rows_start
                                    indptr.ctypes.data + indptr.itemsize,       # rows_end
                                    indices.ctypes.data, data.ctypes.data)
    _check_status(status, 'mkl_sparse_create_csr')

    # mkl_sparse_optimize() is not called: it analyzes the matrix for repeated solves and costs more than it saves
    # in a single call (with the transposed operation for CSC input, it even builds a transposed copy of A)
    try:
        if k == 1:
            status = blas.trsv[dtype](operation, 1.0, handle, descr, rhs.ctypes.data, x.ctypes.data)
            _check_status(status, 'mkl_sparse_trsv')
        else:
            status = blas.trsm[dtype](operation, 1.0, handle, descr, _LAYOUT_ROW_MAJOR,
                                      rhs.ctypes.data, k, k, x.ctypes.data, k)
            _check_status(status, 'mkl_sparse_trsm')
    finally:
        blas.destroy(handle)

    return x.reshape(b.shape)
