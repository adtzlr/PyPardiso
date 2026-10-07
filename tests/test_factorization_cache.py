# coding: utf-8
import numpy as np

from pypardiso import PyPardisoSolver
from utils import create_test_A_b_small


def test_cache_compares_dtype():
    # integer-valued data compares equal with np.array_equal for different dtypes, but a stored factorization
    # must only be reused for a matrix with the same dtype
    A, b = create_test_A_b_small()
    ps = PyPardisoSolver()
    assert ps._csr_matrix_equal(A, A.copy())
    assert not ps._csr_matrix_equal(A, A.astype(np.float32))
    assert ps._hash_csr_matrix(A) == ps._hash_csr_matrix(A.copy())
    assert ps._hash_csr_matrix(A) != ps._hash_csr_matrix(A.astype(np.float32))
