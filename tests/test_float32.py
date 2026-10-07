# coding: utf-8
import warnings

import numpy as np
import pytest
import scipy.sparse as sp

from pypardiso import PyPardisoSolver, spsolve
from pypardiso.pardiso_wrapper import PyPardisoWarning
from utils import create_test_A_b_rand


def assert_small_residual(A, x, b, tol):
    r = A @ x.astype(np.float64) - b.reshape(x.shape)
    assert np.linalg.norm(r) / np.linalg.norm(b) < tol


def test_float32_solve():
    A, b = create_test_A_b_rand(matrix=True)
    ps = PyPardisoSolver()
    x = ps.solve(A.astype(np.float32), b)
    assert x.dtype == np.float32
    assert ps.get_iparm(28) == 1
    assert_small_residual(A, x, b, 1e-3)


def test_float32_first_call_csc():
    # on a fresh solver, iparm(12) (transposed solve for CSC) must already be honored in the first call
    A, b = create_test_A_b_rand(n=200, density=0.05)
    for dtype, tol in ((np.float64, 1e-12), (np.float32, 1e-3)):
        x = PyPardisoSolver().solve(A.astype(dtype).tocsc(), b)
        assert_small_residual(A, x, b, tol)


def test_float64_rhs_is_cast_silently():
    A, b = create_test_A_b_rand()
    with warnings.catch_warnings():
        warnings.simplefilter("error", PyPardisoWarning)
        x = PyPardisoSolver().solve(A.astype(np.float32), b.astype(np.float64))
    assert x.dtype == np.float32


def test_int_rhs_warns_for_float32():
    A, b = create_test_A_b_rand()
    with pytest.warns(PyPardisoWarning):
        PyPardisoSolver().solve(A.astype(np.float32), (100 * b).astype(np.int64))


def test_switch_precision_same_values():
    # integer-valued matrix: float32 and float64 data compare equal with np.array_equal,
    # the stored factorization must not be reused across precisions
    n = 50
    A = sp.diags([-np.ones(n - 1), 4 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1], format="csr")
    b = np.arange(n, dtype=np.float64)
    ps = PyPardisoSolver()
    for dtype in (np.float64, np.float32, np.float64, np.float32):
        Ad = A.astype(dtype)
        ps.factorize(Ad)
        x = ps.solve(Ad, b)
        assert x.dtype == dtype
        assert ps.get_iparm(28) == int(dtype == np.float32)
        assert_small_residual(A, x, b, 1e-5)


def test_phase33_with_wrong_precision_raises():
    A, b = create_test_A_b_rand()
    ps = PyPardisoSolver()
    ps.factorize(A.astype(np.float32))
    ps.set_phase(33)
    with pytest.raises(TypeError):
        ps._call_pardiso(A, b)


def test_spsolve_float32():
    A, b = create_test_A_b_rand()
    x = spsolve(A.astype(np.float32), b)
    assert x.dtype == np.float32
    assert_small_residual(A, x, b, 1e-3)
    # float64 again with the shared solver
    x64 = spsolve(A, b)
    assert x64.dtype == np.float64
    assert_small_residual(A, x64, b, 1e-12)
