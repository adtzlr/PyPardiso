# coding: utf-8
import numpy as np
import pytest
import scipy.sparse as sp

from pypardiso import PyPardisoSolver, spsolve


def create_symmetric(n=200, seed=27):
    # symmetric and positive definite (diagonally dominant)
    rng = np.random.default_rng(seed)
    M = sp.random(n, n, density=0.05, random_state=seed)
    A = (M + M.T + sp.diags(np.full(n, 10.0))).tocsr()
    return A, rng.random(n)


def assert_solution(A, x, b, tol=1e-12):
    r = A @ x.astype(np.float64).reshape(b.shape) - b
    assert np.linalg.norm(r) / np.linalg.norm(b) < tol


@pytest.mark.parametrize('mtype', [2, -2])
def test_upper_triangle_csr(mtype):
    A, b = create_symmetric()
    x = PyPardisoSolver(mtype=mtype).solve(sp.triu(A, format='csr'), b)
    assert_solution(A, x, b)


@pytest.mark.parametrize('mtype', [2, -2])
def test_full_matrix_raises(mtype):
    # with entries below the diagonal, pardiso doesn't return for symmetric matrix types
    A, b = create_symmetric()
    ps = PyPardisoSolver(mtype=mtype)
    with pytest.raises(ValueError, match='upper triangle'):
        ps.solve(A, b)
    with pytest.raises(ValueError, match='upper triangle'):
        ps.factorize(A)
    with pytest.raises(ValueError, match='upper triangle'):
        spsolve(A, b, solver=ps)


def test_lower_triangle_csr_raises():
    A, b = create_symmetric()
    with pytest.raises(ValueError, match='triu'):
        PyPardisoSolver(mtype=-2).solve(sp.tril(A, format='csr'), b)


def test_csc_input():
    # the arrays of a CSC matrix are the CSR arrays of A^T: solve() needs the lower triangle,
    # spsolve() converts CSC to CSR and needs the upper triangle
    A, b = create_symmetric()
    ps = PyPardisoSolver(mtype=-2)
    assert_solution(A, ps.solve(sp.tril(A, format='csc'), b), b)
    with pytest.raises(ValueError, match='tril'):
        ps.solve(sp.triu(A, format='csc'), b)
    assert_solution(A, spsolve(sp.triu(A, format='csc'), b, solver=ps), b)


def test_set_matrix_type():
    A, b = create_symmetric()
    ps = PyPardisoSolver()
    assert_solution(A, ps.solve(A, b), b)  # full matrix for mtype=11
    ps.set_matrix_type(-2)
    with pytest.raises(ValueError):
        ps.solve(A, b)
    assert_solution(A, ps.solve(sp.triu(A, format='csr'), b), b)


def test_missing_diagonal_entry():
    # a symmetric nonsingular matrix without a stored diagonal entry in one row
    A, b = create_symmetric()
    A = A.tolil()
    A[5, 5] = 0
    A[5, 6] = A[6, 5] = 3.0
    A = A.tocsr()
    A.eliminate_zeros()
    U = sp.triu(A, format='csr')
    assert 5 not in U.indices[U.indptr[5]:U.indptr[6]]
    assert_solution(A, PyPardisoSolver(mtype=-2).solve(U, b), b)


def test_float32():
    A, b = create_symmetric()
    x = PyPardisoSolver(mtype=-2).solve(sp.triu(A, format='csr').astype(np.float32), b)
    assert x.dtype == np.float32
    assert_solution(A, x, b, tol=1e-5)


def test_nonsymmetric_types_accept_full_matrix():
    A, b = create_symmetric()
    for mtype in (1, 11):
        assert_solution(A, PyPardisoSolver(mtype=mtype).solve(A, b), b)
