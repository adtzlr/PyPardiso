# coding: utf-8
import numpy as np

from pypardiso import PyPardisoSolver
from utils import create_test_A_b_rand


def test_iparm_defaults_after_init():
    # iparm is filled with the MKL defaults right away, iparm(1)=1 tells MKL to use the given values
    ps = PyPardisoSolver()
    assert ps.get_iparm(1) == 1
    assert ps.get_iparm(13) == 1  # weighted matching, default for real nonsymmetric matrices (mtype=11)


def test_first_call_csc():
    # a CSC matrix is solved with iparm(12)=1 (transposed solve), this has to work in the very first call of
    # a new solver instance. Before, MKL replaced iparm(12) by its default in the first call and the result was
    # the solution of A^T x = b.
    A, b = create_test_A_b_rand(n=200, density=0.05)
    x = PyPardisoSolver().solve(A.tocsc(), b)
    np.testing.assert_array_almost_equal(A @ x, b)


def test_first_call_user_iparm():
    # a user-defined iparm must be used in the first call: no iterative refinement steps with iparm(8)=0
    A, b = create_test_A_b_rand()
    ps = PyPardisoSolver()
    ps.set_iparm(8, 0)
    ps.solve(A, b)
    assert ps.get_iparm(7) == 0  # number of performed iterative refinement steps (output)


def test_set_matrix_type_resets_defaults():
    ps = PyPardisoSolver()
    ps.set_matrix_type(-2)  # real symmetric indefinite
    assert ps.get_iparm(10) == 8  # pivot perturbation 1e-8, default for symmetric indefinite matrices
    assert ps.get_iparm(13) == 0  # no weighted matching
    ps.set_matrix_type(11)
    assert ps.get_iparm(10) == 13
    assert ps.get_iparm(13) == 1
