# coding: utf-8
import warnings

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from numpy.linalg import LinAlgError
from scipy.sparse import SparseEfficiencyWarning

from pypardiso import spsolve_triangular


def create_triangular(n=300, lower=True, fmt='csr', seed=27):
    rng = np.random.default_rng(seed)
    M = sp.random(n, n, density=0.05, random_state=seed) + sp.diags(rng.random(n) + 1)
    T = sp.tril(M) if lower else sp.triu(M)
    return T.asformat(fmt), rng.random(n), rng.random((n, 4))


@pytest.mark.parametrize('lower', [True, False])
@pytest.mark.parametrize('fmt', ['csr', 'csc'])
@pytest.mark.parametrize('two_dim', [False, True])
def test_compare_with_scipy(lower, fmt, two_dim):
    A, b1, b2 = create_triangular(lower=lower, fmt=fmt)
    b = b2 if two_dim else b1
    x = spsolve_triangular(A, b, lower=lower)
    assert x.shape == b.shape
    assert x.dtype == np.float64
    np.testing.assert_allclose(x, sla.spsolve_triangular(A, b, lower=lower), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(A @ x, b, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('lower', [True, False])
def test_column_vector(lower):
    A, b, _ = create_triangular(lower=lower)
    x = spsolve_triangular(A, b[:, None], lower=lower)
    assert x.shape == (len(b), 1)
    np.testing.assert_allclose(A @ x, b[:, None], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('fmt', ['csr', 'csc'])
def test_unit_diagonal(fmt):
    # the stored diagonal is not used
    A, b, _ = create_triangular(fmt=fmt)
    x = spsolve_triangular(A, b, unit_diagonal=True)
    A1 = sp.tril(A, k=-1) + sp.eye(A.shape[0])
    np.testing.assert_allclose(A1 @ x, b, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('lower', [True, False])
@pytest.mark.parametrize('fmt', ['csr', 'csc'])
def test_other_triangle_is_ignored(lower, fmt):
    rng = np.random.default_rng(3)
    n = 100
    M = (sp.random(n, n, density=0.1, random_state=3) + sp.diags(rng.random(n) + 1)).asformat(fmt)
    T = sp.tril(M, format=fmt) if lower else sp.triu(M, format=fmt)
    b = rng.random(n)
    np.testing.assert_allclose(spsolve_triangular(M, b, lower=lower), spsolve_triangular(T, b, lower=lower),
                               rtol=1e-12, atol=1e-12)


def test_float32():
    A, b, B = create_triangular()
    for rhs in (b, B):
        x = spsolve_triangular(A.astype(np.float32), rhs.astype(np.float32))
        assert x.dtype == np.float32
        np.testing.assert_allclose(x, spsolve_triangular(A, rhs), rtol=1e-4, atol=1e-5)


def test_dtype_promotion_like_scipy():
    A, b, _ = create_triangular(n=10)
    assert spsolve_triangular(A.astype(np.float32), b).dtype == np.float64
    assert spsolve_triangular(A, b.astype(np.float32)).dtype == np.float64
    Ai = sp.csr_array(np.tril(np.arange(1, 10).reshape(3, 3)))
    x = spsolve_triangular(Ai, np.ones(3, dtype=np.int64))
    assert x.dtype == np.float64
    np.testing.assert_allclose(x, sla.spsolve_triangular(Ai, np.ones(3)))


def test_complex_raises():
    A, b, _ = create_triangular(n=10)
    with pytest.raises(TypeError):
        spsolve_triangular(A, b + 1j)
    with pytest.raises(TypeError):
        spsolve_triangular(A.astype(np.complex128), b)


@pytest.mark.parametrize('explicit_zero', [True, False])
def test_singular_raises(explicit_zero):
    A, b, _ = create_triangular()
    A = A.tolil()
    A[5, 5] = 0
    A = A.tocsr()
    if not explicit_zero:
        A.eliminate_zeros()
    with pytest.raises(LinAlgError):
        spsolve_triangular(A, b)
    # with a unit diagonal the stored diagonal does not matter
    spsolve_triangular(A, b, unit_diagonal=True)


def test_shape_errors():
    A, b, _ = create_triangular()
    with pytest.raises(ValueError):
        spsolve_triangular(A[:, :-1], b)
    with pytest.raises(ValueError):
        spsolve_triangular(A, b[:-1])
    with pytest.raises(ValueError):
        spsolve_triangular(A, b.reshape(-1, 1, 1))


def test_other_formats_warn():
    A, b, _ = create_triangular()
    for M in (A.tocoo(), A.toarray()):
        with pytest.warns(SparseEfficiencyWarning):
            x = spsolve_triangular(M, b)
        np.testing.assert_allclose(A @ x, b, rtol=1e-12, atol=1e-12)


def test_csr_csc_no_warning():
    A, b, _ = create_triangular()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        spsolve_triangular(A, b)
        spsolve_triangular(A.tocsc(), b)


@pytest.mark.parametrize('overwrite_A', [False, True])
def test_unsorted_and_duplicate_indices(overwrite_A):
    A, b, _ = create_triangular()
    A = A.tocoo()
    rng = np.random.default_rng(5)
    # duplicate the first 20 entries (halved), shuffle all entries
    row, col = np.r_[A.row, A.row[:20]], np.r_[A.col, A.col[:20]]
    data = np.r_[A.data, A.data[:20] / 2]
    data[:20] /= 2
    perm = rng.permutation(len(data))
    row, col, data = row[perm], col[perm], data[perm]
    # build the CSR arrays by hand, so that the duplicates and the unsorted order are kept
    order = np.argsort(row, kind='stable')
    indptr = np.r_[0, np.cumsum(np.bincount(row, minlength=A.shape[0]))]
    M = sp.csr_array((data[order], col[order], indptr), shape=A.shape)
    assert not M.has_canonical_format
    data_before = M.data.copy()

    x = spsolve_triangular(M, b, overwrite_A=overwrite_A)
    np.testing.assert_allclose(A @ x, b, rtol=1e-12, atol=1e-12)
    if not overwrite_A:
        np.testing.assert_array_equal(M.data, data_before)


def test_b_unchanged_and_list_input():
    A, b, B = create_triangular()
    b_before, B_before = b.copy(), B.copy()
    spsolve_triangular(A, b, overwrite_b=True)
    spsolve_triangular(A, B)
    np.testing.assert_array_equal(b, b_before)
    np.testing.assert_array_equal(B, B_before)
    np.testing.assert_allclose(spsolve_triangular(A, list(b)), spsolve_triangular(A, b))


def test_fortran_ordered_rhs():
    A, _, B = create_triangular()
    np.testing.assert_allclose(spsolve_triangular(A, np.asfortranarray(B)), spsolve_triangular(A, B))


def test_empty():
    assert spsolve_triangular(sp.csr_array((0, 0)), np.zeros(0)).shape == (0,)
    A, b, _ = create_triangular(n=10)
    assert spsolve_triangular(A, np.zeros((10, 0))).shape == (10, 0)
