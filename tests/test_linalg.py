"""Tests for utils.linalg frequency-response computation.

Uses a small synthetic SISO system so no FEniCS is needed.
The analytical transfer function is:
    H(w) = C @ inv(jwQ - A) @ B  =  1/(jw+1) + 1/(jw+2)
for A = diag(-1, -2), Q = I, B = [[1],[1]], C = [[1, 1]].
"""

import numpy as np
import pytest
import scipy.sparse as spr
from numpy.testing import assert_allclose

from utils.linalg import get_frequency_response_parallel, get_frequency_response_sequential


@pytest.fixture
def small_system():
    """2-state SISO system with analytic frequency response."""
    A = spr.diags([-1.0, -2.0], format="csc")
    Q = spr.eye(2, format="csc")
    B = np.array([[1.0], [1.0]])
    C = np.array([[1.0, 1.0]])
    ww = np.array([0.1, 1.0, 5.0, 20.0])
    return A, Q, B, C, ww


def _analytic_H(ww: np.ndarray) -> np.ndarray:
    """H(w) = 1/(jw+1) + 1/(jw+2), shape (1, 1, nw)."""
    H = np.zeros((1, 1, len(ww)), dtype=complex)
    for k, w in enumerate(ww):
        H[0, 0, k] = 1.0 / (1j * w + 1.0) + 1.0 / (1j * w + 2.0)
    return H


class TestFrequencyResponse:
    def test_sequential_matches_analytic(self, small_system):
        A, Q, B, C, ww = small_system
        H, ww_out = get_frequency_response_sequential(A, B, C, Q, ww, verbose=False)
        assert_allclose(H, _analytic_H(ww), atol=1e-12)
        assert_allclose(ww_out, ww)

    def test_parallel_matches_sequential(self, small_system):
        A, Q, B, C, ww = small_system
        H_seq, _ = get_frequency_response_sequential(A, B, C, Q, ww, verbose=False)
        H_par, _ = get_frequency_response_parallel(A, B, C, Q, ww, verbose=False, n_jobs=2)
        assert_allclose(H_par, H_seq, atol=1e-14)

    def test_output_shape(self, small_system):
        A, Q, B, C, ww = small_system
        H, _ = get_frequency_response_sequential(A, B, C, Q, ww, verbose=False)
        ny, nu, nw = C.shape[0], B.shape[1], len(ww)
        assert H.shape == (ny, nu, nw)

    def test_mimo_shape(self):
        """MIMO: ny=2, nu=3, nw=4."""
        n = 4
        A = spr.diags([-float(i + 1) for i in range(n)], format="csc")
        Q = spr.eye(n, format="csc")
        B = np.random.default_rng(0).standard_normal((n, 3))
        C = np.random.default_rng(1).standard_normal((2, n))
        ww = np.linspace(0.1, 10.0, 4)
        H, _ = get_frequency_response_sequential(A, B, C, Q, ww, verbose=False)
        assert H.shape == (2, 3, 4)
        assert np.all(np.isfinite(H))
