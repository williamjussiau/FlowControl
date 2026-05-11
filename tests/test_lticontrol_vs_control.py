"""Comparison tests: local lticontrol implementations vs python-control equivalents.

Each test checks that a local wrapper produces the same result as the library.
Passing tests mean the local function can be replaced by the library call shown in
the comment above each test class.
"""

import control
import numpy as np
import pytest

from utils.lticontrol import lqg_regulator, norm

# ---------------------------------------------------------------------------
# Test systems
# ---------------------------------------------------------------------------


def _siso_first_order():
    """Stable first-order SISO, no feed-through."""
    return control.ss([[-2.0]], [[1.0]], [[3.0]], [[0.0]])


def _siso_second_order():
    """Stable second-order SISO."""
    A = np.array([[-1.0, -2.0], [1.0, 0.0]])
    B = np.array([[1.0], [0.0]])
    C = np.array([[0.0, 1.0]])
    D = np.array([[0.0]])
    return control.ss(A, B, C, D)


def _mimo_2x2():
    """Stable 2x2 MIMO."""
    A = np.diag([-1.0, -2.0, -3.0])
    B = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    C = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    D = np.zeros((2, 2))
    return control.ss(A, B, C, D)


def _siso_with_feedthrough():
    """Stable SISO with non-zero D (H2 norm → inf)."""
    return control.ss([[-1.0]], [[1.0]], [[1.0]], [[2.0]])


STABLE_SYSTEMS = [_siso_first_order(), _siso_second_order(), _mimo_2x2()]


# ---------------------------------------------------------------------------
# H2 norm: norm(G, p=2)  →  control.norm(G, 2)
# ---------------------------------------------------------------------------


class TestH2Norm:
    # Replacement: control.norm(G, 2)

    @pytest.mark.parametrize("G", STABLE_SYSTEMS)
    def test_stable_system(self, G):
        assert np.isclose(norm(G, p=2), control.norm(G, 2), rtol=1e-6)

    def test_nonzero_D_returns_inf(self):
        G = _siso_with_feedthrough()
        assert norm(G, p=2) == np.inf
        assert control.norm(G, 2) == np.inf

    def test_unstable_returns_inf(self):
        G = control.ss([[1.0]], [[1.0]], [[1.0]], [[0.0]])
        assert norm(G, p=2) == np.inf
        assert control.norm(G, 2) == np.inf


# ---------------------------------------------------------------------------
# H-inf norm: norm(G, p=inf)  →  control.linfnorm(G)[0]
# ---------------------------------------------------------------------------


class TestHinfNorm:
    # Replacement: control.linfnorm(G)[0]
    # Note: control.norm(G, np.inf) raises ControlArgument in 0.10.1 — use linfnorm.

    @pytest.mark.parametrize("G", STABLE_SYSTEMS)
    def test_stable_system(self, G):
        local = norm(G, p=np.inf)
        library, _ = control.linfnorm(G)
        assert np.isclose(local, library, rtol=1e-4)

    def test_unstable_returns_inf(self):
        G = control.ss([[1.0]], [[1.0]], [[1.0]], [[0.0]])
        assert norm(G, p=np.inf) == np.inf


# ---------------------------------------------------------------------------
# Kalman gain in lqg_regulator  →  control.lqe
#
# Sign convention: lqg_regulator returns L = -L_kalman (observer gain defined
# for the formulation x_dot = (A + LC)x + ..., so L absorbs the minus sign).
# control.lqe returns L_kalman = P C^T Rv^{-1} > 0.
# Expected relation: L_ours == -control.lqe(...)[0]
# ---------------------------------------------------------------------------


class TestKalmanGain:
    # Replacement for the Kalman step:
    #   L_kalman, _, _ = control.lqe(A, np.eye(n), C, Qw * np.eye(n), Rv * np.eye(p))
    #   L = -L_kalman

    @pytest.fixture
    def plant(self):
        A = np.array([[-1.0, -2.0], [1.0, 0.0]])
        B = np.array([[1.0], [0.0]])
        C = np.array([[0.0, 1.0]])
        D = np.array([[0.0]])
        return control.ss(A, B, C, D)

    @pytest.mark.parametrize(
        "Qw, Rv",
        [
            (1.0, 1.0),
            (10.0, 0.1),
            (0.01, 5.0),
            (100.0, 100.0),
        ],
    )
    def test_kalman_gain_sign_and_value(self, plant, Qw, Rv):
        A, _, C, _ = plant.A, plant.B, plant.C, plant.D
        n, p = A.shape[0], C.shape[0]

        _, _, L = lqg_regulator(plant, Qx=1.0, Ru=1.0, Qw=Qw, Rv=Rv)

        L_lib, _, _ = control.lqe(A, np.eye(n), C, Qw * np.eye(n), Rv * np.eye(p))
        assert np.allclose(L, -L_lib, atol=1e-8), (
            f"Qw={Qw}, Rv={Rv}: L_ours={L.T}, -L_lqe={(-L_lib).T}"
        )

    def test_lqr_gain_unchanged(self, plant):
        """Spot-check that F (LQR gain) from lqg_regulator matches control.lqr directly."""
        A, B, C, D = plant.A, plant.B, plant.C, plant.D
        n, m = A.shape[0], B.shape[1]
        Qx, Ru = 1.0, 1.0

        _, F, _ = lqg_regulator(plant, Qx=Qx, Ru=Ru, Qw=1.0, Rv=1.0)
        F_lib = -np.array(control.lqr(A, B, Qx * np.eye(n), Ru * np.eye(m))[0])
        assert np.allclose(F, F_lib, atol=1e-8)
