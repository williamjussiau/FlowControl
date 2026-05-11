"""Tests for flowcontrol.controller.Controller.

Covers:
- Initialization (direct, from_matrices, initial state)
- Binary operations (__add__, __mul__, inv)
- step(): output correctness, state advancement, scalar input, dt caching
"""

from pathlib import Path

import control
import numpy as np
import pytest
from numpy.testing import assert_allclose

from flowcontrol.controller import Controller

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def siso():
    """Stable SISO first-order system: dx/dt = -x + u,  y = x."""
    A = np.array([[-1.0]])
    B = np.array([[1.0]])
    C = np.array([[1.0]])
    D = np.array([[0.0]])
    return A, B, C, D


@pytest.fixture
def mimo():
    """3-state, 2-input, 3-output system (from __main__ in controller.py)."""
    A = np.array([[1.0, 1.0, 1.0], [0.2, -1.0, 0.0], [0.0, 1.0, 1.0]])
    B = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    C = 0.5 * np.eye(3)
    D = np.zeros((3, 2))
    return A, B, C, D


# ── Initialization ────────────────────────────────────────────────────────────


class TestInit:
    def test_state_is_zero_by_default(self, siso):
        A, B, C, D = siso
        K = Controller(A, B, C, D)
        assert_allclose(K.x, np.zeros(1))

    def test_reset_zeroes_state(self, siso):
        A, B, C, D = siso
        K = Controller(A, B, C, D, x0=np.array([3.7]))
        K.reset()
        assert_allclose(K.x, np.zeros(1))

    def test_reset_after_steps_zeroes_state(self, siso):
        A, B, C, D = siso
        K = Controller(A, B, C, D)
        for _ in range(5):
            K.step(np.array([1.0]), dt=0.01)
        assert not np.allclose(K.x, np.zeros(1))
        K.reset()
        assert_allclose(K.x, np.zeros(1))

    def test_custom_initial_state(self, siso):
        A, B, C, D = siso
        x0 = np.array([2.5])
        K = Controller(A, B, C, D, x0=x0)
        assert_allclose(K.x, x0)

    def test_file_is_none_by_default(self, siso):
        A, B, C, D = siso
        K = Controller(A, B, C, D)
        assert K.file is None

    def test_dimensions_inherited(self, mimo):
        A, B, C, D = mimo
        K = Controller(A, B, C, D)
        assert K.nstates == 3
        assert K.ninputs == 2
        assert K.noutputs == 3

    def test_from_matrices_matches_direct(self, mimo):
        A, B, C, D = mimo
        K1 = Controller(A, B, C, D)
        K2 = Controller.from_matrices(A, B, C, D)
        assert_allclose(K1.A, K2.A)
        assert_allclose(K1.B, K2.B)
        assert_allclose(K1.C, K2.C)
        assert_allclose(K1.D, K2.D)
        assert_allclose(K1.x, K2.x)

    def test_from_matrices_with_x0(self, siso):
        A, B, C, D = siso
        x0 = np.array([3.0])
        K = Controller.from_matrices(A, B, C, D, x0=x0)
        assert_allclose(K.x, x0)


# ── Binary operations ─────────────────────────────────────────────────────────


class TestOperations:
    def test_add_returns_controller(self, siso):
        A, B, C, D = siso
        K1 = Controller(A, B, C, D)
        K2 = Controller(A, B, C, D)
        assert isinstance(K1 + K2, Controller)

    def test_add_concatenates_states(self, siso):
        A, B, C, D = siso
        K1 = Controller(A, B, C, D, x0=np.array([1.0]))
        K2 = Controller(A, B, C, D, x0=np.array([2.0]))
        result = K1 + K2
        assert_allclose(result.x, np.array([1.0, 2.0]))

    def test_binary_op_file_always_none(self, siso):
        """Derived controllers have no single file origin: file is always None."""
        A, B, C, D = siso
        p = Path("some.mat")
        K1 = Controller(A, B, C, D, file=p)
        K2 = Controller(A, B, C, D, file=p)
        assert (K1 + K2).file is None
        assert (K1 * K2).file is None

    def test_mul_returns_controller(self, siso):
        A, B, C, D = siso
        K = Controller(A, B, C, D)
        assert isinstance(K * K, Controller)

    def test_inv_roundtrip(self, siso):
        """K * K.inv() should be close to identity (D must be invertible)."""
        A, B, C, D = siso
        D_inv = np.array([[2.0]])  # non-zero D
        K = Controller(A, B, C, D_inv)
        result = K * K.inv()
        assert_allclose(result.D, np.eye(1), atol=1e-10)


# ── step() ────────────────────────────────────────────────────────────────────


class TestStep:
    def _forced_response_ref(self, A, B, C, D, x0, y, dt):
        """Reference output using control.forced_response (old implementation)."""
        sys_ref = control.StateSpace(A, B, C, D)
        U = np.column_stack([y, y])
        _, yout, _ = control.forced_response(
            sys_ref, T=[0, dt], U=U, X0=x0.copy(), interpolate=False, return_x=True
        )
        return np.atleast_2d(yout)[:, 0]

    def test_siso_output_matches_forced_response(self, siso):
        A, B, C, D = siso
        dt, y, x0 = 0.1, np.array([1.0]), np.array([0.5])
        K = Controller(A, B, C, D, x0=x0.copy())
        u = K.step(y, dt)
        u_ref = self._forced_response_ref(A, B, C, D, x0, y, dt)
        assert_allclose(u, u_ref, atol=1e-12)

    def test_mimo_output_matches_forced_response(self, mimo):
        A, B, C, D = mimo
        dt, y, x0 = 0.05, np.array([1.2, -0.8]), np.array([1.0, 2.0, 3.0])
        K = Controller(A, B, C, D, x0=x0.copy())
        u = K.step(y, dt)
        u_ref = self._forced_response_ref(A, B, C, D, x0, y, dt)
        assert_allclose(u, u_ref, atol=1e-10)

    def test_state_advances_after_step(self, siso):
        A, B, C, D = siso
        K = Controller(A, B, C, D)
        K.step(np.array([1.0]), dt=0.1)
        assert not np.allclose(K.x, np.zeros(1))

    def test_scalar_y_accepted(self, siso):
        """step() must accept a bare float without raising."""
        A, B, C, D = siso
        K = Controller(A, B, C, D)
        u = K.step(1.0, dt=0.1)
        assert u.shape == (1,)

    def test_multistep_state_matches_zoh_recursion(self, siso):
        """After N steps, internal state must match manual ZOH recursion."""
        A, B, C, D = siso
        dt, y, n = 0.1, np.array([1.0]), 5
        K = Controller(A, B, C, D)

        for _ in range(n):
            K.step(y, dt)

        sysd = control.c2d(control.StateSpace(A, B, C, D), dt, method="zoh")
        x = np.zeros(1)
        for _ in range(n):
            x = sysd.A @ x + sysd.B @ y

        assert_allclose(K.x, x, atol=1e-12)

    def test_dt_change_triggers_rediscretization(self, siso):
        """Switching dt mid-simulation must give the same result as starting fresh."""
        A, B, C, D = siso
        y = np.array([1.0])

        K_switched = Controller(A, B, C, D)
        K_switched.step(y, dt=0.1)  # warm up with dt=0.1
        x_mid = K_switched.x.copy()  # state after first step, before dt change
        u_switched = K_switched.step(y, dt=0.2)  # switch to dt=0.2

        # Reference: fresh controller initialized to state after the first step,
        # then stepped once at dt=0.2 only — must match despite the dt switch.
        K_ref = Controller(A, B, C, D, x0=x_mid)
        u_ref = K_ref.step(y, dt=0.2)

        assert_allclose(u_switched, u_ref, atol=1e-12)

    def test_same_dt_reuses_cache(self, siso):
        """Repeated steps with the same dt must not change the cached matrices."""
        A, B, C, D = siso
        dt, y = 0.1, np.array([1.0])
        K = Controller(A, B, C, D)
        K.step(y, dt)
        Ad_after_first = K._Ad.copy()
        K.step(y, dt)
        assert_allclose(K._Ad, Ad_after_first)
