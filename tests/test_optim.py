"""Tests for utils.optim pure-Python helpers.

Requires all optional optimization dependencies (mpi4py, smt, blackbox_opt).
If any dependency is missing the entire module is skipped.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("utils.optim")

from utils.optim import (  # noqa: E402
    compute_control_cost,
    compute_signal_cost,
    construct_simplex,
    cummin,
    nm_select_evaluated_points,
    optimizer_check_options,
    optimizer_default_options,
    sobol_sample,
)


# ── construct_simplex ─────────────────────────────────────────────────────────


class TestConstructSimplex:
    def test_shape_rectangular(self):
        x0 = np.array([1.0, 2.0, 3.0])
        s = construct_simplex(x0, rectangular=True)
        assert s.shape == (4, 3)

    def test_shape_regular(self):
        x0 = np.array([0.0, 0.0])
        s = construct_simplex(x0, rectangular=False)
        assert s.shape == (3, 2)

    def test_rectangular_first_row_is_x0(self):
        x0 = np.array([1.0, -2.0])
        s = construct_simplex(x0, rectangular=True, edgelen=0.5)
        assert_allclose(s[0], x0)

    def test_rectangular_other_rows_offset_by_edgelen(self):
        x0 = np.array([1.0, 2.0])
        edgelen = 0.5
        s = construct_simplex(x0, rectangular=True, edgelen=edgelen)
        for i in range(len(x0)):
            expected = x0.copy()
            expected[i] += edgelen
            assert_allclose(s[i + 1], expected)

    def test_rectangular_per_dim_edgelen(self):
        x0 = np.zeros(3)
        edgelen = [1.0, 2.0, 3.0]
        s = construct_simplex(x0, rectangular=True, edgelen=edgelen)
        for i in range(3):
            assert_allclose(s[i + 1, i], edgelen[i])

    def test_regular_centroid_is_x0_with_unit_edgelen(self):
        x0 = np.array([3.0, -1.0, 2.0])
        s = construct_simplex(x0, rectangular=False, edgelen=1.0)
        assert_allclose(s.mean(axis=0), x0, atol=1e-12)

    def test_1d_input(self):
        x0 = np.array([5.0])
        s = construct_simplex(x0, rectangular=True, edgelen=1.0)
        assert s.shape == (2, 1)
        assert_allclose(s[0], [5.0])
        assert_allclose(s[1], [6.0])


# ── cummin ────────────────────────────────────────────────────────────────────


class TestCummin:
    def test_monotone_non_increasing(self):
        y = np.array([[3.0], [1.0], [2.0], [0.5], [4.0]])
        y_cm, _ = cummin(y)
        assert np.all(np.diff(y_cm[:, 0]) <= 0)

    def test_values_correct(self):
        y = np.array([[3.0], [1.0], [2.0], [0.5], [4.0]])
        y_cm, _ = cummin(y)
        assert_allclose(y_cm[:, 0], [3.0, 1.0, 1.0, 0.5, 0.5])

    def test_indices_point_to_first_occurrence(self):
        y = np.array([[3.0], [1.0], [2.0], [0.5], [4.0]])
        _, idx = cummin(y)
        # cummin values: [3, 1, 1, 0.5, 0.5] → first achieved at indices [0, 1, 1, 3, 3]
        assert_allclose(idx, [0, 1, 1, 3, 3])

    def test_return_index_false(self):
        y = np.array([[2.0], [1.0], [3.0]])
        result = cummin(y, return_index=False)
        assert not isinstance(result, tuple)
        assert_allclose(result[:, 0], [2.0, 1.0, 1.0])

    def test_already_sorted(self):
        y = np.array([[1.0], [2.0], [3.0]])
        y_cm, idx = cummin(y)
        assert_allclose(y_cm[:, 0], [1.0, 1.0, 1.0])
        assert_allclose(idx, [0, 0, 0])

    def test_strictly_decreasing(self):
        y = np.array([[5.0], [3.0], [1.0]])
        y_cm, idx = cummin(y)
        assert_allclose(y_cm[:, 0], [5.0, 3.0, 1.0])
        assert_allclose(idx, [0, 1, 2])


# ── nm_select_evaluated_points ────────────────────────────────────────────────


class TestNmSelectEvaluatedPoints:
    def test_basic_retrieval(self):
        x_best = np.array([[1.0, 2.0], [3.0, 4.0]])
        x_all = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        y_all = [10.0, 5.0]
        x_good, y_good = nm_select_evaluated_points(x_best, x_all, y_all)
        assert len(x_good) == 2
        assert_allclose(y_good, [10.0, 5.0])

    def test_deduplication(self):
        x_best = np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 0.0]])
        x_all = [np.array([1.0, 0.0]), np.array([2.0, 0.0])]
        y_all = [3.0, 7.0]
        x_good, y_good = nm_select_evaluated_points(x_best, x_all, y_all)
        assert len(x_good) == 2

    def test_missing_point_raises(self):
        x_best = np.array([[9.0, 9.0]])
        x_all = [np.array([1.0, 2.0])]
        y_all = [1.0]
        with pytest.raises(ValueError, match="not found in x_all"):
            nm_select_evaluated_points(x_best, x_all, y_all)

    def test_order_preserved(self):
        x_best = np.array([[2.0], [1.0], [3.0]])
        x_all = [np.array([1.0]), np.array([2.0]), np.array([3.0])]
        y_all = [10.0, 20.0, 30.0]
        x_good, y_good = nm_select_evaluated_points(x_best, x_all, y_all)
        assert_allclose(x_good[0], [2.0])
        assert_allclose(x_good[1], [1.0])
        assert_allclose(x_good[2], [3.0])


# ── optimizer_default_options / optimizer_check_options ───────────────────────


class TestOptimizerOptions:
    @pytest.mark.parametrize("alg", ["nm", "cobyla", "bfgs", "slsqp", "dfo", "bo"])
    def test_default_options_returns_dict(self, alg):
        opts = optimizer_default_options(alg)
        assert isinstance(opts, dict)
        assert len(opts) > 0

    def test_default_options_unknown_alg_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            optimizer_default_options("nonexistent")

    def test_default_options_returns_copy(self):
        opts1 = optimizer_default_options("nm")
        opts2 = optimizer_default_options("nm")
        opts1["maxfev"] = 99999
        assert opts2["maxfev"] != 99999

    def test_check_options_override_known_key(self):
        defaults = {"maxfev": 100, "disp": False, "xatol": 1e-4}
        result = optimizer_check_options(defaults, {"maxfev": 50})
        assert result["maxfev"] == 50

    def test_check_options_unknown_keys_dropped(self):
        defaults = {"maxfev": 100}
        result = optimizer_check_options(defaults, {"maxfev": 50, "unknown_key": 99})
        assert "unknown_key" not in result

    def test_check_options_missing_keys_use_defaults(self):
        defaults = {"maxfev": 100, "disp": False}
        result = optimizer_check_options(defaults, {"maxfev": 50})
        assert result["disp"] is False

    def test_check_options_empty_user_dict_returns_defaults(self):
        defaults = {"a": 1, "b": 2}
        result = optimizer_check_options(defaults, {})
        assert result == defaults


# ── compute_signal_cost ───────────────────────────────────────────────────────


class TestComputeSignalCost:
    @pytest.fixture
    def signal(self):
        return pd.Series([1.0, 2.0, 3.0, 4.0])

    def test_integral_sums_and_scales(self, signal):
        Tnorm = 0.5
        result = compute_signal_cost(signal, Tnorm, criterion="integral")
        assert result == pytest.approx(sum(signal) * Tnorm)

    def test_terminal_returns_last_value(self, signal):
        result = compute_signal_cost(signal, Tnorm=1.0, criterion="terminal")
        assert result == pytest.approx(4.0)

    def test_scaling_applied_to_integral(self, signal):
        result = compute_signal_cost(
            signal, Tnorm=1.0, criterion="integral", scaling=lambda x: x**2
        )
        assert result == pytest.approx(sum(x**2 for x in signal))

    def test_scaling_applied_to_terminal(self, signal):
        result = compute_signal_cost(
            signal, Tnorm=1.0, criterion="terminal", scaling=lambda x: x * 2
        )
        assert result == pytest.approx(8.0)

    def test_unknown_criterion_raises(self, signal):
        with pytest.raises(ValueError, match="Unknown criterion"):
            compute_signal_cost(signal, Tnorm=1.0, criterion="rms")

    def test_returns_float(self, signal):
        result = compute_signal_cost(signal, Tnorm=1.0, criterion="integral")
        assert isinstance(result, float)


# ── compute_control_cost ──────────────────────────────────────────────────────


class TestComputeControlCost:
    def test_series_scalar_cost(self):
        u = pd.Series([1.0, 2.0, 3.0])
        Tnorm = 0.1
        assert compute_control_cost(u, Tnorm) == pytest.approx((1 + 4 + 9) * Tnorm)

    def test_dataframe_sums_all_channels(self):
        u = pd.DataFrame({"u1": [1.0, 0.0], "u2": [0.0, 2.0]})
        Tnorm = 1.0
        assert compute_control_cost(u, Tnorm) == pytest.approx(1.0 + 4.0)

    def test_zero_input_gives_zero(self):
        u = pd.Series([0.0, 0.0, 0.0])
        assert compute_control_cost(u, Tnorm=1.0) == pytest.approx(0.0)

    def test_returns_float(self):
        u = pd.Series([1.0, 2.0])
        assert isinstance(compute_control_cost(u, 1.0), float)

    def test_tnorm_scales_result(self):
        u = pd.Series([1.0, 1.0])
        assert compute_control_cost(u, Tnorm=3.0) == pytest.approx(
            compute_control_cost(u, Tnorm=1.0) * 3.0
        )


# ── sobol_sample ──────────────────────────────────────────────────────────────


class TestSobolSample:
    def test_shape(self):
        X = sobol_sample(ndim=3, npt=16)
        assert X.shape == (16, 3)

    def test_values_in_unit_hypercube(self):
        X = sobol_sample(ndim=4, npt=32)
        assert np.all(X >= 0.0)
        assert np.all(X <= 1.0)

    def test_xlimits_scales_samples(self):
        xlimits = np.array([[-1.0, 1.0], [0.0, 10.0]])
        X = sobol_sample(ndim=2, npt=16, xlimits=xlimits)
        assert np.all(X[:, 0] >= -1.0) and np.all(X[:, 0] <= 1.0)
        assert np.all(X[:, 1] >= 0.0) and np.all(X[:, 1] <= 10.0)

    def test_xlimits_transposed_shape_accepted(self):
        xlimits = np.array([[-1.0, 0.0, 2.0], [1.0, 5.0, 4.0]])  # shape (2, ndim)
        X = sobol_sample(ndim=3, npt=8, xlimits=xlimits)
        assert X.shape == (8, 3)

    def test_wrong_xlimits_shape_raises(self):
        with pytest.raises(ValueError, match="xlimits has wrong shape"):
            sobol_sample(ndim=3, npt=8, xlimits=np.ones((4, 2)))

    def test_different_seeds_give_different_samples(self):
        X1 = sobol_sample(ndim=2, npt=16, seed=0)
        X2 = sobol_sample(ndim=2, npt=16, seed=1)
        assert not np.allclose(X1, X2)
