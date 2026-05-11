"""Tests for utils.optim_algs: optimizer wrappers and NM helpers.

Requires optional dependencies (mpi4py, smt, blackbox_opt).
If any dependency is missing the entire module is skipped.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("utils.optim_algs")

from utils.optim_algs import (  # noqa: E402
    construct_simplex,
    minimize,
    nm_select_evaluated_points,
    optimizer_check_options,
    optimizer_default_options,
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


# ── minimize ─────────────────────────────────────────────────────────────────


class TestMinimize:
    @pytest.mark.parametrize("alg", ["nm", "cobyla", "bfgs", "slsqp"])
    def test_scipy_methods_find_quadratic_minimum(self, alg):
        """All scipy-backed algorithms must converge on a simple quadratic."""
        costfun = lambda x: float((x[0] - 1.0) ** 2 + (x[1] + 0.5) ** 2)
        x0 = np.array([0.0, 0.0])
        res = minimize(costfun, x0, alg=alg, options={}, verbose=False)
        assert_allclose(res.x, [1.0, -0.5], atol=1e-3)

    def test_unknown_alg_raises(self):
        with pytest.raises(ValueError, match="Unknown optimization algorithm"):
            minimize(lambda x: 0.0, np.zeros(2), alg="unknown", options={})
