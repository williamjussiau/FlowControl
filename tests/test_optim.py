"""Tests for utils.optim pure-Python helpers.

Requires all optional optimization dependencies (mpi4py, smt, blackbox_opt).
If any dependency is missing the entire module is skipped.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("utils.optim")

from utils.optim import (  # noqa: E402
    compute_control_cost,
    compute_signal_cost,
    cummin,
    sobol_sample,
    write_optim_csv,
    write_results,
)


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

    def test_duplicate_minimum_index_points_to_first_occurrence(self):
        """When the same minimum value appears twice, index must point to the first."""
        y = np.array([[2.0], [3.0], [2.0], [1.0]])
        _, idx = cummin(y)
        # cummin = [2, 2, 2, 1]; the value 2 first appears at index 0
        assert idx[0] == 0
        assert idx[1] == 0
        assert idx[2] == 0
        assert idx[3] == 3


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


# ── write_results ─────────────────────────────────────────────────────────────


class TestWriteResults:
    def test_csv_files_created(self):
        x_data = np.array([[0.0, 1.0], [1.0, 2.0], [0.5, 0.5]])
        y_data = [3.0, 1.0, 2.0]
        with tempfile.TemporaryDirectory() as tmpdir:
            write_results(x_data, y_data, tmpdir)
            assert (Path(tmpdir) / "J_costfun.csv").exists()
            assert (Path(tmpdir) / "J_costfun_cummin.csv").exists()

    def test_cummin_csv_is_non_increasing(self):
        x_data = np.array([[0.0], [1.0], [0.5], [2.0]])
        y_data = [4.0, 2.0, 3.0, 1.0]
        with tempfile.TemporaryDirectory() as tmpdir:
            write_results(x_data, y_data, tmpdir)
            df = pd.read_csv(Path(tmpdir) / "J_costfun_cummin.csv")
            assert np.all(np.diff(df["J"].values) <= 0)

    def test_all_evaluations_csv_row_count(self):
        x_data = np.array([[float(i)] for i in range(5)])
        y_data = [5.0, 3.0, 4.0, 1.0, 2.0]
        with tempfile.TemporaryDirectory() as tmpdir:
            write_results(x_data, y_data, tmpdir)
            df = pd.read_csv(Path(tmpdir) / "J_costfun.csv")
            assert len(df) == 5


# ── write_optim_csv ───────────────────────────────────────────────────────────


class TestWriteOptimCsv:
    def test_file_created_with_correct_name(self):
        df = pd.DataFrame({"time": [0.0, 0.1], "u": [1.0, 2.0]})
        with tempfile.TemporaryDirectory() as tmpdir:
            ts_dir = Path(tmpdir) / "timeseries"
            ts_dir.mkdir()
            write_optim_csv(df, tmpdir, diverged=False, iteration=3)
            assert (ts_dir / "timeseries_iter_0003.csv").exists()

    def test_diverged_suffix_appended(self):
        df = pd.DataFrame({"time": [0.0], "u": [0.0]})
        with tempfile.TemporaryDirectory() as tmpdir:
            ts_dir = Path(tmpdir) / "timeseries"
            ts_dir.mkdir()
            write_optim_csv(df, tmpdir, diverged=True, iteration=7)
            assert (ts_dir / "timeseries_iter_0007_DIVERGED.csv").exists()

    def test_no_diverged_suffix_when_false(self):
        df = pd.DataFrame({"time": [0.0]})
        with tempfile.TemporaryDirectory() as tmpdir:
            ts_dir = Path(tmpdir) / "timeseries"
            ts_dir.mkdir()
            write_optim_csv(df, tmpdir, diverged=False, iteration=42)
            files = list(ts_dir.iterdir())
            assert len(files) == 1
            assert "DIVERGED" not in files[0].name

    def test_iteration_zero_padded_to_four_digits(self):
        df = pd.DataFrame({"time": [0.0]})
        with tempfile.TemporaryDirectory() as tmpdir:
            ts_dir = Path(tmpdir) / "timeseries"
            ts_dir.mkdir()
            write_optim_csv(df, tmpdir, diverged=False, iteration=1)
            assert (ts_dir / "timeseries_iter_0001.csv").exists()


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
