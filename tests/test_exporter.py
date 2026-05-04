"""Tests for flowcontrol.exporter.FlowExporter — log/timeseries logic only.

The export_xdmf path (which needs actual dolfin functions and file I/O) is not
tested here; it is exercised by the integration tests.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from flowcontrol.exporter import FlowExporter
from flowcontrol.flowfield import FlowFieldCollection, SimPaths


def _make_paths(tmpdir: Path) -> SimPaths:
    """Build a minimal SimPaths with all fields pointing inside tmpdir."""
    return SimPaths(
        U0=tmpdir / "U0.xdmf",
        P0=tmpdir / "P0.xdmf",
        U=tmpdir / "U.xdmf",
        P=tmpdir / "P.xdmf",
        Uprev=tmpdir / "Uprev.xdmf",
        U_restart=tmpdir / "U_restart.xdmf",
        Uprev_restart=tmpdir / "Uprev_restart.xdmf",
        P_restart=tmpdir / "P_restart.xdmf",
        timeseries=tmpdir / "timeseries.csv",
        metadata=tmpdir / "metadata.json",
        steady_meta=tmpdir / "steady_meta.json",
        mesh=tmpdir / "mesh.xdmf",
    )


@pytest.fixture
def exporter(tmp_path):
    paths = _make_paths(tmp_path)
    fields = FlowFieldCollection()
    return FlowExporter(paths=paths, fields=fields, V=None, P=None)


# ── log_ic ────────────────────────────────────────────────────────────────────


class TestLogIc:
    def test_appends_one_record(self, exporter):
        exporter.log_ic(t=0.0, y_meas=np.array([1.0, 2.0]), dE=0.5)
        df = exporter.to_dataframe()
        assert len(df) == 1

    def test_record_fields(self, exporter):
        exporter.log_ic(t=0.5, y_meas=np.array([3.0]), dE=0.1)
        df = exporter.to_dataframe()
        assert df.loc[0, "time"] == pytest.approx(0.5)
        assert df.loc[0, "dE"] == pytest.approx(0.1)
        assert df.loc[0, "runtime"] == pytest.approx(0.0)
        assert df.loc[0, "y_meas_1"] == pytest.approx(3.0)


# ── log ───────────────────────────────────────────────────────────────────────


class TestLog:
    def test_appends_records(self, exporter):
        for i in range(3):
            exporter.log(
                u_ctrl=np.array([float(i)]),
                y_meas=np.array([float(i) * 2]),
                dE=float(i),
                t=float(i) * 0.01,
                runtime=0.01,
            )
        df = exporter.to_dataframe()
        assert len(df) == 3

    def test_column_names(self, exporter):
        exporter.log(
            u_ctrl=np.array([1.0, 2.0]),
            y_meas=np.array([0.5]),
            dE=0.0,
            t=0.1,
            runtime=0.02,
        )
        df = exporter.to_dataframe()
        assert "u_ctrl_1" in df.columns
        assert "u_ctrl_2" in df.columns
        assert "y_meas_1" in df.columns

    def test_values_stored_correctly(self, exporter):
        exporter.log(
            u_ctrl=np.array([7.0]),
            y_meas=np.array([3.5]),
            dE=1.2,
            t=0.25,
            runtime=0.05,
        )
        df = exporter.to_dataframe()
        row = df.iloc[0]
        assert row["u_ctrl_1"] == pytest.approx(7.0)
        assert row["y_meas_1"] == pytest.approx(3.5)
        assert row["dE"] == pytest.approx(1.2)
        assert row["time"] == pytest.approx(0.25)


# ── timeseries shape after log_ic + N×log ────────────────────────────────────


class TestTimeseriesShape:
    def test_correct_row_count(self, exporter):
        exporter.log_ic(t=0.0, y_meas=np.array([0.0]), dE=0.0)
        for i in range(5):
            exporter.log(
                u_ctrl=np.array([0.0]),
                y_meas=np.array([0.0]),
                dE=0.0,
                t=float(i + 1) * 0.01,
                runtime=0.01,
            )
        df = exporter.to_dataframe()
        assert len(df) == 6  # 1 IC + 5 steps


# ── write_timeseries ──────────────────────────────────────────────────────────


class TestWriteTimeseries:
    def test_csv_written(self, tmp_path):
        paths = _make_paths(tmp_path)
        exporter = FlowExporter(paths=paths, fields=FlowFieldCollection(), V=None, P=None)
        exporter.log_ic(t=0.0, y_meas=np.array([1.0]), dE=0.5)
        exporter.write_timeseries()
        assert paths.timeseries.exists()

    def test_csv_has_correct_columns(self, tmp_path):
        paths = _make_paths(tmp_path)
        exporter = FlowExporter(paths=paths, fields=FlowFieldCollection(), V=None, P=None)
        exporter.log_ic(t=0.0, y_meas=np.array([1.0, 2.0]), dE=0.1)
        exporter.write_timeseries()
        df = pd.read_csv(paths.timeseries)
        assert "time" in df.columns
        assert "y_meas_1" in df.columns
        assert "y_meas_2" in df.columns


# ── reset ─────────────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_records(self, exporter):
        exporter.log_ic(t=0.0, y_meas=np.array([1.0]), dE=0.5)
        exporter.reset()
        assert len(exporter.to_dataframe()) == 0

    def test_reset_clears_checkpoint_count(self, exporter):
        exporter._checkpoints_written = 5
        exporter.reset()
        assert exporter._checkpoints_written == 0
