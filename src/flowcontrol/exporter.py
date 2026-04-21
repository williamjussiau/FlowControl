"""Export helpers for flow simulation data.

FlowExporter handles all output: XDMF field snapshots and timeseries CSV.
It has no knowledge of meshes, BCs, or variational forms.

Timeseries are accumulated as a list of dicts (one per completed step) and
converted to a DataFrame only when writing. This avoids pre-allocation,
eliminates pandas row-assignment overhead, and ensures the log only contains
steps that actually completed.

Typical usage::

    exporter = FlowExporter(paths=fs.paths, fields=fs.fields, V=fs.V, P=fs.P)

    # Log the initial condition before any stepping
    exporter.log_ic(t=0.0, y_meas=y0, dE=dE0)

    # Inside the time loop
    exporter.log(u_ctrl=u_ctrl, y_meas=y_meas, dE=dE, t=t, runtime=rt)
    exporter.export_xdmf(u_n, u_nn, p_n, time=t, adjust_baseflow=1.0)
    exporter.write_timeseries()

    # Retrieve as DataFrame at any point
    df = exporter.to_dataframe()
"""

import json
import logging
from typing import Optional

import dolfin
import numpy as np
import pandas as pd
from numpy.typing import NDArray

import utils.utils_flowsolver as flu
from flowcontrol.flowfield import FlowFieldCollection, SimPaths

logger = logging.getLogger(__name__)


class FlowExporter:
    """Handles XDMF field export and timeseries CSV logging.

    Parameters
    ----------
    paths:
        Path dictionary as returned by ``FlowSolver._define_paths()``.
        Keys used: ``"U_restart"``, ``"Uprev_restart"``, ``"P_restart"``,
        ``"timeseries"``.
    fields:
        Shared FlowFieldCollection. ``fields.STEADY.u/p`` are read for
        base-flow reconstruction; ``fields.Usave``, ``fields.Usave_n``,
        and ``fields.Psave`` are created lazily on first export.
    V:
        Velocity function space (used to allocate Usave on first call).
    P:
        Pressure function space (used to allocate Psave on first call).
    """

    def __init__(
        self,
        paths: SimPaths,
        fields: FlowFieldCollection,
        V: dolfin.FunctionSpace,
        P: dolfin.FunctionSpace,
        Tstart: float = 0.0,
        dt: float = 0.0,
        save_every: int = 0,
    ) -> None:
        self.paths = paths
        self.fields = fields
        self.V = V
        self.P = P
        self._Tstart = Tstart
        self._dt = dt
        self._save_every = save_every
        self._records: list[dict] = []
        self._checkpoints_written: int = 0
        self._u_cols: Optional[list[str]] = None  # cached on first log() call
        self._y_cols: Optional[list[str]] = None  # cached on first log() call

    # ── Field export ──────────────────────────────────────────────────────────

    def export_xdmf(
        self,
        u_n: dolfin.Function,
        u_nn: dolfin.Function,
        p_n: dolfin.Function,
        time: float,
        append: bool = True,
        write_mesh: bool = False,
        adjust_baseflow: float = 0.0,
    ) -> None:
        """Write velocity and pressure snapshots to XDMF files.

        The exported fields can be the raw perturbation (``adjust_baseflow=0``)
        or the full flow (``adjust_baseflow=1``, i.e. perturbation + base flow).

        Parameters
        ----------
        u_n:
            Velocity perturbation at the previous time step.
        u_nn:
            Velocity perturbation two time steps back (used for restart).
        p_n:
            Pressure perturbation at the previous time step.
        time:
            Simulation time to embed in the XDMF file.
        append:
            Append to existing XDMF file (True) or overwrite (False).
        write_mesh:
            Whether to embed the mesh in the XDMF file.
        adjust_baseflow:
            Scalar multiplier for the base flow added to the perturbation.
            0 → perturbation only; 1 → full field.
        """
        # Lazy allocation of save buffers
        if self.fields.Usave is None:
            self.fields.Usave = dolfin.Function(self.V)
        if self.fields.Usave_n is None:
            self.fields.Usave_n = dolfin.Function(self.V)
        if self.fields.Psave is None:
            self.fields.Psave = dolfin.Function(self.P)

        pmbf = adjust_baseflow
        U0v = self.fields.U0.vector()[:]
        P0v = self.fields.P0.vector()[:]

        self.fields.Usave.vector()[:] = u_n.vector()[:] + pmbf * U0v
        self.fields.Usave_n.vector()[:] = u_nn.vector()[:] + pmbf * U0v
        self.fields.Psave.vector()[:] = p_n.vector()[:] + pmbf * P0v

        for vec in [self.fields.Usave, self.fields.Usave_n, self.fields.Psave]:
            vec.vector().apply("insert")

        self._checkpoints_written += 1
        logger.debug(
            f"Exporting fields at t={time:.4f} to {self.paths.U_restart.parent}"
        )

        flu.write_xdmf(
            filename=self.paths.U_restart,
            func=self.fields.Usave,
            name="U",
            time_step=time,
            append=append,
            write_mesh=write_mesh,
        )
        flu.write_xdmf(
            filename=self.paths.Uprev_restart,
            func=self.fields.Usave_n,
            name="U_n",
            time_step=time,
            append=append,
            write_mesh=write_mesh,
        )
        flu.write_xdmf(
            filename=self.paths.P_restart,
            func=self.fields.Psave,
            name="P",
            time_step=time,
            append=append,
            write_mesh=write_mesh,
        )

    # ── Timeseries ────────────────────────────────────────────────────────────

    def log_ic(
        self,
        t: float,
        y_meas: NDArray[np.float64],
        dE: float,
    ) -> None:
        """Append the initial-condition record (before any stepping).

        Parameters
        ----------
        t:
            Start time of the simulation.
        y_meas:
            Measurement at the initial condition.
        dE:
            Perturbation kinetic energy at the initial condition.
        """
        row: dict = {"time": t, "dE": dE, "runtime": 0.0}
        for i, v in enumerate(y_meas):
            row[f"y_meas_{i + 1}"] = float(v)
        self._records.append(row)

    def log(
        self,
        u_ctrl: NDArray[np.float64],
        y_meas: NDArray[np.float64],
        dE: float,
        t: float,
        runtime: float,
    ) -> None:
        """Append one record for the step that just completed.

        Each record holds the control that was applied, the measurement that
        resulted, the energy, the time, and the wall-clock cost — all
        belonging to the same completed step.

        Parameters
        ----------
        u_ctrl:
            Control input applied during this step.
        y_meas:
            Measurement vector produced by this step.
        dE:
            Perturbation kinetic energy at the end of this step.
        t:
            Simulation time at the end of this step.
        runtime:
            Wall-clock duration of this step (seconds).
        """
        if self._u_cols is None:
            self._u_cols = [f"u_ctrl_{i + 1}" for i in range(len(u_ctrl))]
            self._y_cols = [f"y_meas_{i + 1}" for i in range(len(y_meas))]
        row: dict = {"time": t, "dE": dE, "runtime": runtime}
        row.update(zip(self._u_cols, (float(v) for v in u_ctrl)))
        row.update(zip(self._y_cols, (float(v) for v in y_meas)))
        self._records.append(row)

    def to_dataframe(self) -> pd.DataFrame:
        """Return all logged records as a DataFrame.

        Columns are in the order: time, u_ctrl_*, y_meas_*, dE, runtime.
        Missing values (e.g. u_ctrl in the IC row) are filled with NaN.
        """
        return pd.DataFrame(self._records)

    def write_metadata(self) -> None:
        """Write a JSON sidecar describing this run's checkpoints (rank-0 only).

        The sidecar is updated after every checkpoint so that a crashed run
        still produces a valid (partial) metadata file. Its presence signals
        that the corresponding XDMF files are safe to use for restart.
        """
        meta = {
            "Tstart": self._Tstart,
            "dt": self._dt,
            "save_every": self._save_every,
            "checkpoints_written": self._checkpoints_written,
            "restart_order": 2,
            "files": {
                "U": self.paths.U_restart.name,
                "Uprev": self.paths.Uprev_restart.name,
                "P": self.paths.P_restart.name,
            },
        }
        if flu.MpiUtils.get_rank() == 0:
            self.paths.metadata.parent.mkdir(parents=True, exist_ok=True)
            self.paths.metadata.write_text(json.dumps(meta, indent=2))

    def write_timeseries(self) -> None:
        """Write the timeseries to CSV (rank-0 only)."""
        if flu.MpiUtils.get_rank() == 0:
            self.to_dataframe().to_csv(self.paths.timeseries, sep=",", index=False)

    def log_progress(
        self,
        iter: int,
        num_steps: int,
        t: float,
        t_end: float,
        runtime: float,
    ) -> None:
        """Log a one-line progress message for the current time step."""
        logger.info(
            "--- iter: %5d/%5d --- time: %3.3f/%3.3f --- elapsed %5.5f ---",
            iter, num_steps, t, t_end, runtime,
        )

    def reset(self) -> None:
        """Clear all accumulated records (e.g. before a restart run)."""
        self._records.clear()
        self._checkpoints_written = 0
