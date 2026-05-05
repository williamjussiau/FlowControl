"""Tests for flowcontrol.flowsolverparameters.

These tests exercise only the dataclass construction and __post_init__ logic
(no mesh, no FEM assembly).  They do require the FEniCS conda env because
flowsolverparameters.py imports from actuator.py and sensor.py, which in
turn import dolfin.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from flowcontrol.flowsolverparameters import (
    ParamControl,
    ParamFlow,
    ParamIC,
    ParamMesh,
    ParamRestart,
    ParamSave,
    ParamSolver,
    ParamTime,
)

# ── ParamTime ─────────────────────────────────────────────────────────────────


class TestParamTime:
    def test_tfinal_auto_computed(self):
        p = ParamTime(num_steps=100, dt=0.01, Tstart=0.0)
        assert p.Tfinal == pytest.approx(1.0)

    def test_tfinal_non_zero_tstart_not_included(self):
        """Tstart does not affect Tfinal — it's purely num_steps * dt."""
        p = ParamTime(num_steps=50, dt=0.02, Tstart=5.0)
        assert p.Tfinal == pytest.approx(1.0)

    def test_tfinal_fractional_dt(self):
        p = ParamTime(num_steps=200, dt=0.005, Tstart=0.0)
        assert p.Tfinal == pytest.approx(1.0)

    def test_tfinal_not_user_settable(self):
        with pytest.raises(TypeError):
            ParamTime(num_steps=10, dt=0.1, Tstart=0.0, Tfinal=99.0)


# ── ParamControl ──────────────────────────────────────────────────────────────


class TestParamControl:
    def _make_sensors(self, n):
        return [MagicMock() for _ in range(n)]

    def _make_actuators(self, n):
        return [MagicMock() for _ in range(n)]

    def test_sensor_number_auto_computed(self):
        p = ParamControl(
            sensor_list=self._make_sensors(3),
            actuator_list=self._make_actuators(2),
        )
        assert p.sensor_number == 3

    def test_actuator_number_auto_computed(self):
        p = ParamControl(
            sensor_list=self._make_sensors(1),
            actuator_list=self._make_actuators(4),
        )
        assert p.actuator_number == 4

    def test_empty_lists_give_zero_counts(self):
        p = ParamControl(sensor_list=[], actuator_list=[])
        assert p.sensor_number == 0
        assert p.actuator_number == 0

    def test_sensor_and_actuator_number_not_user_settable(self):
        with pytest.raises(TypeError):
            ParamControl(
                sensor_list=[],
                actuator_list=[],
                sensor_number=99,
            )


# ── ParamFlow ─────────────────────────────────────────────────────────────────


class TestParamFlow:
    def test_construction(self):
        p = ParamFlow(Re=100.0)
        assert p.Re == pytest.approx(100.0)

    def test_uinf_default(self):
        p = ParamFlow(Re=200.0)
        assert p.uinf == pytest.approx(1.0)

    def test_uinf_custom(self):
        p = ParamFlow(Re=50.0, uinf=2.5)
        assert p.uinf == pytest.approx(2.5)


# ── ParamMesh ─────────────────────────────────────────────────────────────────


class TestParamMesh:
    def test_construction(self):
        p = ParamMesh(meshpath=Path("/tmp/mesh.xdmf"))
        assert p.meshpath == Path("/tmp/mesh.xdmf")


# ── ParamIC ───────────────────────────────────────────────────────────────────


class TestParamIC:
    def test_defaults(self):
        p = ParamIC()
        assert p.xloc == pytest.approx(0.0)
        assert p.yloc == pytest.approx(0.0)
        assert p.radius == pytest.approx(1.0)
        assert p.amplitude == pytest.approx(1.0)

    def test_custom_values(self):
        p = ParamIC(xloc=1.0, yloc=-0.5, radius=0.2, amplitude=3.0)
        assert p.xloc == pytest.approx(1.0)
        assert p.yloc == pytest.approx(-0.5)
        assert p.radius == pytest.approx(0.2)
        assert p.amplitude == pytest.approx(3.0)


# ── ParamSave ─────────────────────────────────────────────────────────────────


class TestParamSave:
    def test_construction(self):
        p = ParamSave(path_out=Path("/tmp/out"), save_every=10)
        assert p.path_out == Path("/tmp/out")
        assert p.save_every == 10

    def test_energy_every_default(self):
        p = ParamSave(path_out=Path("/tmp"), save_every=5)
        assert p.energy_every == 1


# ── ParamSolver ───────────────────────────────────────────────────────────────


class TestParamSolver:
    def test_defaults(self):
        p = ParamSolver()
        assert p.throw_error is True
        assert p.shift == pytest.approx(0.0)
        assert p.is_eq_nonlinear is True
        assert p.time_scheme == "bdf"

    def test_custom_values(self):
        p = ParamSolver(throw_error=False, shift=0.5, is_eq_nonlinear=False, time_scheme="cn")
        assert p.throw_error is False
        assert p.shift == pytest.approx(0.5)
        assert p.is_eq_nonlinear is False
        assert p.time_scheme == "cn"


# ── ParamRestart ──────────────────────────────────────────────────────────────


class TestParamRestart:
    def test_defaults(self):
        p = ParamRestart()
        assert p.save_every_old == 0
        assert p.restart_order == 2
        assert p.dt_old == pytest.approx(0.0)
        assert p.Trestartfrom == pytest.approx(0.0)

    def test_custom_values(self):
        p = ParamRestart(save_every_old=5, restart_order=1, dt_old=0.01, Trestartfrom=2.5)
        assert p.save_every_old == 5
        assert p.restart_order == 1
        assert p.dt_old == pytest.approx(0.01)
        assert p.Trestartfrom == pytest.approx(2.5)


# ── user_data (inherited from ParamFlowSolver) ────────────────────────────────


class TestUserData:
    def test_user_data_default_empty_dict(self):
        p = ParamFlow(Re=100.0)
        assert p.user_data == {}

    def test_user_data_can_hold_arbitrary_values(self):
        p = ParamFlow(Re=100.0, user_data={"D": 0.1, "L": 2.2})
        assert p.user_data["D"] == pytest.approx(0.1)
