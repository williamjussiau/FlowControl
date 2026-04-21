"""Tests for flowcontrol.sensor.

No-mesh section: SENSOR_TYPE enum, abstract enforcement, instantiation,
    require_loading defaults, pre-load state.
With-mesh section: SensorPoint.eval on a known field, SensorHorizontalWallShear.load,
    and the key consistency check eval(up) == assemble(linear_form(up)).
"""

import dolfin
import numpy as np
import pytest
from numpy.testing import assert_allclose

from flowcontrol.sensor import (
    SENSOR_INDEX_DEFAULT,
    SENSOR_TYPE,
    Sensor,
    SensorHorizontalWallShear,
    SensorIntegral,
    SensorPoint,
)

# ── Shared mock ───────────────────────────────────────────────────────────────


class MockFlowSolver:
    """Minimal stand-in for FlowSolver with mesh, W and V."""

    def __init__(self):
        mesh = dolfin.UnitSquareMesh(8, 8)
        P2 = dolfin.VectorElement("Lagrange", mesh.ufl_cell(), 2)
        P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        W = dolfin.FunctionSpace(mesh, P2 * P1)
        self.mesh = mesh
        self.W = W
        self.V = W.sub(0).collapse()


@pytest.fixture(scope="module")
def mock_fs():
    return MockFlowSolver()


# ── No-mesh: SENSOR_TYPE enum ─────────────────────────────────────────────────


class TestSensorTypeEnum:
    def test_u_is_zero(self):
        assert SENSOR_TYPE.U == 0

    def test_v_is_one(self):
        assert SENSOR_TYPE.V == 1

    def test_p_is_two(self):
        assert SENSOR_TYPE.P == 2

    def test_other_out_of_bounds(self):
        """OTHER must be >= 3 so it cannot be used as an index into (u, v, p)."""
        assert SENSOR_TYPE.OTHER >= 3


# ── No-mesh: abstract enforcement ─────────────────────────────────────────────


class TestAbstract:
    def test_sensor_is_abstract(self):
        with pytest.raises(TypeError):
            Sensor(sensor_type=SENSOR_TYPE.U, require_loading=False)  # type: ignore

    def test_sensor_integral_is_abstract(self):
        with pytest.raises(TypeError):
            SensorIntegral(sensor_type=SENSOR_TYPE.OTHER, sensor_index=1)  # type: ignore


# ── No-mesh: instantiation and defaults ───────────────────────────────────────


class TestSensorPointInit:
    def test_instantiation(self):
        s = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([0.5, 0.5]))
        assert s.sensor_type is SENSOR_TYPE.V
        assert_allclose(s.position, [0.5, 0.5])

    def test_require_loading_is_false(self):
        s = SensorPoint(sensor_type=SENSOR_TYPE.U, position=np.array([0.0, 0.0]))
        assert s.require_loading is False


class TestSensorHorizontalWallShearInit:
    def test_instantiation_defaults(self):
        s = SensorHorizontalWallShear(sensor_type=SENSOR_TYPE.OTHER, sensor_index=42)
        assert s.x_sensor_left == 1.0
        assert s.x_sensor_right == 1.1
        assert s.y_sensor == 0.0

    def test_require_loading_is_true(self):
        s = SensorHorizontalWallShear(sensor_type=SENSOR_TYPE.OTHER, sensor_index=42)
        assert s.require_loading is True

    def test_ds_none_before_load(self):
        s = SensorHorizontalWallShear(sensor_type=SENSOR_TYPE.OTHER, sensor_index=42)
        assert s.ds is None

    def test_subdomain_none_before_load(self):
        s = SensorHorizontalWallShear(sensor_type=SENSOR_TYPE.OTHER, sensor_index=42)
        assert s.subdomain is None

    def test_sensor_index_default(self):
        s = SensorHorizontalWallShear(
            sensor_type=SENSOR_TYPE.OTHER, sensor_index=SENSOR_INDEX_DEFAULT
        )
        assert s.sensor_index == SENSOR_INDEX_DEFAULT


# ── With mesh: SensorPoint.eval ───────────────────────────────────────────────


class TestSensorPointEval:
    """Evaluate a constant mixed field and check each component is returned correctly."""

    @pytest.fixture(scope="class")
    def constant_field(self, mock_fs):
        """Mixed field with u=(1, 2), p=3 everywhere."""
        return dolfin.interpolate(dolfin.Constant((1.0, 2.0, 3.0)), mock_fs.W)

    def test_u_component(self, constant_field):
        s = SensorPoint(sensor_type=SENSOR_TYPE.U, position=np.array([0.5, 0.5]))
        assert_allclose(s.eval(constant_field), 1.0, atol=1e-12)

    def test_v_component(self, constant_field):
        s = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([0.5, 0.5]))
        assert_allclose(s.eval(constant_field), 2.0, atol=1e-12)


# ── With mesh: SensorHorizontalWallShear ──────────────────────────────────────


class TestSensorHorizontalWallShear:
    """Bottom wall sensor: y=0, x in [0.25, 0.75]."""

    @pytest.fixture(scope="class")
    def loaded_sensor(self, mock_fs):
        s = SensorHorizontalWallShear(
            sensor_type=SENSOR_TYPE.OTHER,
            sensor_index=50,
            x_sensor_left=0.25,
            x_sensor_right=0.75,
            y_sensor=0.0,
        )
        s.load(mock_fs)
        return s

    def test_load_sets_ds(self, loaded_sensor):
        assert loaded_sensor.ds is not None

    def test_load_sets_subdomain(self, loaded_sensor):
        assert loaded_sensor.subdomain is not None

    def test_eval_matches_linear_form(self, loaded_sensor, mock_fs):
        """Core refactoring check: eval(up) must equal assemble(linear_form(up)).

        Uses a random field so equality is non-trivial.
        """
        up = dolfin.Function(mock_fs.W)
        up.vector()[:] = np.random.default_rng(0).standard_normal(mock_fs.W.dim())

        val_eval = loaded_sensor.eval(up)
        val_form = dolfin.assemble(loaded_sensor.linear_form(up))

        assert_allclose(val_eval, val_form, rtol=1e-12)

    def test_linear_form_with_testfunction_gives_vector(self, loaded_sensor, mock_fs):
        """linear_form(TestFunction) must assemble to a vector (the C matrix row)."""
        v = dolfin.TestFunction(mock_fs.W)
        c_row = dolfin.assemble(loaded_sensor.linear_form(v))
        assert c_row.size() == mock_fs.W.dim()

    def test_c_row_consistent_with_eval(self, loaded_sensor, mock_fs):
        """C_row · up_vec must equal eval(up) (linearity check)."""
        up = dolfin.Function(mock_fs.W)
        up.vector()[:] = np.random.default_rng(1).standard_normal(mock_fs.W.dim())

        v = dolfin.TestFunction(mock_fs.W)
        c_row = dolfin.assemble(loaded_sensor.linear_form(v)).get_local()
        y_from_c = c_row @ up.vector().get_local()

        y_eval = loaded_sensor.eval(up)

        assert_allclose(y_from_c, y_eval, rtol=1e-10)
