"""Tests for flowcontrol.flowfield — requires FEniCS/dolfin."""

import dolfin
import pytest

from flowcontrol.flowfield import BoundaryConditions, FlowField, FlowFieldCollection


# ── FlowField ─────────────────────────────────────────────────────────────────


class TestFlowField:
    def test_u_and_p_are_split_from_up(self, mixed_space):
        up = dolfin.Function(mixed_space)
        ff = FlowField(up=up)
        assert isinstance(ff.u, dolfin.Function)
        assert isinstance(ff.p, dolfin.Function)

    def test_u_lives_in_velocity_subspace(self, mixed_space):
        up = dolfin.Function(mixed_space)
        ff = FlowField(up=up)
        assert ff.u.value_rank() == 1  # vector field

    def test_p_lives_in_pressure_subspace(self, mixed_space):
        up = dolfin.Function(mixed_space)
        ff = FlowField(up=up)
        assert ff.p.value_rank() == 0  # scalar field

    def test_up_field_is_stored(self, mixed_space):
        up = dolfin.Function(mixed_space)
        ff = FlowField(up=up)
        assert ff.up is up


# ── FlowFieldCollection ───────────────────────────────────────────────────────


class TestFlowFieldCollection:
    def test_all_fields_default_to_none(self):
        ffc = FlowFieldCollection()
        for attr in ["U0", "P0", "UP0", "ic", "u_", "p_", "up_", "u_n", "u_nn", "p_n",
                     "Usave", "Psave", "Usave_n"]:
            assert getattr(ffc, attr) is None

    def test_fields_are_assignable(self, mixed_space):
        ffc = FlowFieldCollection()
        u_func = dolfin.Function(mixed_space.sub(0).collapse())
        ffc.U0 = u_func
        assert ffc.U0 is u_func

    def test_construction_with_kwargs(self, mixed_space):
        U0 = dolfin.Function(mixed_space.sub(0).collapse())
        ffc = FlowFieldCollection(U0=U0)
        assert ffc.U0 is U0
        assert ffc.P0 is None


# ── BoundaryConditions ────────────────────────────────────────────────────────


class TestBoundaryConditions:
    def test_construction_and_access(self, mixed_space):
        bc = dolfin.DirichletBC(
            mixed_space.sub(0),
            dolfin.Constant((0.0, 0.0)),
            "on_boundary",
        )
        bcs = BoundaryConditions(bcu=[bc], bcp=[])
        assert len(bcs.bcu) == 1
        assert len(bcs.bcp) == 0
        assert bcs.bcu[0] is bc

    def test_empty_lists(self):
        bcs = BoundaryConditions(bcu=[], bcp=[])
        assert bcs.bcu == []
        assert bcs.bcp == []
