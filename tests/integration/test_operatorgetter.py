"""Integration tests for OperatorGetter: get_A, get_mass_matrix, get_B, get_C.

Two fixtures:
  - fs_cylinder: CylinderFlowSolver (BC actuators) — used for the rigorous get_A checks.
  - fs_cavity:   CavityFlowSolver   (FORCE actuator) — used to exercise the FORCE path in B.

get_A checks (cylinder and cavity):
  1. Both paths (autodiff=True/False) produce the same matrix-vector product.
  2. Finite-difference validation: A @ x ≈ -(F(UP0+h·x) - F(UP0)) / h on interior DOFs.
  3. Regression: Frobenius norm locked to a reference value.

get_mass_matrix / get_B / get_C: smoke tests + shape checks on both flows.
"""

import dolfin
import numpy as np
import pytest

from examples.cavity.cavityflowsolver import CavityFlowSolver
from examples.cylinder.cylinderflowsolver import CylinderFlowSolver
from flowcontrol.operatorgetter import OperatorGetter

_A_FROBENIUS_REF = {
    "fs_cylinder": 55.37024024761875,
    "fs_cavity": 47.318499,  # set after first run with -s
}


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def fs_cylinder(tmp_path_factory):
    """CylinderFlowSolver (BC actuators) with a converged steady state."""
    fs = CylinderFlowSolver.make_default(
        Re=100, path_out=tmp_path_factory.mktemp("opget_cylinder")
    )
    fs.compute_steady_state(method="picard", max_iter=3, tol=1e-7, u_ctrl=[0.0, 0.0])
    fs.compute_steady_state(
        method="newton", max_iter=25, u_ctrl=[0.0, 0.0], initial_guess=fs.fields.UP0
    )
    return fs


@pytest.fixture(scope="session")
def fs_cavity(tmp_path_factory):
    """CavityFlowSolver (FORCE actuator) with a converged steady state."""
    fs = CavityFlowSolver.make_default(
        Re=500, path_out=tmp_path_factory.mktemp("opget_cavity")
    )
    fs.compute_steady_state(method="picard", max_iter=10, tol=1e-7, u_ctrl=[0.0])
    fs.compute_steady_state(
        method="newton", max_iter=10, u_ctrl=[0.0], initial_guess=fs.fields.UP0
    )
    return fs


# ── Helpers ───────────────────────────────────────────────────────────────────


def _assemble_residual(fs, UP0: dolfin.Function) -> np.ndarray:
    """Assemble the steady NS residual F(UP0) as a local numpy vector."""
    f = fs._gather_actuators_expressions()
    F_form = fs.forms.steady(UP0, f)
    return dolfin.assemble(F_form).get_local()


def _matvec(A: dolfin.PETScMatrix, x: np.ndarray) -> np.ndarray:
    """Compute A @ x using PETSc multiply (no dense allocation)."""
    x_vec = dolfin.Vector(dolfin.MPI.comm_world, A.size(1))
    x_vec.set_local(x)
    x_vec.apply("insert")
    y_vec = dolfin.Vector(dolfin.MPI.comm_world, A.size(0))
    A.mult(x_vec, y_vec)
    return y_vec.get_local()


def _interior_dofs(fs) -> np.ndarray:
    """Return indices of DOFs that are NOT subject to a Dirichlet BC."""
    W = fs.W
    all_dofs = np.arange(len(W.dofmap().dofs()), dtype=np.int32)
    bc_dofs = set()
    for bc in fs.bc.bcu:
        bc_dofs.update(bc.get_boundary_values().keys())
    ownership = W.dofmap().ownership_range()
    bc_dofs_local = {
        d - ownership[0] for d in bc_dofs if ownership[0] <= d < ownership[1]
    }
    mask = np.ones(len(all_dofs), dtype=bool)
    mask[list(bc_dofs_local)] = False
    return all_dofs[mask]


# ── get_A tests ───────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.parametrize("fs_fixture", ["fs_cylinder", "fs_cavity"])
def test_get_A_paths_agree(fs_fixture, request):
    """autodiff and manual linearization must produce the same matrix-vector product."""
    fs = request.getfixturevalue(fs_fixture)
    opget = OperatorGetter(fs)
    A_auto = opget.get_A(autodiff=True)
    A_manual = opget.get_A(autodiff=False)

    rng = np.random.default_rng(0)
    x = rng.standard_normal(A_auto.size(1))

    y_auto = _matvec(A_auto, x)
    y_manual = _matvec(A_manual, x)

    rel_err = np.linalg.norm(y_auto - y_manual) / (np.linalg.norm(y_auto) + 1e-14)
    assert rel_err < 1e-10, f"autodiff vs manual relative error: {rel_err:.2e}"


@pytest.mark.slow
@pytest.mark.parametrize("fs_fixture", ["fs_cylinder", "fs_cavity"])
def test_get_A_finite_difference(fs_fixture, request):
    """A @ x must agree with the FD approximation of -dF/dUP0 @ x on interior DOFs."""
    fs = request.getfixturevalue(fs_fixture)
    opget = OperatorGetter(fs)
    A = opget.get_A(autodiff=True)

    UP0 = fs.fields.UP0
    interior = _interior_dofs(fs)

    rng = np.random.default_rng(1)
    x_full = np.zeros(A.size(1))
    x_full[interior] = rng.standard_normal(len(interior))

    h = 1e-6
    UP0_pert = UP0.copy(deepcopy=True)
    UP0_pert.vector().set_local(UP0.vector().get_local() + h * x_full)
    UP0_pert.vector().apply("insert")

    fd_approx = -(_assemble_residual(fs, UP0_pert) - _assemble_residual(fs, UP0)) / h
    Ax = _matvec(A, x_full)

    rel_err = np.linalg.norm(Ax[interior] - fd_approx[interior]) / (
        np.linalg.norm(Ax[interior]) + 1e-14
    )
    assert rel_err < 1e-4, f"FD validation relative error: {rel_err:.2e}"


@pytest.mark.slow
@pytest.mark.parametrize("fs_fixture", ["fs_cylinder", "fs_cavity"])
def test_get_A_regression(fs_fixture, request):
    """Frobenius norm of A must match the stored reference value."""
    fs = request.getfixturevalue(fs_fixture)
    opget = OperatorGetter(fs)
    A = opget.get_A(autodiff=True)
    frob = A.norm("frobenius")

    ref = _A_FROBENIUS_REF[fs_fixture]
    if ref is None:
        pytest.skip(
            f"Reference not yet captured — set _A_FROBENIUS_REF['{fs_fixture}'] = {frob!r}"
        )

    assert np.isclose(frob, ref, rtol=1e-6), f"||A||_F = {frob} != reference {ref}"


# ── get_mass_matrix tests ─────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.parametrize("fs_fixture", ["fs_cylinder", "fs_cavity"])
def test_get_mass_matrix_shape(fs_fixture, request):
    """E must be square with size W.dim() × W.dim() and have finite entries."""
    fs = request.getfixturevalue(fs_fixture)
    opget = OperatorGetter(fs)
    E = opget.get_mass_matrix()

    n = fs.W.dim()
    assert E.size(0) == n
    assert E.size(1) == n
    assert np.isfinite(E.norm("frobenius"))


# ── get_B tests ───────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_get_B_shape_bc_actuators(fs_cylinder):
    """B must have shape (local_dofs × n_actuators) for BC actuators."""
    opget = OperatorGetter(fs_cylinder)
    B = opget.get_B()

    n_local = len(fs_cylinder.W.dofmap().dofs())
    n_act = fs_cylinder.params_control.actuator_number
    assert B.shape == (n_local, n_act), f"B.shape={B.shape}, expected ({n_local}, {n_act})"
    assert np.all(np.isfinite(B))


@pytest.mark.slow
def test_get_B_shape_force_actuator(fs_cavity):
    """B must have shape (local_dofs × n_actuators) for a FORCE actuator."""
    opget = OperatorGetter(fs_cavity)
    B = opget.get_B()

    n_local = len(fs_cavity.W.dofmap().dofs())
    n_act = fs_cavity.params_control.actuator_number
    assert B.shape == (n_local, n_act), f"B.shape={B.shape}, expected ({n_local}, {n_act})"
    assert np.all(np.isfinite(B))


# ── get_C tests ───────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.parametrize("fs_fixture", ["fs_cylinder", "fs_cavity"])
def test_get_C_shape(fs_fixture, request):
    """C must have shape (n_sensors × local_dofs) and finite entries."""
    fs = request.getfixturevalue(fs_fixture)
    opget = OperatorGetter(fs)
    C = opget.get_C()

    n_local = len(fs.W.dofmap().dofs())
    n_sens = fs.params_control.sensor_number
    assert C.shape == (n_sens, n_local), f"C.shape={C.shape}, expected ({n_sens}, {n_local})"
    assert np.all(np.isfinite(C))
