import sys
from pathlib import Path

import dolfin
import pytest

# Add src/ to path so tests can import without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Shared fixtures for all tests ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def unit_square_mesh():
    """4×4 unit square mesh - default for fast unit tests (API, structure)."""
    return dolfin.UnitSquareMesh(4, 4)


@pytest.fixture(scope="module")
def unit_square_mesh_fine():
    """8×8 unit square mesh - for tests checking numerical values."""
    return dolfin.UnitSquareMesh(8, 8)


@pytest.fixture(scope="module")
def mixed_space(unit_square_mesh):
    """Taylor-Hood P2/P1 mixed space on 4×4 unit square."""
    P2 = dolfin.VectorElement("Lagrange", unit_square_mesh.ufl_cell(), 2)
    P1 = dolfin.FiniteElement("Lagrange", unit_square_mesh.ufl_cell(), 1)
    return dolfin.FunctionSpace(unit_square_mesh, P2 * P1)


@pytest.fixture(scope="module")
def mixed_space_fine(unit_square_mesh_fine):
    """Taylor-Hood P2/P1 mixed space on 8×8 unit square."""
    P2 = dolfin.VectorElement("Lagrange", unit_square_mesh_fine.ufl_cell(), 2)
    P1 = dolfin.FiniteElement("Lagrange", unit_square_mesh_fine.ufl_cell(), 1)
    return dolfin.FunctionSpace(unit_square_mesh_fine, P2 * P1)


@pytest.fixture(scope="module")
def mock_flowsolver(mixed_space):
    """Minimal FlowSolver-like object with mesh and spaces for testing."""
    class Mock:
        mesh = mixed_space.mesh()
        W = mixed_space
        V = mixed_space.sub(0).collapse()
        P = mixed_space.sub(1).collapse()
    return Mock()
