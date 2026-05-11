import pytest

gmsh = pytest.importorskip("gmsh")


def pytest_collection_modifyitems(items):
    """Auto-mark every test in tests/integration/ as slow, except _fast tests."""
    for item in items:
        if "integration" in str(item.fspath) and "_fast" not in item.nodeid:
            item.add_marker(pytest.mark.slow)


# ── Fixtures for fast CI tests with generated small meshes ──────────────────

@pytest.fixture(scope="session")
def coarse_cylinder_mesh(tmp_path_factory):
    """Small cylinder mesh (D=1, medium density) for fast CI tests."""
    from utils.mesh_generation.cylinder import generate_mesh
    path = tmp_path_factory.mktemp("meshes") / "cylinder_coarse"
    # Use default geometry but reduced mesh densities for speed
    # n1=8, n2=4, n3=2 is reliable while still being fast
    generate_mesh(
        str(path),
        formats=("xdmf",),
        D=1.0,                      # Standard diameter
        n1=8, n2=4, n3=2,          # Reduced densities (from 10, 5, 1)
        segments=80,               # Reduced from 360
    )
    return path.with_suffix(".xdmf")


@pytest.fixture(scope="session")
def coarse_cavity_mesh(tmp_path_factory):
    """Small cavity mesh for fast CI tests."""
    from utils.mesh_generation.cavity import generate_mesh
    path = tmp_path_factory.mktemp("meshes") / "cavity_coarse"
    # Cavity uses n, n1+, n2+, n3+, n1-, n2-, n3- parameters
    # Keys with +/- must be passed via dict unpacking
    generate_mesh(
        str(path),
        formats=("xdmf",),
        n=8,
        **{"n1+": 4, "n2+": 2, "n3+": 2, "n1-": 4, "n2-": 2, "n3-": 2},
    )
    return path.with_suffix(".xdmf")


@pytest.fixture(scope="session")
def coarse_pinball_mesh(tmp_path_factory):
    """Small pinball mesh (3 cylinders, D=1) for fast CI tests."""
    from utils.mesh_generation.pinball import generate_mesh
    path = tmp_path_factory.mktemp("meshes") / "pinball_coarse"
    # Use default geometry but reduced mesh densities for speed
    # n1=8, n2=4, n3=2 is reliable while still being fast
    generate_mesh(
        str(path),
        formats=("xdmf",),
        D=1.0,                      # Standard diameter
        n1=8, n2=4, n3=2,          # Reduced densities (from 10, 5, 1)
        segments=60,               # Reduced from 100
    )
    return path.with_suffix(".xdmf")


@pytest.fixture(scope="session")
def coarse_lidcavity_mesh(tmp_path_factory):
    """Small lid-driven cavity mesh for fast CI tests."""
    from utils.mesh_generation.lidcavity import generate_mesh
    path = tmp_path_factory.mktemp("meshes") / "lidcavity_coarse"
    # Lid-driven cavity: fixed unit square [0,1]x[0,1], only density params
    generate_mesh(
        str(path),
        formats=("xdmf",),
        n1=8, n2=4, n3=2,          # Reduced densities (from 20, 10, 5)
    )
    return path.with_suffix(".xdmf")
