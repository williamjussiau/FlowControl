"""Mesh generation for the lid-driven cavity flow domain."""

import gmsh

from ._common import _bbox, _write_mesh

_DEFAULT_MESH_PARAM = {
    "n1": 20.0,
    "n2": 10.0,
    "n3": 5.0,
}


def generate_mesh(filename, formats=("xml", "xdmf"), verbose=False, **mesh_param):
    """Generate and write a 2D lid-driven cavity mesh.

    Parameters
    ----------
    filename : str
        Output file path without extension.
    formats : sequence of str
        Output formats; any of ``"xml"``, ``"xdmf"``.
    verbose : bool
        Print mesh parameters before building.
    **mesh_param :
        Cell density overrides.

        - ``n1``: density in the top zone, near the moving lid (default 20.0)
        - ``n2``: density in the middle zone (default 10.0)
        - ``n3``: density in the bottom zone (default 5.0)

    Notes
    -----
    Domain geometry is a unit square [0, 1] x [0, 1] (fixed).

    Mesh topology (y-coordinates shown on the right)::

        0                   1
        |-------------------|  1  ← moving lid
        |        n1         |
        |-------------------|  0.7
        |                   |
        |        n2         |
        |                   |
        |-------------------|  0.3
        |        n3         |
        |-------------------|  0
    """
    prm = {**_DEFAULT_MESH_PARAM, **mesh_param}
    if verbose:
        print("Making mesh with parameters:", prm)
    gmsh.initialize()
    try:
        gmsh.model.add("lidcavity")
        _build_mesh(prm)
        gmsh.model.mesh.generate(2)
        _write_mesh(filename, formats)
    finally:
        gmsh.finalize()


def _build_mesh(prm):
    factory = gmsh.model.occ

    # Fixed domain geometry
    L, H = 1.0, 1.0

    # Density zone boundaries (y-coordinates)
    y_n1_n2 = 0.7  # n1/n2 boundary (near lid)
    y_n2_n3 = 0.3  # n2/n3 boundary (near bottom)

    # Points: bottom row, mid-low row, mid-high row, top row
    p_bl = factory.addPoint(0.0, 0.0, 0)
    p_br = factory.addPoint(L, 0.0, 0)
    p_mll = factory.addPoint(0.0, y_n2_n3, 0)
    p_mlr = factory.addPoint(L, y_n2_n3, 0)
    p_mhl = factory.addPoint(0.0, y_n1_n2, 0)
    p_mhr = factory.addPoint(L, y_n1_n2, 0)
    p_tl = factory.addPoint(0.0, H, 0)
    p_tr = factory.addPoint(L, H, 0)

    # Lines
    l_bot = factory.addLine(p_bl, p_br)
    l_rbot = factory.addLine(p_br, p_mlr)
    l_z1 = factory.addLine(p_mlr, p_mll)  # first zone boundary, right → left
    l_lbot = factory.addLine(p_mll, p_bl)
    l_rmid = factory.addLine(p_mlr, p_mhr)
    l_z2 = factory.addLine(p_mhr, p_mhl)  # second zone boundary, right → left
    l_lmid = factory.addLine(p_mhl, p_mll)
    l_rtop = factory.addLine(p_mhr, p_tr)
    l_top = factory.addLine(p_tr, p_tl)
    l_ltop = factory.addLine(p_tl, p_mhl)

    # Three stacked rectangular surfaces (bottom → top)
    loop_bot = factory.addCurveLoop([l_bot, l_rbot, l_z1, l_lbot])
    loop_mid = factory.addCurveLoop([-l_z1, l_rmid, l_z2, l_lmid])
    loop_top = factory.addCurveLoop([-l_z2, l_rtop, l_top, l_ltop])

    factory.addPlaneSurface([loop_bot])
    factory.addPlaneSurface([loop_mid])
    factory.addPlaneSurface([loop_top])

    factory.synchronize()

    eps = 0.01

    # Bottom → top, each call overrides the previous.
    gmsh.model.mesh.setSize(_bbox(-eps, -eps, L + eps, H + eps), 1 / prm["n3"])
    gmsh.model.mesh.setSize(_bbox(-eps, y_n2_n3 - eps, L + eps, H + eps), 1 / prm["n2"])
    gmsh.model.mesh.setSize(_bbox(-eps, y_n1_n2 - eps, L + eps, H + eps), 1 / prm["n1"])
