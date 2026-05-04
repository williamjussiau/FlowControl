"""Mesh generation for the open cavity flow domain."""

import gmsh

from ._common import _bbox, _write_mesh

_DEFAULT_MESH_PARAM = {
    "n": 50.0,
    "n1+": 25.0,
    "n2+": 10.0,
    "n3+": 5.0,
    "n1-": 25.0,
    "n2-": 10.0,
    "n3-": 5.0,
}


def generate_mesh(filename, formats=("xml", "xdmf"), verbose=False, **mesh_param):
    """Generate and write a 2D open cavity flow mesh.

    Parameters
    ----------
    filename : str
        Output file path without extension.
    formats : sequence of str
        Output formats; any of ``"xml"``, ``"xdmf"``.
    verbose : bool
        Print mesh parameters before building.
    **mesh_param :
        Cell density overrides. Parameter names follow Sipp & Lebedev (2007).
        Keys containing ``+``/``-`` must be passed via dict unpacking:
        ``generate_mesh(f, **{"n1+": 30})``.

        - ``n``: density near the cavity opening (default 50.0 / SL2007: 350)
        - ``n1+``: density in first upper layer (default 25.0 / SL2007: 200)
        - ``n2+``: density in second upper layer (default 10.0 / SL2007: 100)
        - ``n3+``: density in outer channel (default 5.0 / SL2007: 100)
        - ``n1-``: density in first cavity layer, near mouth (default 25.0 / SL2007: 150)
        - ``n2-``: density in second cavity layer (default 10.0 / SL2007: 100)
        - ``n3-``: density in third cavity layer, near bottom (default 5.0 / SL2007: 50)

    Notes
    -----
    Domain geometry is fixed (Sipp & Lebedev 2007). Only cell densities are configurable.

    Mesh topology (y-coordinates shown on the right, x-coordinates below)::

      -----------------------------------------------------------------  0.5
      |                                                               |
      |                        n3+                                    |
      |---------------------------------------------------------------|  0.3
      |                        n2+                                    |
      |---------------------------------------------------------------|  0.15
      |                        n1+                                    |
      |        -------------------------------------------------------|  0.1
      |        |               n                                      |
      ---------------------              ------------------------------  0
    -1.2     -0.6         0              1                           2.5
                          |--------------| -0.1
                          |    n1-       |
                          |--------------| -0.2
                          |    n2-       |
                          |--------------| -0.35
                          |    n3-       |
                          |              |
                          ---------------- -1
    """
    prm = {**_DEFAULT_MESH_PARAM, **mesh_param}
    if verbose:
        print("Making mesh with parameters:", prm)
    gmsh.initialize()
    try:
        gmsh.model.add("cavity")
        _build_mesh(prm)
        gmsh.model.mesh.generate(2)
        _write_mesh(filename, formats)
    finally:
        gmsh.finalize()


def _build_mesh(prm):
    factory = gmsh.model.occ

    # Fixed domain geometry (Sipp & Lebedev 2007)
    x_left, x_right = -1.2, 2.5
    x_cav_l, x_cav_r = 0.0, 1.0
    y_top, y_cav_bot = 0.5, -1.0

    # Density zone boundaries
    x_refine_start = -0.6  # x start of n1+/n channel zones
    y_n3_n2 = 0.3  # n3+/n2+ boundary
    y_n2_n1 = 0.15  # n2+/n1+ boundary
    y_n1_n = 0.1  # n1+/n boundary (also n zone upper limit in cavity)
    y_n_cav = -0.1  # n zone lower limit in cavity
    y_n1m_n2m = -0.2  # n1-/n2- boundary
    y_n2m_n3m = -0.35  # n2-/n3- boundary

    # Outer boundary: L-shaped domain (channel + open cavity)
    p_cl = factory.addPoint(x_left, 0.0, 0)
    p_tl = factory.addPoint(x_left, y_top, 0)
    p_tr = factory.addPoint(x_right, y_top, 0)
    p_cr = factory.addPoint(x_right, 0.0, 0)
    p_cvr = factory.addPoint(x_cav_r, 0.0, 0)
    p_cbr = factory.addPoint(x_cav_r, y_cav_bot, 0)
    p_cbl = factory.addPoint(x_cav_l, y_cav_bot, 0)
    p_cvl = factory.addPoint(x_cav_l, 0.0, 0)

    l_left = factory.addLine(p_cl, p_tl)
    l_top = factory.addLine(p_tl, p_tr)
    l_right = factory.addLine(p_tr, p_cr)
    l_floor_r = factory.addLine(p_cr, p_cvr)
    l_cav_r = factory.addLine(p_cvr, p_cbr)
    l_cav_bot = factory.addLine(p_cbr, p_cbl)
    l_cav_l = factory.addLine(p_cbl, p_cvl)
    l_floor_l = factory.addLine(p_cvl, p_cl)

    loop = factory.addCurveLoop([l_left, l_top, l_right, l_floor_r, l_cav_r, l_cav_bot, l_cav_l, l_floor_l])
    factory.addPlaneSurface([loop])

    factory.synchronize()

    eps = 0.01

    # Channel: coarsest to finest, moving toward the cavity opening.
    # Each call overrides the previous for all points inside the bounding box.
    gmsh.model.mesh.setSize(_bbox(x_left - eps, -eps, x_right + eps, y_top + eps), 1 / prm["n3+"])
    gmsh.model.mesh.setSize(_bbox(x_left - eps, -eps, x_right + eps, y_n3_n2 + eps), 1 / prm["n2+"])
    gmsh.model.mesh.setSize(_bbox(x_refine_start - eps, -eps, x_right + eps, y_n2_n1 + eps), 1 / prm["n1+"])
    gmsh.model.mesh.setSize(_bbox(x_refine_start - eps, -eps, x_right + eps, y_n1_n + eps), 1 / prm["n"])

    # Cavity: coarsest (bottom) to finest (mouth).
    gmsh.model.mesh.setSize(_bbox(x_cav_l - eps, y_cav_bot - eps, x_cav_r + eps, eps), 1 / prm["n3-"])
    gmsh.model.mesh.setSize(_bbox(x_cav_l - eps, y_n2m_n3m - eps, x_cav_r + eps, eps), 1 / prm["n2-"])
    gmsh.model.mesh.setSize(_bbox(x_cav_l - eps, y_n1m_n2m - eps, x_cav_r + eps, eps), 1 / prm["n1-"])

    # Re-apply finest channel density to the shared y=0 boundary (cavity mouth),
    # overriding the cavity sizing that was just applied there.
    gmsh.model.mesh.setSize(_bbox(x_cav_l - eps, y_n_cav - eps, x_cav_r + eps, y_n1_n + eps), 1 / prm["n"])
