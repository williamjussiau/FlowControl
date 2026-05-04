"""Mesh generation for the fluidic pinball (3-cylinder) flow domain."""

import math

import gmsh

from ._common import _bbox, _write_mesh

_CYLINDER_MARGIN = 0.1

_DEFAULT_MESH_PARAM = {
    "xinfa": -6.0,
    "xinf": 20.0,
    "yinf": 6.0,
    "D": 1.0,
    "inftola": 2.0,
    "inftol": 5.0,
    "yint": 3.0,
    "n1": 10.0,
    "n2": 5.0,
    "n3": 1.0,
    "segments": 100,
}


def generate_mesh(filename, formats=("xml", "xdmf"), verbose=False, **mesh_param):
    """Generate and write a 2D fluidic pinball mesh (3 cylinders).

    Parameters
    ----------
    filename : str
        Output file path without extension.
    formats : sequence of str
        Output formats; any of ``"xml"``, ``"xdmf"``.
    verbose : bool
        Print mesh parameters before building.
    **mesh_param :
        Geometry and density overrides.

        - ``xinfa``: upstream domain extent (default -6.0)
        - ``xinf``: downstream domain extent (default 20.0)
        - ``yinf``: lateral domain half-width (default 6.0)
        - ``D``: cylinder diameter (default 1.0)
        - ``inftola``: left margin from domain to middle zone (default 2.0)
        - ``inftol``: right margin from domain to middle zone (default 5.0)
        - ``yint``: half-height of middle zone (default 3.0)
        - ``n1``: cell density in inner zone, around cylinders (default 10.0)
        - ``n2``: cell density in middle zone (default 5.0)
        - ``n3``: cell density in outer zone (default 1.0)
        - ``segments``: elements per cylinder boundary (default 100)

    Notes
    -----
    Cylinder centers follow the equilateral triangle layout (Delphin-Ruche 2019),
    with R = D/2::

        mid: (-3R·cos(π/6),  0)    upstream cylinder
        top: (0,            +3R/2)  top downstream cylinder
        bot: (0,            -3R/2)  bottom downstream cylinder

    For D=1: mid ≈ (-1.30, 0), top = (0, +0.75), bot = (0, -0.75).

    The inner zone encloses all three cylinders with a margin of 2R on each side.
    For D=1: x ∈ [-2.30, 1.00], y ∈ [-1.75, +1.75].

    Mesh topology (not to scale)::

             xinfa                                    xinf
             ------------------------------------------------  <<<yinf
             |  yint>>>  ---------------------------------  |
             |           |        ------------           |  |
             |           |   n2   | n1    O  |    n3     |  |
             |           |        |    O     |           |  |
             |<-inftola->|        |       O  |  <inftol> |  |
             |           |        ------------           |  |
             |           ---------------------------------  |
             ------------------------------------------------
    """
    prm = {**_DEFAULT_MESH_PARAM, **mesh_param}
    if verbose:
        print("Making mesh with parameters:", prm)
    gmsh.initialize()
    try:
        gmsh.model.add("pinball")
        _build_mesh(prm)
        gmsh.model.mesh.generate(2)
        _write_mesh(filename, formats)
    finally:
        gmsh.finalize()


def _build_mesh(prm):
    factory = gmsh.model.occ

    R = prm["D"] / 2

    # Cylinder centers: equilateral triangle layout (Delphin-Ruche 2019)
    mid_x, mid_y = -3 * R * math.cos(math.pi / 6), 0.0
    top_x, top_y = 0.0, 3 * R / 2
    bot_x, bot_y = 0.0, -3 * R / 2

    # Inner zone: single rectangle enclosing all three cylinders with 2R margin
    x_in_l = mid_x - 2 * R
    x_in_r = 2 * R
    y_in = top_y + 2 * R

    # Outer rectangle
    pul = factory.addPoint(prm["xinfa"], prm["yinf"], 0)
    pll = factory.addPoint(prm["xinfa"], -prm["yinf"], 0)
    plr = factory.addPoint(prm["xinf"], -prm["yinf"], 0)
    pur = factory.addPoint(prm["xinf"], prm["yinf"], 0)

    # Middle rectangle
    inftol, inftola = prm["inftol"], prm["inftola"]
    pmul = factory.addPoint(prm["xinfa"] + inftola, prm["yint"], 0)
    pmll = factory.addPoint(prm["xinfa"] + inftola, -prm["yint"], 0)
    pmlr = factory.addPoint(prm["xinf"] - inftol, -prm["yint"], 0)
    pmur = factory.addPoint(prm["xinf"] - inftol, prm["yint"], 0)

    # Inner rectangle (one big zone around all 3 cylinders)
    piul = factory.addPoint(x_in_l, y_in, 0)
    pill = factory.addPoint(x_in_l, -y_in, 0)
    pilr = factory.addPoint(x_in_r, -y_in, 0)
    piur = factory.addPoint(x_in_r, y_in, 0)

    # Three cylinder boundaries
    circ_mid = factory.addCircle(mid_x, mid_y, 0, R)
    circ_top = factory.addCircle(top_x, top_y, 0, R)
    circ_bot = factory.addCircle(bot_x, bot_y, 0, R)

    # Outer rectangle lines
    line_ele = factory.addLine(pul, pll)
    line_elo = factory.addLine(pll, plr)
    line_eri = factory.addLine(plr, pur)
    line_eup = factory.addLine(pur, pul)

    # Middle rectangle lines
    line_mle = factory.addLine(pmul, pmll)
    line_mlo = factory.addLine(pmll, pmlr)
    line_mri = factory.addLine(pmlr, pmur)
    line_mup = factory.addLine(pmur, pmul)

    # Inner rectangle lines
    line_ile = factory.addLine(piul, pill)
    line_ilo = factory.addLine(pill, pilr)
    line_iri = factory.addLine(pilr, piur)
    line_iup = factory.addLine(piur, piul)

    rect_ext = factory.addCurveLoop([line_ele, line_elo, line_eri, line_eup])
    rect_mid = factory.addCurveLoop([line_mle, line_mlo, line_mri, line_mup])
    rect_in = factory.addCurveLoop([line_ile, line_ilo, line_iri, line_iup])
    loop_mid = factory.addCurveLoop([circ_mid])
    loop_top = factory.addCurveLoop([circ_top])
    loop_bot = factory.addCurveLoop([circ_bot])

    factory.addPlaneSurface([rect_ext, rect_mid])
    factory.addPlaneSurface([rect_mid, rect_in])
    factory.addPlaneSurface([rect_in, loop_mid, loop_top, loop_bot])

    factory.synchronize()

    # Coarse → fine, each call overrides the previous.
    ext_ = _bbox(prm["xinfa"], -prm["yinf"], prm["xinf"], prm["yinf"])
    mid_ = _bbox(prm["xinfa"], -prm["yint"], prm["xinf"], prm["yint"])
    in_ = _bbox(prm["xinfa"], -y_in, prm["xinf"], y_in)
    gmsh.model.mesh.setSize(ext_, 1 / prm["n3"])
    gmsh.model.mesh.setSize(mid_, 1 / prm["n2"])
    gmsh.model.mesh.setSize(in_, 1 / prm["n1"])

    # Per-cylinder boundary refinement
    cyl_size = min(1 / prm["n1"], 2 * math.pi / prm["segments"])
    r_bbox = R + _CYLINDER_MARGIN
    for cx, cy in [(mid_x, mid_y), (top_x, top_y), (bot_x, bot_y)]:
        pts = _bbox(cx - r_bbox, cy - r_bbox, cx + r_bbox, cy + r_bbox)
        gmsh.model.mesh.setSize(pts, cyl_size)
