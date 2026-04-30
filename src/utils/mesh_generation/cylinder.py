"""Mesh generation for the cylinder flow domain."""

import math

import gmsh

from ._common import _write_mesh

_DEFAULT_MESH_PARAM = {
    "xinfa": -10.0,
    "xinf": 20.0,
    "yinf": 8.0,
    "xplus": 1.5,
    "yint": 3.0,
    "lint": 1.5,
    "inftol": 5.0,
    "inftola": 5.0,
    "n1": 10.0,
    "n2": 5.0,
    "n3": 1.0,
    "segments": 360,
    "D": 1,
}


def generate_mesh(filename, formats=("xml", "xdmf"), verbose=False, **mesh_param):
    """Generate and write a 2D cylinder flow mesh.

    Parameters
    ----------
    filename : str
        Output file path without extension.
    formats : sequence of str
        Output formats; any of ``"xml"``, ``"xdmf"``.
    verbose : bool
        Print mesh parameters before building.
    **mesh_param :
        Geometry and density overrides. Parameter names follow Sipp & Lebedev (2007).

        - ``xinfa``: upstream domain extent (default -10.0)
        - ``xinf``: downstream domain extent (default 20.0)
        - ``yinf``: lateral domain half-width (default 8.0)
        - ``xplus``: length of interior zone downstream of cylinder (default 1.5)
        - ``yint``: half-height of middle zone (default 3.0)
        - ``lint``: half-size of interior zone (default 1.5)
        - ``inftol``: margin from exterior to middle zone (default 5.0)
        - ``inftola``: upstream margin (default 5.0)
        - ``n1``: cell density in interior zone (default 10.0)
        - ``n2``: cell density in middle zone (default 5.0)
        - ``n3``: cell density in exterior zone (default 1.0)
        - ``segments``: elements on cylinder boundary (default 360)
        - ``D``: cylinder diameter (default 1.0)

    Notes
    -----
    Mesh topology (not to scale)::

             xinfa                              xinf
             --------------------------------------------  <<<yinf
             |  yint>>>-------------------------        |
             |         |  lint      xplus      |        |
             |  lint>>>|  -------------        |        |
             |         |  |  O  n1    |  n2    |   n3   |
             |         |  -------------        |        |
             |<inftola>|                       |<inftol>|
             |         -------------------------        |
             --------------------------------------------
    """
    prm = {**_DEFAULT_MESH_PARAM, **mesh_param}
    if verbose:
        print("Making mesh with parameters:", prm)
    gmsh.initialize()
    try:
        gmsh.model.add("cylinder")
        _build_mesh(prm)
        gmsh.model.mesh.generate(2)
        _write_mesh(filename, formats)
    finally:
        gmsh.finalize()


def _build_mesh(prm):
    factory = gmsh.model.occ

    pul = factory.addPoint(prm["xinfa"], prm["yinf"], 0)
    pll = factory.addPoint(prm["xinfa"], -prm["yinf"], 0)
    plr = factory.addPoint(prm["xinf"], -prm["yinf"], 0)
    pur = factory.addPoint(prm["xinf"], prm["yinf"], 0)

    inftol, inftola = prm["inftol"], prm["inftola"]
    pmul = factory.addPoint(prm["xinfa"] + inftola, prm["yint"], 0)
    pmll = factory.addPoint(prm["xinfa"] + inftola, -prm["yint"], 0)
    pmlr = factory.addPoint(prm["xinf"] - inftol, -prm["yint"], 0)
    pmur = factory.addPoint(prm["xinf"] - inftol, prm["yint"], 0)

    piul = factory.addPoint(-prm["lint"], prm["lint"], 0)
    pill = factory.addPoint(-prm["lint"], -prm["lint"], 0)
    pilr = factory.addPoint(prm["xplus"], -prm["lint"], 0)
    piur = factory.addPoint(prm["xplus"], prm["lint"], 0)

    circ = factory.addCircle(0, 0, 0, prm["D"] / 2)

    line_ele = factory.addLine(pul, pll)
    line_elo = factory.addLine(pll, plr)
    line_eri = factory.addLine(plr, pur)
    line_eup = factory.addLine(pur, pul)

    line_mle = factory.addLine(pmul, pmll)
    line_mlo = factory.addLine(pmll, pmlr)
    line_mri = factory.addLine(pmlr, pmur)
    line_mup = factory.addLine(pmur, pmul)

    line_ile = factory.addLine(piul, pill)
    line_ilo = factory.addLine(pill, pilr)
    line_iri = factory.addLine(pilr, piur)
    line_iup = factory.addLine(piur, piul)

    rect_ext = factory.addCurveLoop([line_ele, line_elo, line_eri, line_eup])
    rect_mid = factory.addCurveLoop([line_mle, line_mlo, line_mri, line_mup])
    rect_in = factory.addCurveLoop([line_ile, line_ilo, line_iri, line_iup])
    circle = factory.addCurveLoop([circ])

    factory.addPlaneSurface([rect_ext, rect_mid])
    factory.addPlaneSurface([rect_mid, rect_in])
    factory.addPlaneSurface([rect_in, circle])

    factory.synchronize()

    ext_ = gmsh.model.getEntitiesInBoundingBox(prm["xinfa"], -prm["yinf"], -1, prm["xinf"], prm["yinf"], 1, dim=0)
    mid_ = gmsh.model.getEntitiesInBoundingBox(prm["xinfa"], -prm["yint"], -1, prm["xinf"], prm["yint"], 1, dim=0)
    in_ = gmsh.model.getEntitiesInBoundingBox(prm["xinfa"], -prm["lint"], -1, prm["xinf"], prm["lint"], 1, dim=0)
    gmsh.model.mesh.setSize(ext_, 1 / prm["n3"])
    gmsh.model.mesh.setSize(mid_, 1 / prm["n2"])
    gmsh.model.mesh.setSize(in_, 1 / prm["n1"])

    r = prm["D"] / 2 + 0.1
    cyl_ = gmsh.model.getEntitiesInBoundingBox(-r, -r, -1, r, r, 1, dim=0)
    gmsh.model.mesh.setSize(cyl_, min(1 / prm["n1"], 2 * math.pi / prm["segments"]))
