"""Mesh generation and conversion utilities (meshio-based)."""

import logging
from pathlib import Path

import meshio

logger = logging.getLogger(__name__)

_EXT_XML = ".xml"
_EXT_MSH = ".msh"
_EXT_VTU = ".vtu"
_EXT_XDMF = ".xdmf"


def convert_mesh_xml2xdmf(xmlfile: str | Path) -> None:
    """Convert mesh from .xml to .xdmf."""
    src = Path(xmlfile).with_suffix(_EXT_XML)
    dst = src.with_suffix(_EXT_XDMF)
    logger.info("Reading xml file: %s", src)
    meshxml = meshio.read(src)
    meshxdmf = meshio.Mesh(
        points=meshxml.points, cells={"triangle": meshxml.cells_dict["triangle"]}
    )
    logger.info("Writing xdmf/h5 file: %s", dst)
    meshio.write(dst, meshxdmf)


def convert_mesh_msh2xdmf(mshfile: str | Path) -> None:
    """Convert GMSH .msh to .xdmf.
    For FreeFEM .msh, use FEconv instead."""
    src = Path(mshfile).with_suffix(_EXT_MSH)
    mesh = meshio.read(src)
    meshxdmf = meshio.Mesh(
        points=mesh.points[:, :2], cells={"triangle": mesh.cells_dict["triangle"]}
    )
    meshio.write(src.with_suffix(_EXT_XDMF), meshxdmf)


def convert_mesh_msh2xml(mshfile: str | Path) -> None:
    """Convert GMSH .msh to .xml."""
    src = Path(mshfile).with_suffix(_EXT_MSH)
    mesh = meshio.read(src)
    mesh.prune_z_0()
    meshio.write(src.with_suffix(_EXT_XML), mesh)


def convert_mesh_vtu2xdmf(mshfile: str | Path) -> None:
    """Convert FEconv .vtu output to .xdmf."""
    src = Path(mshfile).with_suffix(_EXT_VTU)
    mesh = meshio.read(src)
    mesh.prune_z_0()
    meshio.write(src.with_suffix(_EXT_XDMF), mesh)
