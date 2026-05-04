import gmsh
import meshio


def _bbox(x0, y0, x1, y1):
    return gmsh.model.getEntitiesInBoundingBox(x0, y0, -1, x1, y1, 1, dim=0)


def _write_mesh(filename, formats):
    gmsh.write(filename + ".msh")
    mesh = meshio.read(filename + ".msh")
    mesh.points = mesh.points[:, :2]
    if "xml" in formats:
        mesh.write(filename + ".xml")
    if "xdmf" in formats:
        meshxdmf = meshio.Mesh(points=mesh.points, cells={"triangle": mesh.cells_dict["triangle"]})
        meshio.write(filename + ".xdmf", meshxdmf)
