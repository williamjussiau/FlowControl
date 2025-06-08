from pathlib import Path

import dolfin


def make_mesh(nx, ny, export=True, filename="mesh.xdmf", option="left"):
    """Make simple unit square mesh with dolfin native function
    The mesh can be exported to a xdmf file"""
    mesh = dolfin.UnitSquareMesh(nx, ny, option)  # options: left, right, crossed

    if export:
        meshpath = Path.cwd() / "src" / "examples" / "lidcavity" / "data_input"
        with dolfin.XDMFFile(str(meshpath / filename)) as meshfile:
            print(f"--- Exported mesh file in {meshpath} ---")
            meshfile.write(mesh)

    return mesh


if __name__ == "__main__":
    # make_mesh(128, 128, "mesh128.xdmf", "crossed")
    pass
