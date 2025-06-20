"""
----------------------------------------------------------------------
Lid-driven cavity
Nondimensional incompressible Navier-Stokes equations
Supercritical Hopf bifurcation near Re_c=7700
----------------------------------------------------------------------
This file converts meshes from .msh (gmsh) to .xdmf
----------------------------------------------------------------------
"""

from pathlib import Path

import meshio

data_dir = Path(__file__).parent / "data_input"

for i in range(5):
    msh_path = data_dir / f"lidcavity_{i}.msh"
    xdmf_path = data_dir / f"lidcavity_{i}.xdmf"

    if not msh_path.exists():
        print(f"File not found: {msh_path}")
        continue

    print(f"Converting {msh_path} -> {xdmf_path}")

    mesh = meshio.read(msh_path)

    meshxdmf = meshio.Mesh(
        points=mesh.points[:, :2],  # Keep only x and y coordinates
        cells=[c for c in mesh.cells if c.type == "triangle"],  # Keep only triangles
    )

    meshio.write(xdmf_path, meshxdmf)
