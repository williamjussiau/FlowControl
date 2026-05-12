## Mesh generation
<img src="https://github.com/user-attachments/assets/194818f3-9cd5-4233-a10c-aaeaf617063f" alt="gmsh logo" height="120"/>
<img src="https://github.com/user-attachments/assets/5b589d43-6fcf-473c-a67a-b959e4c94b4f" alt="meshio logo" height="100"/>

No meshing tools are shipped with this code, but [gmsh](https://gmsh.info/) (and [its Python API](https://pypi.org/project/gmsh/)) is suggested for generating meshes. The mesh should be exported to xdmf format, which can be generated thanks to [meshio](https://github.com/nschloe/meshio/tree/main). 

Beware that the user is the sole responsible for the coherence of the mesh with respect to their definition of the boundaries (overridden `_make_boundaries()` method) and boundary conditions (overridden `_make_bcs()` method).

For mesh conversion, the tool FEconv is also suggested - see [the Github repo](https://github.com/victorsndvg/FEconv) and [the website](https://victorsndvg.github.io/FEconv/description.xhtml). It was especially useful to convert meshes from FreeFem++ to FEniCS, in order to replicate results.


## Visualization
<img src="https://github.com/user-attachments/assets/6658c108-1506-4057-92d7-a7a64d978b50" alt="paraview logo" height="90"/>

[Paraview](https://www.paraview.org/) is suggested for visualizations, whether it be for csv timeseries or fields saved as xdmf.