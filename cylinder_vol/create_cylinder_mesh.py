'''
Create mesh for cylinder

Procedure is the following:
    import meshing module
    create mesh generator
    set mesh generator parameters
    build mesh
    export mesh
    check on Paraview if necessary

Parameters of the mesh are defined in gmsh_generate_cylinder.py as such:
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
See help(set_mesh_parameter) for more info
Defaults are:
    'xinfa': -10.0,
    'xinf':  20.0,
    'yinf':  8.0,
    'xplus': 1.5,
    'yint':  3.0,
    'lint':  1.5,
    'inftol': 5.0,
    'inftola': 5.0,
    'n1':    10.0, # density in interior zone
    'n2':    5.0, # density in middle zone
    'n3':    1.0, # density in exterior zone 
    'segments': 360, # 1 element per degree on cylinder
    'D': 1 # cylinder diameter 
'''

import gmsh_generate_cylinder as gm
mg = gm.MeshGenerator()
mg.set_mesh_param(default=False,
                  xinfa=-10,
                  xinf=20,
                  yinf=10,
                  inftol=5,
                  inftola=5,
                  n1=6,
                  n2=3,
                  n3=1,
                  segments=540,
                  yint=3.0,
                  lint=1.5,
                  xplus=3.0)
filename = '/stck/wjussiau/fenics-python/mesh/O1'
mg.make_mesh_all(filename, verbose=True)



