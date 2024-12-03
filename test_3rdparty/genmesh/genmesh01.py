"""
----------------------------------------------------------------------
Mesh generation tests
----------------------------------------------------------------------
"""

from __future__ import print_function
from dolfin import *


import numpy as np
import sys

import time


from mshr import Sphere, Box, Cylinder, Circle, Rectangle, generate_mesh

###############################################################################
# Create mesh
#xinf = 10 # 20 # 20
#yinf = 3 # 5 # 8
#xinfa = -3 # -5 # -10
#channel = Rectangle(Point(xinfa, -yinf), Point(xinf, yinf))
#cylinder = Circle(Point(0.0, 0.0), d/2)
#domain = channel - cylinder
#mesh = generate_mesh(domain, 32) # was 64

testcase = 0

if testcase==0:
    print('helo')


if testcase==1:
    s1 = Sphere(Point(0, 0, 0), 1.4)
    b1 = Box(Point(-1, -1, -1), Point(1, 1, 1))
    c1 = Cylinder(Point(-2, 0, 0), Point(2, 0, 0), 0.8, 0.8)
    c2 = Cylinder(Point(0, -2, 0), Point(0, 2, 0), 0.8, 0.8)
    c3 = Cylinder(Point(0, 0, -2), Point(0, 0, 2), 0.8, 0.8)
    
    geometry = s1*b1 - (c1 + c2 + c3)
    mesh = generate_mesh(geometry, 64)

    File("/stck/wjussiau/fenics-python/genmesh/results/classic.pvd") << mesh


if testcase==2:
    domain = Rectangle(Point(0., 0.), Point(5., 5.)) \
    - Rectangle(Point(2., 1.25), Point(3., 1.75)) \
    - Circle(Point(1, 4), .25) \
    - Circle(Point(4, 4), .25)
                           
    domain.set_subdomain(1, Rectangle(Point(1., 1.), Point(4., 3.)))
    domain.set_subdomain(2, Rectangle(Point(2., 2.), Point(3., 4.)))

    print("Verbose output of 2D geometry:")
    info(domain, True)
 
    # Generate and plot mesh
    mesh = generate_mesh(domain, 45)
    print(mesh)
 
    # Convert subdomains to mesh function for plotting
    mf = MeshFunction("size_t", mesh, 2, mesh.domains())

    File("/stck/wjussiau/fenics-python/genmesh/results/subd2d.pvd") << mesh

# Save to file and plot


