"""
----------------------------------------------------------------------
Mesh generation tests
----------------------------------------------------------------------
"""

from __future__ import print_function
from dolfin import *

import meshio
import gmsh
import numpy as np

###########################################################
# Mesh conversion only
convert = False
if convert:
    mesh = meshio.read('cylinder.msh')
    mesh.write('cylinder.xdmf')
###########################################################





###########################################################
# Build mesh with gmsh
gmsh.clear()
gmsh.initialize()
gmsh.model.add("cylinder")

# Define variables
xinfa = -10.0
xinf = 20.0
yinf = 8.0
xplus = 5.0
yint = 3.0
lint = 1.5

D = 1

n1 = 10.0
n2 = 5.0
n3 = 2.0

# factory
factory = gmsh.model.geo

## points
# exterior
pul = factory.addPoint(xinfa, yinf, 0 ) # upper left
pll = factory.addPoint(xinfa, -yinf, 0) # lower left
plr = factory.addPoint(xinf, -yinf, 0 ) # lower right
pur = factory.addPoint(xinf, yinf, 0  ) # upper right
# mid
pmul = factory.addPoint(xinfa, yint, 0 ) # mid up left
pmll = factory.addPoint(xinfa, -yint, 0) # mid low left
pmlr = factory.addPoint(xinf, -yint, 0 ) # mid low right
pmur = factory.addPoint(xinf, yint, 0  ) # mid up right
# inside
piul = factory.addPoint(-lint, lint, 0  ) # inside up left
pill = factory.addPoint(-lint, -lint, 0 ) # inside low left
pilr = factory.addPoint(xplus, -lint, 0 ) # inside low right
piur = factory.addPoint(xplus, lint, 0  ) # inside up right
# cylinder points
pcl = factory.addPoint(-D/2, 0, 0) # circle left
pcc = factory.addPoint(0,    0, 0) # circle center
pcr = factory.addPoint(D/2,  0, 0) # circle right

## lines
# cylinder
circ_up = factory.addCircleArc(pcl, pcc, pcr) # circle arc up
circ_lo = factory.addCircleArc(pcr, pcc, pcl) # circle arc down
# rectangles
line_le1 = factory.addLine(pul, pmul)
line_lo1 = factory.addLine(pmul, pmur)
line_ri1 = factory.addLine(pmur, pur)
line_up1 = factory.addLine(pur, pul)

line_le2 = factory.addLine(pmll, pll)
line_lo2 = factory.addLine(pll, plr)
line_ri2 = factory.addLine(plr, pmlr)
line_up2 = factory.addLine(pmlr, pmll)

line_mle = factory.addLine(pmul, pmll)
line_mlo = factory.addLine(pmll, pmlr)
line_mri = factory.addLine(pmlr, pmur)
line_mup = factory.addLine(pmur, pmul)

line_ile = factory.addLine(piul, pill)
line_ilo = factory.addLine(pill, pilr)
line_iri = factory.addLine(pilr, piur)
line_iup = factory.addLine(piur, piul)

# curve loops
rect_ext_up = factory.addCurveLoop([line_le1, line_lo1, line_ri1, line_up1])
rect_ext_lo = factory.addCurveLoop([line_le2, line_lo2, line_ri2, line_up2])
rect_mid = factory.addCurveLoop([line_mle, line_mlo, line_mri, line_mup])
rect_in = factory.addCurveLoop([line_ile, line_ilo, line_iri, line_iup])
circle = factory.addCurveLoop([circ_up, circ_lo])

# surfaces
factory.addPlaneSurface([rect_ext_up])
factory.addPlaneSurface([rect_ext_lo])
factory.addPlaneSurface([rect_mid, rect_in])
factory.addPlaneSurface([rect_in, circle])

# sync
factory.synchronize()

# set mesh size
ext_ = gmsh.model.getEntitiesInBoundingBox(xinfa, -yinf, -1, xinf, yinf, 1, dim=0)    
mid_ = gmsh.model.getEntitiesInBoundingBox(xinfa, -yint, -1, xinf, yint, 1, dim=0)    
in_ =  gmsh.model.getEntitiesInBoundingBox(xinfa, -lint, -1, xinf, lint, 1, dim=0)    
gmsh.model.mesh.setSize(ext_, 1/n3)
gmsh.model.mesh.setSize(mid_, 1/n2)
gmsh.model.mesh.setSize(in_,  1/n1)

## generate
gmsh.model.mesh.generate(2)

## write
# in gmsh.msh
filename = 'results/automesh'
gmsh.write(filename+'.msh')
# convert with meshio
mesh = meshio.read(filename+'.msh')
# as xml
mesh.write(filename+'.xml')
# as xdmf
meshxdmf = meshio.Mesh(points=mesh.points, cells={'triangle': mesh.cells_dict['triangle']})
meshio.write(filename+'.xdmf', meshxdmf)


###########################################################
meshdolfin = Mesh()
with XDMFFile(filename+'.xdmf') as infile:
    infile.read(meshdolfin)



