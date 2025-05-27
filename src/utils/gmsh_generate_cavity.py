"""
----------------------------------------------------------------------
Mesh generation
FILE UNDER WORK ______ NOT FUNCTIONAL
----------------------------------------------------------------------
"""

from __future__ import print_function

import meshio
import gmsh
import numpy as np
#from mesh_utils import *
#import mesh_utils

#import pdb


class MeshGenerator():
    def __init__(self, **mesh_param):
       # Initialize parameters
       self.set_mesh_param(default=True, **mesh_param)


    def make_mesh_all(self, filename, verbose=False):
        '''Build and write mesh with given parameters'''
        if verbose:
            print('Making mesh with parameters: ', self.mesh_param)
        self.init_gmsh()
        self.build_mesh()
        self.generate_mesh()
        self.write_mesh(filename)
        gmsh.finalize()

    
    def init_gmsh(self, name="cylinder"):
        '''Initialize mesh object in gmsh, with given name (unused)'''
        # Build mesh with gmsh
        gmsh.initialize()
        #gmsh.clear()
        gmsh.model.add(name)
    
    
    def set_mesh_param(self, default=False, **mesh_param):
        '''Define mesh geometry using input parameters
        Parameters are as in Sipp & Lebedev (2007)
        lengths: xinfa, xinf, yinf, xplus
        cell densities: n, n1+, n2+, n3+, n1-, n2-, n3-
        additional parameters (not in SL2007): yint, lint, inftola, inftol
        yint: size along y of middle zone
        lint: size along x and y of inside zone
        inftol: margin from exterior zone to middle zone
        Mesh topology is the following:
        
        
        -----------------------------------------------------------------  0.5
        |                                                               |  
        |                        n3+                                    |
        |---------------------------------------------------------------|  0.3 
        |                        n2+                                    |  
        |---------------------------------------------------------------|  0.15 
        |                        n1+                                    |   
        |        -------------------------------------------------------|  0.1
        |        |               n                                      |  
        ---------------------              ------------------------------  0
      -1.2     -0.6         |--------------| -0.1                      2.5
                            |    n1-       | 
                            |--------------| -0.2
                            |    n2-       | 
                            |--------------| -0.35 
                            |    n3-       |
                            |              |
                            ---------------- -1
                            0              1

        default=True resets parameters to default
        default parameters are:
        default_mesh_param = {
                'n': 50.0,
                'n1+':  25.0,
                'n2+':  10.0,
                'n3+':  5.0,
                'n1-':  25.0,
                'n2-':  10.0,
                'n3-':  5.0,
                }
        default in SL2007 are:
            n = 350
            n1+ = 200
            n2+ = 100
            n3+ = 100
            n1- = 150
            n2- = 100
            n3- = 50
        '''
        # Default parameters
        if default:
            default_mesh_param = {
                'n': 50.0,
                'n1+':  25.0,
                'n2+':  10.0,
                'n3+':  5.0,
                'n1-':  25.0,
                'n2-':  10.0,
                'n3-':  5.0,
                }
            self.mesh_param = default_mesh_param

        # Concatenate new parameter with old ones (may be default if default=True)
        new_param = {**self.mesh_param, **mesh_param}
        # Set mesh_param as dictionary
        self.mesh_param = new_param

    
    def build_mesh(self):  
        prm = self.mesh_param
        D = prm['D']

        # factory
        #factory = gmsh.model.geo
        factory = gmsh.model.occ
        
        ## points
        # exterior
        pul = factory.addPoint(prm['xinfa'], prm['yinf'],   0 ) # upper left
        pll = factory.addPoint(prm['xinfa'], -prm['yinf'],  0 ) # lower left
        plr = factory.addPoint(prm['xinf'], -prm['yinf'],   0 ) # lower right
        pur = factory.addPoint(prm['xinf'], prm['yinf'],    0 ) # upper right
        # mid
        inftol = prm['inftol']
        inftola = prm['inftola']
        pmul = factory.addPoint(prm['xinfa'] + inftola, prm['yint'],  0 ) # mid up left
        pmll = factory.addPoint(prm['xinfa'] + inftola, -prm['yint'], 0 ) # mid low left
        pmlr = factory.addPoint(prm['xinf'] - inftol, -prm['yint'],  0 ) # mid low right
        pmur = factory.addPoint(prm['xinf'] - inftol, prm['yint'],   0 ) # mid up right
        # inside
        piul = factory.addPoint(-prm['lint'], prm['lint'],  0 ) # inside up left
        pill = factory.addPoint(-prm['lint'], -prm['lint'], 0 ) # inside low left
        pilr = factory.addPoint(prm['xplus'], -prm['lint'], 0 ) # inside low right
        piur = factory.addPoint(prm['xplus'], prm['lint'],  0 ) # inside up right
        # cylinder points
        #pcl = factory.addPoint(-D/2, 0, 0) # circle left
        #pcc = factory.addPoint(0,    0, 0) # circle center
        #pcr = factory.addPoint(D/2,  0, 0) # circle right
        
        ## lines
        # cylinder
        #circ_up = factory.addCircleArc(pcl, pcc, pcr) # circle arc up
        #circ_lo = factory.addCircleArc(pcr, pcc, pcl) # circle arc down
        circ = factory.addCircle(0, 0, 0, D/2)# default is:, angle1=0, angle2=2*np.pi)
    
        #pdb.set_trace()
        # rectangles
        #line_le1 = factory.addLine(pul, pmul)
        #line_lo1 = factory.addLine(pmul, pmur)
        #line_ri1 = factory.addLine(pmur, pur)
        #line_up1 = factory.addLine(pur, pul)
        #line_l = factory.addLine(pul, pll)
        #line_r = factory.addLine(plr, pur)
        #
        #line_le2 = factory.addLine(pmll, pll)
        #line_lo2 = factory.addLine(pll, plr)
        #line_ri2 = factory.addLine(plr, pmlr)
        #line_up2 = factory.addLine(pmlr, pmll)
       
        # line exterior
        line_ele = factory.addLine(pul, pll) 
        line_elo = factory.addLine(pll, plr) 
        line_eri = factory.addLine(plr, pur) 
        line_eup = factory.addLine(pur, pul) 
        
        # line mid
        line_mle = factory.addLine(pmul, pmll)
        line_mlo = factory.addLine(pmll, pmlr)
        line_mri = factory.addLine(pmlr, pmur)
        line_mup = factory.addLine(pmur, pmul)
        
        # line interior
        line_ile = factory.addLine(piul, pill)
        line_ilo = factory.addLine(pill, pilr)
        line_iri = factory.addLine(pilr, piur)
        line_iup = factory.addLine(piur, piul)
        
        # curve loops
        #rect_ext_up = factory.addCurveLoop([line_le1, line_lo1, line_ri1, line_up1])
        #rect_ext_lo = factory.addCurveLoop([line_le2, line_lo2, line_ri2, line_up2])
        #rect_ext = factory.addCurveLoop([line_l, line_lo2, line_r, line_up1])
        rect_ext = factory.addCurveLoop([line_ele, line_elo, line_eri, line_eup])
        rect_mid = factory.addCurveLoop([line_mle, line_mlo, line_mri, line_mup])
        rect_in = factory.addCurveLoop([line_ile, line_ilo, line_iri, line_iup])
        #circle = factory.addCurveLoop([circ_up, circ_lo])
        circle = factory.addCurveLoop([circ])
        
        # surfaces
        #factory.addPlaneSurface([rect_ext_up])
        #factory.addPlaneSurface([rect_ext_lo])
        factory.addPlaneSurface([rect_ext, rect_mid])
        factory.addPlaneSurface([rect_mid, rect_in])
        factory.addPlaneSurface([rect_in, circle])
        
        # sync
        factory.synchronize()
    
        # set mesh size
        ext_ = gmsh.model.getEntitiesInBoundingBox(prm['xinfa'], 
            -prm['yinf'], -1, prm['xinf'], prm['yinf'], 1, dim=0)    
        mid_ = gmsh.model.getEntitiesInBoundingBox(prm['xinfa'], 
            -prm['yint'], -1, prm['xinf'], prm['yint'], 1, dim=0)    
        in_ =  gmsh.model.getEntitiesInBoundingBox(prm['xinfa'], 
            -prm['lint'], -1, prm['xinf'], prm['lint'], 1, dim=0)    
        gmsh.model.mesh.setSize(ext_, 1/prm['n3'])
        gmsh.model.mesh.setSize(mid_, 1/prm['n2'])
        gmsh.model.mesh.setSize(in_,  1/prm['n1'])
    
        cyl_ = gmsh.model.getEntitiesInBoundingBox(-1.1, -1.1, -1, 1.1, 1.1, 1, dim=0)
        gmsh.model.mesh.setSize(cyl_, np.min([1/prm['n1'], 2*np.pi/prm['segments']])) 
            # size so that n segments = 2 pi, or size of elements in interior zone (if smaller) 
    
    
    def generate_mesh(self):
        '''Generate mesh in 2D'''
        ## generate
        gmsh.model.mesh.generate(2)
    
    
    def write_mesh(self, filename='mesh_autoname', formats=['xml', 'xdmf']):
        '''Write mesh file as xml(legacy)+xdmf by default'''
        # in gmsh.msh
        #print('found name of file: ', filename)
        print('Writing mesh as file: ', filename, ' with formats: ', formats)
        gmsh.write(filename + '.msh')
        # convert with meshio
        mesh = meshio.read(filename+'.msh')
        mesh.prune_z_0()
        # as xml
        if 'xml' in formats:
            mesh.write(filename+'.xml')
        # as xdmf
        if 'xdmf' in formats:
            meshxdmf = meshio.Mesh(
                points=mesh.points, 
                cells={'triangle': mesh.cells_dict['triangle']})
            meshio.write(filename+'.xdmf', meshxdmf)


