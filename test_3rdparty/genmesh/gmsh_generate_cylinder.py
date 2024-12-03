"""
----------------------------------------------------------------------
Mesh generation tests
----------------------------------------------------------------------
"""

from __future__ import print_function
#from dolfin import *

import meshio
import gmsh
import numpy as np

import pdb

def make_mesh_all(filename, **mesh_param):
    '''Build and write mesh with given parameters'''
    init_mesh()
    build_mesh(**mesh_param)
    generate_mesh()
    write_mesh(filename)
    gmsh.finalize()

def init_mesh(name="cylinder"):
    '''Initialize mesh object in gmsh, with given name (unused)'''
    # Build mesh with gmsh
    gmsh.initialize()
    #gmsh.clear()
    gmsh.model.add(name)

def build_mesh(**mesh_param):
    '''Define mesh geometry using input parameters
    Parameters are as in Sipp & Lebedev (2007)
    lengths: xinfa, xinf, yinf, xplus
    cell densities: n1, n2, n3
    additional parameters (not in SL2007): yint, lint'''
    # Default or **kwargs
    default_mesh_param = {
        'xinfa': -10.0,
        'xinf':  20.0,
        'yinf':  8.0,
        'xplus': 5.0,
        'yint':  3.0,
        'lint':  1.5,
        'n1':    10.0,
        'n2':    5.0,
        'n3':    1.0,
        'segments': 120} # 120 segments makes approx 360 elts around the cylinder (idk)
    prm = {**default_mesh_param, **mesh_param}
    print('param default are:', default_mesh_param)
    print('param input are  :', mesh_param)
    print('param concat are :', prm)
    
    # diameter should be 1
    D = 1
    
    # factory
    factory = gmsh.model.geo
    factory = gmsh.model.occ
    
    ## points
    # exterior
    pul = factory.addPoint(prm['xinfa'], prm['yinf'], 0 ) # upper left
    pll = factory.addPoint(prm['xinfa'], -prm['yinf'], 0) # lower left
    plr = factory.addPoint(prm['xinf'], -prm['yinf'], 0 ) # lower right
    pur = factory.addPoint(prm['xinf'], prm['yinf'], 0  ) # upper right
    # mid
    pmul = factory.addPoint(prm['xinfa'], prm['yint'], 0 ) # mid up left
    pmll = factory.addPoint(prm['xinfa'], -prm['yint'], 0) # mid low left
    pmlr = factory.addPoint(prm['xinf'], -prm['yint'], 0 ) # mid low right
    pmur = factory.addPoint(prm['xinf'], prm['yint'], 0  ) # mid up right
    # inside
    piul = factory.addPoint(-prm['lint'], prm['lint'], 0  ) # inside up left
    pill = factory.addPoint(-prm['lint'], -prm['lint'], 0 ) # inside low left
    pilr = factory.addPoint(prm['xplus'], -prm['lint'], 0 ) # inside low right
    piur = factory.addPoint(prm['xplus'], prm['lint'], 0  ) # inside up right
    # cylinder points
    #pcl = factory.addPoint(-D/2, 0, 0) # circle left
    #pcc = factory.addPoint(0,    0, 0) # circle center
    #pcr = factory.addPoint(D/2,  0, 0) # circle right
    
    ## lines
    # cylinder
    #circ_up = factory.addCircleArc(pcl, pcc, pcr) # circle arc up
    #circ_lo = factory.addCircleArc(pcr, pcc, pcl) # circle arc down
    circ = gmsh.model.occ.addCircle(0, 0, 0, D/2)# default is:, angle1=0, angle2=2*np.pi)

    #pdb.set_trace()
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
    #circle = factory.addCurveLoop([circ_up, circ_lo])
    circle = factory.addCurveLoop([circ])
    
    # surfaces
    factory.addPlaneSurface([rect_ext_up])
    factory.addPlaneSurface([rect_ext_lo])
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
    gmsh.model.mesh.setSize(cyl_, 2/prm['segments']) # factor 2 should be removed
####################################################################################



####################################################################################
def generate_mesh():
    '''Generate mesh in 2D'''
    ## generate
    gmsh.model.mesh.generate(2)
####################################################################################



####################################################################################
def write_mesh(filename='mesh_autoname', formats=['xml', 'xdmf']):
    '''Write mesh file as xml(legacy)+xdmf by default'''
    # in gmsh.msh
    print('found name of file: ', filename)
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
####################################################################################


####################################################################################
def convert_mesh_xml2xdmf(xmlfile):
    """Convert mesh from xml to xdmf
    No extension is provided as input"""
    # Add extension if needed
    xmlfile = set_file_extension(xmlfile, '.xml')
    # Resulting file is always "xmlfile.xml"
    # Create xdmf files
    xdmffile =  set_file_extension(xmlfile, '.xdmf')
    # read xml
    print('Reading xml file... ', xmlfile)
    meshxml = meshio.read(xmlfile)
    # write xdmf
    meshxdmf = meshio.Mesh(
        points=meshxml.points, 
        cells={'triangle': meshxml.cells_dict['triangle']})
    print('Writing xdmf/h5 file... ', xdmffile)
    meshio.write(xdmffile, meshxdmf)


def convert_mesh_msh2xdmf(mshfile):
    mshfile = set_file_extension(mshfile, '.msh')
    mesh = meshio.read(mshfile)
    mesh.prune_z_0()
    # as xdmf
    meshxdmf = meshio.Mesh(
        points=mesh.points, 
        cells={'triangle': mesh.cells_dict['triangle']})
    meshio.write(set_file_extension(mshfile, '.xdmf'), meshxdmf)


def convert_mesh_msh2xml(mshfile):
    mshfile = append_extension(mshfile, '.msh')
    mesh = meshio.read(mshfile)
    mesh.prune_z_0()
    # as xml
    mesh.write(set_file_extension(mshfile)+'.xml')

def get_file_wo_extension(filename):
    '''Return filename without its extension
    Works with double extensions (.tar.gz)'''
    return filename.split('.')[0]

def set_file_extension(filename, extension):
    '''Append extension to file if not already present
    extension (input) should contain the dot .
    Example: extension=.xdmf
    Does not replace full double extensions (.tar.gz)'''
    return get_file_wo_extension(filename) + extension







