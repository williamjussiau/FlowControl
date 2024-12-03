"""
----------------------------------------------------------------------
Mesh generation tests
----------------------------------------------------------------------
"""

from __future__ import print_function

import meshio


####################################################################################
def convert_mesh_xml2xdmf(filename):
    """Convert mesh from xml to xdmf """
    xmlfile = filename + '.xml'
    xdmffile = filename + '.xdmf'
    # read xml
    print('Reading xml file... ', xmlfile)
    meshxml = meshio.read(xmlfile)
    # write xdmf
    meshxdmf = meshio.Mesh(
        points=meshxml.points, 
        cells={'triangle': meshxml.cells_dict['triangle']})
    print('Writing xdmf/h5 file... ', xdmffile)
    meshio.write(filename+'.xdmf', meshxdmf)
####################################################################################



