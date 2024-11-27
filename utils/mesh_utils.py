"""
----------------------------------------------------------------------
Mesh generation and conversion
----------------------------------------------------------------------
"""

from __future__ import print_function

import meshio
import gmsh
import numpy as np

import pdb


####################################################################################
def convert_mesh_xml2xdmf(xmlfile):
    """Convert mesh from xml to xdmf
    The input is provided without extension"""
    # Add extension if needed
    xmlfile = set_file_extension(xmlfile, '.xml')
    # Resulting file is always "xmlfile.xml"
    # Create xdmf files
    xdmffile = set_file_extension(xmlfile, '.xdmf')
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
    '''Convert mesh file from .msh to .xdmf
    msh is GMSH msh, not FREEFEM msh
    To convert FREEFEM msh, use FEconv executable
    The input is provided without extension
    '''
    mshfile = set_file_extension(mshfile, '.msh')
    mesh = meshio.read(mshfile)
    mesh.prune_z_0()
    # as xdmf
    meshxdmf = meshio.Mesh(
        points=mesh.points, 
        cells={'triangle': mesh.cells_dict['triangle']})
    meshio.write(set_file_extension(mshfile, '.xdmf'), meshxdmf)


def convert_mesh_msh2xml(mshfile):
    '''Convert mesh file from .msh to .xml
    The input is provided without extension
    '''
    mshfile = set_file_extension(mshfile, '.msh')
    mesh = meshio.read(mshfile)
    mesh.prune_z_0()
    # as xml
    meshio.write(set_file_extension(mshfile, '.xml'), mesh)


def convert_mesh_vtu2xdmf(mshfile):
    '''Convert mesh file from .vtu to .xdmf
    vtu is the output format of FEconv
    The input is provided without extension
    '''
    mshfile = set_file_extension(mshfile, '.vtu') 
    mesh = meshio.read(mshfile)
    mesh.prune_z_0()
    # as xdmf
    meshio.write(set_file_extension(mshfile, '.xdmf'), mesh)


def get_file_wo_extension(filename):
    '''Return filename without its extension
    Works with double extensions (.tar.gz)
    but not with files that have . in their name'''
    return filename.split('.')[0]


def set_file_extension(filename, extension):
    '''Append extension to file if not already present
    extension (input) should contain the dot .
    Example: extension=.xdmf
    Does not replace full double extensions (.tar.gz)'''
    return get_file_wo_extension(filename) + extension
####################################################################################






