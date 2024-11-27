"""
FEniCS tutorial demo program: Incompressible Navier-Stokes equations
for let around a cylinder using the Incremental Pressure Correction
Scheme (IPCS).

  u' + u . nabla(u)) - div(sigma(u, p)) = f
                                 div(u) = 0


----------------------------------------------------------------------
Equations were made non-dimensional with Reynolds numbers
Equations from DENIS SIPP in FreeFem++
----------------------------------------------------------------------
"""

from __future__ import print_function
from dolfin import *
import ufl
from mshr import Rectangle, Circle, generate_mesh
import numpy as np
import sys
import os 
import time
import pandas as pd
#from fenicstools.Probe import Probe, Probes
#from mpi4py import MPI as mpi 
import sympy as sp
import functools
from scipy import signal as ss
import control
import scipy.sparse as spr
import scipy.io as sio
import matplotlib.pyplot as plt
import petsc4py
from petsc4py import PETSc

import pdb

import importlib
import utils_flowsolver as flu
importlib.reload(flu)

# FEniCS log level
set_log_level(LogLevel.INFO) # DEBUG TRACE PROGRESS INFO


class FlowSolver():
    '''Base class for calculating flow
    Is instantiated with several structures (dicts) containing parameters
    See method .step and main for time-stepping (possibly actuated)
    Contain methods for frequency-response computation'''
    def __init__(self, 
                 params_flow={'Re': 100, 'uinf': 1, 'd': 1, 'init_pert': 0,
                    'set_initial_state': False, 'init_state': 0}, 
                 params_time={'dt': 0.01, 'Tstart': 0, 'num_steps': 10, 'Tc': 1e3}, 
                 params_save={'save_every': 10, 'save_every_old': 10, 'compute_norms': True},
                 params_solver={'solver_type': 'Krylov', 'equations': 'ipcs'},
                 params_mesh={},
                 verbose=True):

        # Probably bad practice
        # Unwrap all dictionaries into self.attribute
        alldict = {**params_flow, 
                   **params_time, 
                   **params_save, 
                   **params_solver,
                   **params_mesh}
        for key, item in alldict.items(): # all input dicts
            setattr(self, key, item) # set corresponding attribute

        self.verbose = verbose
        # Parameters
        self.r = self.d/2
        self.nu = self.uinf*self.d/self.Re # dunnu touch
        # Time
        self.Tf = self.num_steps*self.dt # final time
        # Sensors
        self.sensor_nr = self.sensor_location.shape[0]
        # Initial state default
        self.initial_state = None

        # Save
        # params_save should contain savedir0
        self.define_paths() 
        # shortcuts (self-explanatory)
        self.make_mesh()
        self.make_function_spaces()
        self.make_boundaries()
        self.make_actuator()
        self.make_bcs()
        # important: make sensor after making bcs
        self.make_sensor()

        # for energy computation
        self.create_indicator_function()
        self.u_ = Function(self.V)
        self.u_restrict = Function(self.V)

    def define_paths(self):
        '''Define attribute (dict) containing useful paths (save, etc.)'''
        # Files location directory is params_save['savedir0']
        # dunnu touch below
        savedir0 = self.savedir0
        Tstart = self.Tstart
        file_extension = '' if Tstart==0 else '_restart'+str(np.round(Tstart, decimals=3)).replace('.',',')
        file_extension_xdmf = file_extension + '.xdmf'
        file_extension_csv = file_extension + '.csv'
        filename_u0 = savedir0 + 'steady/u0.xdmf'
        filename_p0 = savedir0 + 'steady/p0.xdmf'
        filename_u = savedir0 + 'u.xdmf'
        filename_uprev = savedir0 + 'uprev.xdmf'
        filename_p = savedir0 + 'p.xdmf'
        filename_u_restart = savedir0 + 'u' + file_extension_xdmf
        filename_uprev_restart = savedir0 + 'uprev'+ file_extension_xdmf 
        filename_p_restart = savedir0 + 'p'+ file_extension_xdmf
        filename_timeseries = savedir0 + 'timeseries1D' + file_extension_csv # 1D data as pd.DataFrame
        # Note:
        # extension is .xdmf if no restart
        # else _restart($Tstart).xdmf

        self.paths = {'u0': filename_u0,
                      'p0': filename_p0,
                      'u' : filename_u,
                      'p' : filename_p,
                      'uprev' : filename_uprev,
                      'u_restart': filename_u_restart,
                      'uprev_restart' : filename_uprev_restart,
                      'p_restart': filename_p_restart,
                      'timeseries': filename_timeseries,
                      'mesh': self.meshpath}
    

    def make_mesh(self):
        '''Define mesh
        params_mesh contains either name of existing mesh
        or geometry parameters: xinf, yinf, xinfa, nx...'''
        # Set params
        genmesh = self.genmesh
        meshdir = self.paths['mesh'] #'/stck/wjussiau/fenics-python/mesh/' 
        xinf = self.xinf #20 # 20 # 20
        yinf = self.yinf #8 # 5 # 8
        xinfa = self.xinfa #-5 # -5 # -10
        # dunnu touch below
        # Working as follows:
        # if genmesh: 
        #   if does not exist (with given params): generate with meshr
        #   and prepare to not read file (because mesh is already in memory)
        # else:
        #   set file name and prepare to read file
        # read file
        readmesh = True
        if genmesh:
            nx = self.nx #32
            meshname = 'cylinder_' + str(nx) + '.xdmf'
            meshpath = os.path.join(meshdir, meshname)
            if not os.path.exists(meshpath) or self.remesh:
                if self.verbose:
                    print('Mesh does not exist @:', meshpath)
                    print('-- Creating mesh...')
                channel = Rectangle(Point(xinfa, -yinf), Point(xinf, yinf))
                cyl = Circle(Point(0.0, 0.0), self.d/2, segments=self.segments)
                domain = channel - cyl
                mesh = generate_mesh(domain, nx)
                with XDMFFile(MPI.comm_world, meshpath) as fm:
                    fm.write(mesh)
                readmesh = False
        else:
            #if self.meshname is None:
            #    meshname = 'cylinder_avg.xdmf'
            #else:
            meshname = self.meshname
        
        # if mesh was not generated on the fly, read file
        if readmesh:
            mesh = Mesh(MPI.comm_world)
            meshpath = os.path.join(meshdir, meshname)
            if self.verbose:
                print('Mesh exists @: ', meshpath)
                print('--- Reading mesh...')
            with XDMFFile(MPI.comm_world, meshpath) as fm:
                fm.read(mesh)
            #mesh = Mesh(MPI.comm_world, meshpath) # if xml

        if self.verbose:
            print('Mesh has: %d cells' %(mesh.num_cells()) )

        # assign mesh & facet normals
        self.mesh = mesh
        self.n  = FacetNormal(mesh)


    def make_function_spaces(self):
        '''Define function spaces (u, p) = (CG2, CG1)'''
        # Function spaces on mesh
        # taylor-hood
        Ve = VectorElement('CG', self.mesh.ufl_cell(), 2) # was CG2
        Pe = FiniteElement('CG', self.mesh.ufl_cell(), 1) # was CG1
        We = MixedElement([Ve, Pe])
        self.V = FunctionSpace(self.mesh, Ve)
        self.P = FunctionSpace(self.mesh, Pe)
        self.W = FunctionSpace(self.mesh, We) 

	### mini (bubble) elements
        ##P1 = FiniteElement("Lagrange", self.mesh.ufl_cell(), 1)
        ##B = FiniteElement("Bubble",   self.mesh.ufl_cell(), self.mesh.topology().dim() + 1)
        ##V = VectorElement(NodalEnrichedElement(P1, B))
        ##Q = P1
        ##self.V = FunctionSpace(self.mesh, V) 
        ##self.P = FunctionSpace(self.mesh, P1)
        ##self.W = FunctionSpace(self.mesh, V * Q)

        if self.verbose:
            print('Function Space [V(CG2), P(CG1)] has: %d DOFs' %(self.W.dim()))


    def make_boundaries(self):
        '''Define boundaries (inlet, outlet, walls, and so on)
        Geometry and boundaries are the following:
                         sf
        ------------------------------------------
        |                                        |
      in|                                        |out
        |                                        |   
        ------x0ns---      -----x0ns+-------------
           sf     ns|      | ns           sf
                    |      |
                    |      |
                    --------
                       ns
        '''
        MESH_TOL = DOLFIN_EPS 
        L = self.d
        D = self.d
        xinfa = self.xinfa
        xinf = self.xinf
        yinf = self.yinf
        x0ns_left = -0.4 # sf left from x0ns, ns right from x0ns
        x0ns_right = 1.75 # ns left from x0ns, sf right from x0ns
        # Define as compiled subdomains

        ## Inlet
        inlet = CompiledSubDomain('on_boundary && \
                near(x[0], xinfa, MESH_TOL)', 
                    xinfa=xinfa, MESH_TOL=MESH_TOL)

        ## Outlet
        outlet = CompiledSubDomain('on_boundary && \
                near(x[0], xinf, MESH_TOL)',
                    xinf=xinf, MESH_TOL=MESH_TOL)
                    
        ## Upper wall
        upper_wall = CompiledSubDomain('on_boundary && \
                     near(x[1], yinf, MESH_TOL)', 
                     yinf=yinf, MESH_TOL=MESH_TOL)

        ## Open cavity
        # cavity left
        class bnd_cavity_left(SubDomain):
            '''Left wall of cavity'''
            def inside(self, x, on_boundary):
                return on_boundary and between(x[1], (-D, 0)) and near(x[0], 0)
        cavity_left = bnd_cavity_left()
        # cavity bottom
        class bnd_cavity_botm(SubDomain):
            '''Bottom wall of cavity'''
            def inside(self, x, on_boundary):
                return on_boundary and between(x[0], (0, L)) and near(x[1], -D)
        cavity_botm = bnd_cavity_botm()
        # cavity right
        class bnd_cavity_right(SubDomain):
            '''Right wall of cavity'''
            def inside(self, x, on_boundary):
                return on_boundary and between(x[1], (-D, 0)) and near(x[0], L)
        cavity_right = bnd_cavity_right()

        # Lower wall
        # left
        # stress free
        class bnd_lower_wall_left_sf(SubDomain):
            '''Lower wall left, stress free'''
            def inside(self, x, on_boundary):
                return on_boundary and x[0]>=xinfa and x[0]<=x0ns_left+10*MESH_TOL and near(x[1], 0)
                # add MESH_TOL to force all cells to belong to a subdomain
        lower_wall_left_sf = bnd_lower_wall_left_sf()
        # no slip
        class bnd_lower_wall_left_ns(SubDomain):
            '''Lower wall left, no stress'''
            def inside(self, x, on_boundary):
                return on_boundary and x[0]>=x0ns_left-10*MESH_TOL and x[0]<=0 and near(x[1], 0)
                # add MESH_TOL to force all cells to belong to a subdomain
        lower_wall_left_ns = bnd_lower_wall_left_ns()

        # right
        # no slip
        class bnd_lower_wall_right_ns(SubDomain):
            '''Lower wall right, no slip'''
            def inside(self, x, on_boundary):
                return on_boundary and between(x[0], (L, x0ns_right)) and near(x[1], 0)
        lower_wall_right_ns = bnd_lower_wall_right_ns()
        # stress free
        class bnd_lower_wall_right_sf(SubDomain):
            '''Lower wall right, stress free'''
            def inside(self, x, on_boundary):
                return on_boundary and between(x[0], (x0ns_right, xinf)) and near(x[1], 0)
        lower_wall_right_sf = bnd_lower_wall_right_sf()


        # Concatenate all boundaries
        subdmlist = [inlet, outlet, upper_wall, 
                     cavity_left, cavity_botm, cavity_right, 
                     lower_wall_left_sf, lower_wall_left_ns, 
                     lower_wall_right_ns, lower_wall_right_sf]


        #flu.export_subdomains(self.mesh, subdmlist)

        ################################## todo

        # assign boundaries as pd.DataFrame 
        boundaries_list = ['inlet', 'outlet', 'upper_wall',
                           'cavity_left', 'cavity_botm', 'cavity_right',
                           'lower_wall_left_sf', 'lower_wall_left_ns',
                           'lower_wall_right_ns', 'lower_wall_right_sf']
        boundaries_df = pd.DataFrame(index=boundaries_list, 
                                     data={'subdomain': subdmlist})
        #self.actuator_angular_size_rad = delta
        self.boundaries = boundaries_df


    def make_sensor(self):
        '''Define sensor-related quantities (surface of integration, SubDomain...)'''
        # define sensor surface
        xs0 = 1.0
        xs1 = 1.1
        MESH_TOL = DOLFIN_EPS
        sensor_subdm = CompiledSubDomain('on_boundary && near(x[1], 0, MESH_TOL) && x[0]>=xs0 && x[0]<=xs1', MESH_TOL=MESH_TOL, xs0=xs0, xs1=xs1)
        # define function to index cells
        sensor_mark = MeshFunction('size_t', self.mesh, self.mesh.topology().dim()-1)
        # define sensor as index 100 and mark
        SENSOR_IDX = 100
        sensor_subdm.mark(sensor_mark, SENSOR_IDX)
        # define surface element ds on sensor
        ds_sensor = Measure('ds', domain=self.mesh, subdomain_data=sensor_mark)
        self.ds_sensor = ds_sensor
        #self.sensor_subdomain = sensor_subdm

        ## for measuring: integrate dy(u) on sensor surface
        #yy = assemble(self.u_[0].dx(1) * ds_sensor(SENSOR_IDX))
        # or in another function: yy=assemble(up[0].dx(1) * self.ds_sensor(self.boundaries.loc['sensor'].idx))

        # Append sensor to boundaries but not to markers... might it be dangerous?
        df_sensor = pd.DataFrame(data=dict(subdomain=sensor_subdm, idx=SENSOR_IDX), index=['sensor'])
        self.boundaries = self.boundaries.append(df_sensor)


    def make_actuator(self):
        '''Define actuator: on boundary (with ternary operator cond?true:false) or volumic...'''
        # make actuator with amplitude 1
        actuator_expr = Expression(['0', 'ampl*eta*exp(-0.5*((x[0]-x10)*(x[0]-x10)+(x[1]-x20)*(x[1]-x20))/(sig*sig))'], 
            element=self.V.ufl_element(), 
            ampl=1,
            eta=1,
            sig=0.0849,
            x10=-0.1,
            x20=0.02
            )

        # compute integral on mesh >> 1/eta
        dx = Measure('dx', domain=self.mesh)
        #int2d = assemble( actuator_expr[1] * dx )
        #BtB = assemble(actuator_expr[1] * actuator_expr[1] * dx)
        BtB = norm(actuator_expr, mesh=self.mesh) # coeff for checking things
        # define actuator normalization 1/eta
        actuator_expr.eta = 1/BtB #1/BtB #1/int2d (not the same)
        #print('eta is:', actuator_expr.eta)
        #int2dpost = assemble( actuator_expr[1] * dx)
        self.actuator_expression = actuator_expr


    def make_bcs(self):
        '''Define boundary conditions'''
        # Boundary markers
        boundary_markers = MeshFunction('size_t', self.mesh, self.mesh.topology().dim() - 1)
        cell_markers = MeshFunction('size_t', self.mesh, self.mesh.topology().dim() - 1)
        # Boundary indices
        INLET_IDX = 0
        OUTLET_IDX = 1
        UPPER_WALL_IDX = 2
        CAVITY_LEFT_IDX = 3
        CAVITY_BOTM_IDX = 4
        CAVITY_RIGHT_IDX = 5
        LOWER_WALL_LEFT_SF_IDX = 6
        LOWER_WALL_LEFT_NS_IDX = 7
        LOWER_WALL_RIGHT_NS_IDX = 8
        LOWER_WALL_RIGHT_SF_IDX = 9
       
        boundaries_idx = [INLET_IDX, OUTLET_IDX, UPPER_WALL_IDX,
        CAVITY_LEFT_IDX, CAVITY_BOTM_IDX, CAVITY_RIGHT_IDX,
        LOWER_WALL_LEFT_SF_IDX, LOWER_WALL_LEFT_NS_IDX,
        LOWER_WALL_RIGHT_NS_IDX, LOWER_WALL_RIGHT_SF_IDX
        ]

        # Mark boundaries (for DirichletBC farther)
        for i, boundary_index in enumerate(boundaries_idx):
            self.boundaries.iloc[i].subdomain.mark(boundary_markers, boundary_index)
            self.boundaries.iloc[i].subdomain.mark(cell_markers, boundary_index)
        
        # for split formulation?
        # inlet : u=uinf, v=0
        bcu_inlet = DirichletBC(self.W.sub(0), Constant((self.uinf, 0)), 
            self.boundaries.loc['inlet'].subdomain)
        # upper wall : v=0 + dy(u)=0 (done in weak form) # TODO
        bcu_upper_wall = DirichletBC(self.W.sub(0).sub(1), Constant(0), 
            self.boundaries.loc['upper_wall'].subdomain)
        # lower wall left sf : v=0 + dy(u)=0 # TODO
        bcu_lower_wall_left_sf = DirichletBC(self.W.sub(0).sub(1), Constant(0), 
            self.boundaries.loc['lower_wall_left_sf'].subdomain)
        # lower wall left ns : u=0; v=0
        bcu_lower_wall_left_ns = DirichletBC(self.W.sub(0), Constant((0, 0)), 
            self.boundaries.loc['lower_wall_left_ns'].subdomain)
        # lower wall right ns : u=0; v=0
        bcu_lower_wall_right_ns = DirichletBC(self.W.sub(0), Constant((0, 0)), 
            self.boundaries.loc['lower_wall_right_ns'].subdomain)
        # lower wall right sf : v=0 + dy(u)=0 # TODO
        bcu_lower_wall_right_sf = DirichletBC(self.W.sub(0).sub(1), Constant(0), 
            self.boundaries.loc['lower_wall_right_sf'].subdomain)
        # cavity : no slip, u=0; v=0
        bcu_cavity_left = DirichletBC(self.W.sub(0), Constant((0, 0)), 
            self.boundaries.loc['cavity_left'].subdomain)
        bcu_cavity_botm = DirichletBC(self.W.sub(0), Constant((0, 0)), 
            self.boundaries.loc['cavity_botm'].subdomain)
        bcu_cavity_right = DirichletBC(self.W.sub(0), Constant((0, 0)), 
            self.boundaries.loc['cavity_right'].subdomain)
        # concatenate
        bcu = [bcu_inlet, bcu_upper_wall, 
            bcu_lower_wall_left_sf, bcu_lower_wall_left_ns,
            bcu_lower_wall_right_ns, bcu_lower_wall_right_sf, 
            bcu_cavity_left, bcu_cavity_botm, bcu_cavity_right]
        
        # pressure on outlet
        bcp_outlet = DirichletBC(self.W.sub(1), Constant(0),
            self.boundaries.loc['outlet'].subdomain)
        bcp = [bcp_outlet]

        # Measures (e.g. subscript ds(INLET_IDX))
        ds = Measure('ds', domain=self.mesh, subdomain_data=boundary_markers)
        dx = Measure('dx', domain=self.mesh, subdomain_data=cell_markers)

        # assign all
        self.bc = {'bcu': bcu, 'bcp': bcp}
        self.dx = dx
        self.ds = ds
        self.boundary_markers = boundary_markers # MeshFunction
        self.cell_markers = cell_markers # MeshFunction
        # complete boundaries pd.DataFrame
        self.boundaries['idx'] = boundaries_idx
        self.bc = {'bcu': bcu, 'bcp': bcp}

        # for perturbation formulation (nearly zeroBC)
        # inlet : u=uinf, v=0
        bcu_inlet = DirichletBC(self.W.sub(0), Constant((0, 0)), 
            self.boundaries.loc['inlet'].subdomain)
        # upper wall : dy(u)=0 # TODO
        bcu_upper_wall = DirichletBC(self.W.sub(0).sub(1), Constant(0), 
            self.boundaries.loc['upper_wall'].subdomain)
        # lower wall left sf : v=0 + dy(u)=0 # TODO
        bcu_lower_wall_left_sf = DirichletBC(self.W.sub(0).sub(1), Constant(0), 
            self.boundaries.loc['lower_wall_left_sf'].subdomain)
        # lower wall left ns : u=0; v=0
        bcu_lower_wall_left_ns = DirichletBC(self.W.sub(0), Constant((0, 0)), 
            self.boundaries.loc['lower_wall_left_ns'].subdomain)
        # lower wall right ns : u=0; v=0
        bcu_lower_wall_right_ns = DirichletBC(self.W.sub(0), Constant((0, 0)), 
            self.boundaries.loc['lower_wall_right_ns'].subdomain)
        # lower wall right sf : v=0 + dy(u)=0 # TODO
        bcu_lower_wall_right_sf = DirichletBC(self.W.sub(0).sub(1), Constant(0), 
            self.boundaries.loc['lower_wall_right_sf'].subdomain)
        # cavity : no slip, u=0; v=0
        bcu_cavity_left = DirichletBC(self.W.sub(0), Constant((0, 0)), 
            self.boundaries.loc['cavity_left'].subdomain)
        bcu_cavity_botm = DirichletBC(self.W.sub(0), Constant((0, 0)), 
            self.boundaries.loc['cavity_botm'].subdomain)
        bcu_cavity_right = DirichletBC(self.W.sub(0), Constant((0, 0)), 
            self.boundaries.loc['cavity_right'].subdomain)
        # concatenate
        bcu_p = [bcu_inlet, bcu_upper_wall, 
            bcu_lower_wall_left_sf, bcu_lower_wall_left_ns,
            bcu_lower_wall_right_ns, bcu_lower_wall_right_sf, 
            bcu_cavity_left, bcu_cavity_botm, bcu_cavity_right]
        
        self.bc_p = {'bcu': bcu_p, 'bcp': []} # log perturbation bcs


    def load_steady_state(self, assign=True):
        u0 = Function(self.V) 
        p0 = Function(self.P)
        flu.read_xdmf(self.paths['u0'], u0, "u")
        flu.read_xdmf(self.paths['p0'], p0, "p")
        
        # Assign u0, p0 >>> up0
        fa_VP2W = FunctionAssigner(self.W, [self.V, self.P])
        up0 = Function(self.W)
        fa_VP2W.assign(up0, [u0, p0])

        if assign:
            self.u0 = u0 # full field (u+upert)
            self.p0 = p0
            self.up0 = up0
            self.y_meas_steady = self.make_measurement(mixed_field=up0) 
            # steady energy
            ##self.Eb = 1/2 * norm(u0, norm_type='L2', mesh=self.mesh) # same as <up, Q@up>
            # before 16/05
            #self.Eb = self.compute_energy(field=u0, full=True, diff=False, normalize=False) 
            # modification on 16/05: diff=True otherwise we add u0+u0 I believe
            self.Eb = self.compute_energy(field=u0, full=True, diff=True, normalize=False) 
            ## energy of restricted state
            ##self.u_restrict.vector()[:] = self.u0.vector()[:] * self.IF.vector()[:]
            ##self.Eb_r = 1/2 * norm(self.u_restrict, norm_type='L2', mesh=self.mesh) # Eb restricted by self.IF
            # before 16/05
            #self.Eb_r = self.compute_energy(field=u0, full=False, diff=False, normalize=False)
            # modification on 16/05: diff=True
            self.Eb_r = self.compute_energy(field=u0, full=False, diff=True, normalize=False) 

        return u0, p0, up0


    def compute_steady_state(self, method='newton', u_ctrl=0.0, **kwargs):
        '''Compute flow steady state with given steady control'''

        # Save old control value, just in case
        actuation_ampl_old = self.actuator_expression.ampl
        # Set control value to prescribed u_ctrl
        self.actuator_expression.ampl = u_ctrl

        # If start is zero (i.e. not restart): compute
        # Note : could add a flag 'compute_steady_state' to compute or read... 
        if self.Tstart == 0: # and compute_steady_state
            # Solve
            if method=='newton':
                up0 = self.compute_steady_state_newton(**kwargs)
            else:
                up0 = self.compute_steady_state_picard(**kwargs)
        
            #u0 = flu.projectm(v=u0, V=V, bcs=bcu, mesh=mesh)
            #p0 = flu.projectm(v=p0, V=P, bcs=bcu, mesh=mesh)
            
            # assign up0, u0, p0
            # because up0 cannot be saved to file (MixedFS not supported yet)
            # fa = FunctionAssigner(receiving_fspace, assigning_fspace)
            # fa.assign(receiving_fun, assigning_fun)
            # ex: fspace([V, V], W), fun([v1 v2], w)
            # Assign up0 >>> u0, p0
            fa_W2VP = FunctionAssigner([self.V, self.P], self.W)
            u0 = Function(self.V)
            p0 = Function(self.P)
            fa_W2VP.assign([u0, p0], up0)
           
            # Save steady state
            if self.save_every:
                flu.write_xdmf(self.paths['u0'], u0, "u", time_step=0.0, append=False, write_mesh=True)
                flu.write_xdmf(self.paths['p0'], p0, "p", time_step=0.0, append=False, write_mesh=True)
            if self.verbose:
                print('Stored base flow in: ', self.savedir0)

            self.y_meas_steady = self.make_measurement(mixed_field=up0)

        # If start is not zero: ready steady state (should exist - should check though...)
        else:
            u0, p0, up0 = self.load_steady_state(assign=True)
            
        # Compute lift & drag
        cl, cd = self.compute_force_coefficients(u0, p0)
        if self.verbose:
            print('Lift coefficient is: cl =', cl)
            print('Drag coefficient is: cd =', cd)
        
        # Set old actuator amplitude
        # is it necessary???
        self.actuator_expression.ampl = actuation_ampl_old

        # assign steady state
        self.up0 = up0
        self.u0 = u0
        self.p0 = p0
        # assign steady cl, cd
        self.cl0 = cl
        self.cd0 = cd
        # assign steady energy
        self.Eb = 1/2 * norm(u0, norm_type='L2', mesh=self.mesh)**2 # same as <up, Q@up>
        # Eb restricted
        self.u_restrict.vector()[:] = self.u0.vector()[:] * self.IF.vector()[:]
        self.Eb_r = 1/2 * norm(self.u_restrict, norm_type='L2', mesh=self.mesh)**2 # Eb restricted by self.IF


    def make_form_mixed_steady(self, initial_guess=None):
        '''Make nonlinear forms for steady state computation, in mixed element space.
        Can be used to assign self.F0 and compute state spaces matrices.'''
        v, q = TestFunctions(self.W)
        if initial_guess is None:
            up_ = Function(self.W)
        else: 
            up_ = initial_guess
        u_, p_ = split(up_) # not deep copy, we need the link
        iRe = Constant(1/self.Re)
        f = self.actuator_expression
        # Problem
        F0 = dot(dot(u_, nabla_grad(u_)), v)*dx \
            + iRe*inner(nabla_grad(u_), nabla_grad(v))*dx \
            - p_*div(v)*dx \
            - q*div(u_)*dx \
            - dot(f, v)*dx
        self.F0 = F0
        self.up_ = up_
        self.u_ = u_
        self.p_ = p_


    #def make_form_mixed_steady_perturbation(self):
    #    '''Make form for steady state (linearized perturbation formulation).
    #    Note: unused?'''
    #    v, q = TestFunctions(self.W)
    #    u, p = TrialFunctions(self.W)
    #    iRe = Constant(1/self.Re)
    #    # Problem
    #    u0 = self.u0
    #    F1 = dot( dot(u0, nabla_grad(u)), v)*dx \
    #       + dot( dot(u, nabla_grad(u0)), v)*dx \
    #       + iRe*inner(nabla_grad(u), nabla_grad(v))*dx \
    #       - p*div(v)*dx \
    #       - div(u)*q*dx \
    #       - dot(f, v)*dx
    #    self.F0p = F1


    def compute_steady_state_newton(self, max_iter=25, initial_guess=None):
        '''Compute steady state with built-in nonlinear solver (Newton method)
        initial_guess is a (u,p)_0'''
        self.make_form_mixed_steady(initial_guess=initial_guess)
        #if initial_guess is None:
        #    print('- Newton solver without initial guess')
        up_ = self.up_
        u_, p_ = self.u_, self.p_
        #else:
        #    print('- Newton solver with initial guess')
        #    up_ = initial_guess
        #    #up_ = Function(self.W)
        #    u_, p_ = up_.split(deepcopy=True)
        # Solver param
        nl_solver_param = {"newton_solver":
                                {
                                "linear_solver" : "mumps", 
                                "preconditioner" : "default",
                                "maximum_iterations" : max_iter,
                                "report": bool(self.verbose)
                                }
                          }
        solve(self.F0==0, up_, self.bc['bcu'], solver_parameters=nl_solver_param)
        # Return
        return up_


    def compute_steady_state_picard(self, max_iter=10, tol=1e-14):
        '''Compute steady state with fixed-point iteration
        Should have a larger convergence radius than Newton method
        if initialization is bad in Newton method (and it is)
        TODO: residual not 0 if u_ctrl not 0 (see bc probably)'''
        iRe = Constant(1/self.Re)
        
        # for residual computation
        #bcu_inlet0 = DirichletBC(self.W.sub(0), Constant((0, 0)), self.boundaries.loc['inlet'].subdomain)
        #bcu0 = self.bc['bcu'] + [bcu_inlet0]
        bcu0 = self.bc_p['bcu']
        
        # define forms
        up0 = Function(self.W)
        up1 = Function(self.W)
    
        u, p = TrialFunctions(self.W)
        v, q = TestFunctions(self.W)
        
        class initial_condition(UserExpression):
            def eval(self, value, x):
                # outside cavity: v=(1,0)
                value[0] = 1.0
                value[1] = 0.0
                value[2] = 0.0
                # inside cavity: v=(0,0)
                if x[1]<=0:
                    value[0] = 0.0
            def value_shape(self):
                return (3,)
        
        up0.interpolate(initial_condition())
        u0 = as_vector((up0[0], up0[1]))
        u1 = as_vector((up1[0], up1[1]))
    
        f = self.actuator_expression
        ap = dot( dot(u0, nabla_grad(u)), v) * dx \
            + iRe*inner(nabla_grad(u), nabla_grad(v))*dx \
            - p*div(v)*dx \
            - q*div(u)*dx
        Lp = dot(f, v)* dx + Constant(0)*inner(u0, v)*dx + Constant(0)*q*dx # rhs = actuation
        bp = assemble(Lp)
   
        solverp = LUSolver('mumps')
        ndof = self.W.dim()

        for i in range(max_iter):
            Ap = assemble(ap)
            [bc.apply(Ap, bp) for bc in self.bc['bcu']]

            solverp.solve(Ap, up1.vector(), bp)
    
            up0.assign(up1)
            u, p = up1.split()
    
            # show_max(u, 'u')
            res = assemble(action(ap, up1))
            [bc.apply(res) for bc in bcu0]
            res_norm = norm(res)/sqrt(ndof)
            if self.verbose:
                print('Picard iteration: {0}/{1}, residual: {2}'.format(i+1, max_iter, res_norm))
            if res_norm < tol:
                if self.verbose:
                    print('Residual norm lower than tolerance {0}'.format(tol))
                break
        
        return up1


    def stress_tensor(self, u, p):
        '''Compute stress tensor (for lift & drag)'''
        return 2.0*self.nu*(sym(grad(u))) - p*Identity(p.geometric_dimension())
    

    def compute_force_coefficients(self, u, p, enable=False):
        '''Compute lift & drag coefficients'''
        if enable:
            sigma = self.stress_tensor(u, p)
            Fo = -dot(sigma, self.n)

            # integration surfaces names
            surfaces_names = ['cavity_left', 'cavity_botm', 'cavity_right']
            # integration surfaces indices
            surfaces_idx = [self.boundaries.loc[nm].idx for nm in surfaces_names]
            
            # define drag & lift expressions
            # sum symbolic forces
            drag_sym = sum([Fo[0]*self.ds(int(sfi)) for sfi in surfaces_idx])
            lift_sym = sum([Fo[1]*self.ds(int(sfi)) for sfi in surfaces_idx])
            # integrate sum of symbolic forces
            drag = assemble(drag_sym)
            lift = assemble(lift_sym)

            ## integrate forces on each surface
            #drag_sum = [assemble(Fo[0]*self.ds(sfi)) for sfi in surfaces_idx]
            #lift_sum = [assemble(Fo[1]*self.ds(sfi)) for sfi in surfaces_idx]
            ##print('drag_sum, lift_sum:', drag_sum, lift_sum)
            ### sum integrands
            #drag = sum(drag_sum)
            #lift = sum(lift_sum)

            #print('----------- diagnostic from compute_force_coefficients ------------')
            #print('(display order: assemble(sum(symbolic)), sum(assemble(symbolic)), list')
            #print('lift is: ', lift, sum(lift_sum), lift_sum)
            #print('drag is: ', drag, sum(drag_sum), drag_sum)

            
            # start anew with dedicated subregion
            #cyl_idx = 1
            #cyl_mkr = MeshFunction('size_t', self.mesh, self.mesh.topology().dim() - 1)
            #self.cylinder_intg.mark(cyl_mkr, cyl_idx)
            #dsc = Measure('ds', domain=self.mesh, subdomain_data=cyl_mkr)
            #drag = assemble(Fo[0]*dsc(cyl_idx))
            #lift = assemble(Fo[1]*dsc(cyl_idx))

            # define force coefficients by normalizing
            cd = drag/(1/2*self.uinf**2*self.d)
            cl = lift/(1/2*self.uinf**2*self.d)
        else:
            cl, cd = 0, 1
        return cl, cd

    
    def compute_vorticity(self, u=None):
        '''Compute vorticity field of given velocity field u'''
        if u is None:
            u = self.u_
        # should probably project on space of order n-1 --> self.P
        vorticity = flu.projectm(curl(u), V=self.V.sub(0).collapse())
        return vorticity

    
    def compute_divergence(self, u=None):
        '''Compute divergence field of given velocity field u'''
        if u is None:
            u = self.u_
        divergence = flu.projectm(div(u), self.P)
        return divergence


    # Initial perturbations ######################################################
    class localized_perturbation_u(UserExpression):
        '''Perturbation localized in disk
        Use: u = interpolate(localized_perturbation_u(), self.V)
        or something like that'''
        def eval(self, value, x):
            if((x[0]--2.5)**2 + (x[1]-0.1)**2 <= 1):
                value[0] = 0.05
                value[1] = 0.05
            else:
                value[0] = 0
                value[1] = 0
        def value_shape(self):
            return(2,)

 
    class random_perturbation_u(UserExpression):
        '''Perturbation in the whole volume, random
        Use: see localized_perturbation_u'''
        def eval(self, value, x):
            value[0] = 0.1*np.random.randn()
            value[1] = 0
        def value_shape(self):
            return(2,)


    def get_div0_u(self):
        '''Create velocity field with zero divergence'''
        V = self.V
        P = self.P
       
        # Define courant function
        x0 = 2
        y0 = 0
        nsig = 5
        sigm = 0.5
        xm, ym = sp.symbols('x[0], x[1]')
        rr = (xm-x0)**2 + (ym-y0)**2
        fpsi = 0.25 * sp.exp(-1/2 * rr / sigm**2)
        # Piecewise does not work too well
        #fpsi = sp.Piecewise(   (sp.exp(-1/2 * rr / sigm**2), 
        #rr <= nsig**2 * sigm**2), (0, True) )
        dfx = fpsi.diff(xm, 1)
        dfy = fpsi.diff(ym, 1)
        
        # Take derivatives
        psi = Expression(sp.ccode(fpsi), element=V.ufl_element())
        dfx_expr = Expression(sp.ccode(dfx), element=P.ufl_element())
        dfy_expr = Expression(sp.ccode(dfy), element=P.ufl_element())
       
        # Check
        #psiproj = flu.projectm(psi, P)
        #flu.write_xdmf('psi.xdmf', psiproj, 'psi')
        #flu.write_xdmf('psi_dx.xdmf', flu.projectm(dfx_expr, P), 'psidx')
        #flu.write_xdmf('psi_dy.xdmf', flu.projectm(dfy_expr, P), 'psidy')
        
        # Make velocity field
        upsi = flu.projectm(as_vector([dfy_expr, -dfx_expr]), self.V)
        return upsi
        

    def get_div0_u_random(self, sigma=0.1, seed=0):
        '''Create random velocity field with zero divergence'''
        # CG2 scalar
        P2 = self.V.sub(0).collapse()

        # Make scalar potential field in CG2 (scalar)
        a0 = Function(P2)
        np.random.seed(seed)
        a0.vector()[:] += sigma*np.random.randn(a0.vector()[:].shape[0])

        # Take curl, then by definition div(u0)=div(curl(a0))=0
        Ve = VectorElement('CG', self.mesh.ufl_cell(), 1)
        V1 = FunctionSpace(self.mesh, Ve)

        u0 = flu.projectm(curl(a0), V1)
        
        ##divu0 = flu.projectm(div(u0), self.P)

        ## 1 step div(curl())
        #divcurl1 = flu.projectm(div(curl(a0)), self.V.sub(0).collapse())
        #
        ## 2 steps (div(curl())
        #Pe = FiniteElement('DG', self.mesh.ufl_cell(), 0)
        #P0 = FunctionSpace(self.mesh, Pe)
        #divcurl2 = flu.projectm( div(u0), P0 )

        ## Check
        ## this obviously does not work
        #flu.show_max(divcurl1, 'div(rand_u0) (1 project)')
        #flu.show_max(divcurl2, 'div(rand_u0) (2 project)')

        return u0


    def set_initial_state(self, x0=None):
        '''Define initial state and assign to self.initial_state
        x0: Function(self.W)
        Function needs to be called before self.init_time_stepping()'''
        self.initial_state = x0


    def init_time_stepping(self):
        '''Create varitional functions/forms & flush files & define u(0), p(0)'''
        # Trial and test functions ####################################################
        W = self.W
       
        # Define expressions used in variational forms
        f = Constant((0, 0)) # shape of actuation
        iRe = Constant(1/self.Re)
        II = Identity(2)
        k = Constant(self.dt)
        ##############################################################################
    
        t = self.Tstart
        self.t = t
        self.iter = 0 

        # Prepare IPCS
        # function spaces
        V = self.V
        P = self.P
        # trial and test functions
        u = TrialFunction(V)
        v = TestFunction(V)
        p = TrialFunction(P)
        q = TestFunction(P)
        # solutions 
        u_ = Function(self.V)
        p_ = Function(self.P)

        # if not restart
        if self.Tstart == 0: 
            # first order temporal integration
            self.order = 1

            # Set initial state up in W
            initial_up = Function(self.W)

            # No initial state given -> base flow
            if self.initial_state is None:
                initial_up = Function(self.W)
                if not self.perturbations:
                    initial_up.vector()[:] += self.up0.vector()[:]
            else:
                initial_up = self.initial_state

            # Impulse or state perturbation @ div0
            # Impulse if self.init_pert is inf
            if np.isinf(self.init_pert):
                # not sure this would work in parallel
                initial_up.vector()[:] += self.get_B().reshape((-1,))
            else:
                udiv0 = self.get_div0_u()
                fa = FunctionAssigner(self.W, [self.V, self.P])
                pert0 = Function(self.W)
                fa.assign(pert0, [udiv0, self.p0])
                initial_up.vector()[:] += self.init_pert*pert0.vector()[:]

            initial_up.vector().apply('insert') 
            up1 = initial_up 
             
            # Split up to u, p
            fa = FunctionAssigner([self.V, self.P], self.W) 
            u1 = Function(self.V)
            p1 = Function(self.P)
            fa.assign([u1, p1], up1)

            # this is the initial state
            if self.perturbations:
                bcs = self.bc_p['bcu'] # bcs for perturbation formulation
            else:
                bcs = self.bc['bcu'] # bcs for classic formulation
            u_n = flu.projectm(v=u1, V=self.V, bcs=bcs)
            u_nn = u_n.copy(deepcopy=True)
            p_n = flu.projectm(self.p0, self.P)
            
            u_ = u_n.copy(deepcopy=True)
            p_ = p_n.copy(deepcopy=True)

            # Flush files and save steady state as time_step 0
            if self.save_every:
                if not self.perturbations:
                    flu.write_xdmf(self.paths['u_restart'], u_n, "u",
                               time_step=0., append=False, write_mesh=True)
                    flu.write_xdmf(self.paths['uprev_restart'], u_nn, "u_n",
                               time_step=0., append=False, write_mesh=True)
                    flu.write_xdmf(self.paths['p_restart'], p_n, "p",
                               time_step=0., append=False, write_mesh=True)
                else:
                    u_n_save = Function(self.V)
                    p_n_save = Function(self.P)
                    u_n_save.vector()[:] = u_n.vector()[:] + self.u0.vector()[:]
                    p_n_save.vector()[:] = p_n.vector()[:] + self.p0.vector()[:]
                    flu.write_xdmf(self.paths['u_restart'], u_n_save, "u",
                               time_step=0., append=False, write_mesh=True)
                    flu.write_xdmf(self.paths['uprev_restart'], u_n_save, "u_n",
                               time_step=0., append=False, write_mesh=True)
                    flu.write_xdmf(self.paths['p_restart'], p_n_save, "p",
                               time_step=0., append=False, write_mesh=True)
                    
        else:
            # find index to load saved data
            idxstart = -1 if(self.Tstart==-1) \
                          else int(np.floor(self.Tstart/self.dt/self.save_every_old)) 
            # second order temporal integration
            self.order = 2
            # assign previous solution
            # here: subtract base flow if perturbation
            # if perturbations : read u_n, subtract u0, save
            # if not: read u_n, write u_n
            u_n = Function(self.V)
            u_nn = Function(self.V)
            p_n = Function(self.P)
            flu.read_xdmf(self.paths['u'], u_n, "u", counter=idxstart)
            flu.read_xdmf(self.paths['uprev'], u_nn, "u_n", counter=idxstart)
            flu.read_xdmf(self.paths['p'], p_n, "p", counter=idxstart)
        
            flu.read_xdmf(self.paths['u'], u_, "u", counter=idxstart)
            flu.read_xdmf(self.paths['p'], p_, "p", counter=idxstart)
            # write in new file as first time step
            # important to do this before subtracting base flow (if perturbations)
            if self.save_every:
                flu.write_xdmf(self.paths['u_restart'], u_n, "u", 
                    time_step=self.Tstart, append=False, write_mesh=True)
                flu.write_xdmf(self.paths['uprev_restart'], u_nn, "u_n", 
                    time_step=self.Tstart, append=False, write_mesh=True)
                flu.write_xdmf(self.paths['p_restart'], p_n, "p", 
                    time_step=self.Tstart, append=False, write_mesh=True)
            # if perturbations, remove base flow from loaded file
            # because one prefers to write complete flow (not just perturbations)
            if self.perturbations:
                u_n.vector()[:] = u_n.vector()[:] - self.u0.vector()[:]
                u_nn.vector()[:] = u_nn.vector()[:] - self.u0.vector()[:]
                p_n.vector()[:] = p_n.vector()[:] - self.p0.vector()[:]
                u_.vector()[:] = u_.vector()[:] - self.u0.vector()[:]
                p_.vector()[:] = p_.vector()[:] - self.p0.vector()[:]
            

        if self.verbose and flu.MpiUtils.get_rank()==0:
            print('Starting or restarting from time: ', self.Tstart,
                  ' with temporal scheme order: ', self.order)

        # Assemble IPCS forms (1st and 2nd orders) 
        # + Modify BCs (redirect function spaces W.sub(0)>>>V...)
        # inlet : u=uinf, v=0
        bcu_inlet_g = DirichletBC(V, Constant((self.uinf, 0)), 
            self.boundaries.loc['inlet'].subdomain)
        # upper wall : v=0 + dy(u)=0 (done in weak form)
        bcu_upper_wall_g = DirichletBC(V.sub(1), Constant(0), 
            self.boundaries.loc['upper_wall'].subdomain)
        # lower wall left sf : v=0 + dy(u)=0
        bcu_lower_wall_left_sf_g = DirichletBC(V.sub(1), Constant(0), 
            self.boundaries.loc['lower_wall_left_sf'].subdomain)
        # lower wall left ns : u=0; v=0
        bcu_lower_wall_left_ns_g = DirichletBC(V, Constant((0, 0)), 
            self.boundaries.loc['lower_wall_left_ns'].subdomain)
        # lower wall right ns : u=0; v=0
        bcu_lower_wall_right_ns_g = DirichletBC(V, Constant((0, 0)), 
            self.boundaries.loc['lower_wall_right_ns'].subdomain)
        # lower wall right sf : v=0 + dy(u)=0
        bcu_lower_wall_right_sf_g = DirichletBC(V.sub(1), Constant(0), 
            self.boundaries.loc['lower_wall_right_sf'].subdomain)
        # cavity : no slip, u=0; v=0
        bcu_cavity_left_g = DirichletBC(V, Constant((0, 0)), 
            self.boundaries.loc['cavity_left'].subdomain)
        bcu_cavity_botm_g = DirichletBC(V, Constant((0, 0)), 
            self.boundaries.loc['cavity_botm'].subdomain)
        bcu_cavity_right_g = DirichletBC(V, Constant((0, 0)), 
            self.boundaries.loc['cavity_right'].subdomain)
        bcug = [bcu_inlet_g, bcu_upper_wall_g,
            bcu_lower_wall_left_sf_g, bcu_lower_wall_left_ns_g,
            bcu_lower_wall_right_ns_g, bcu_lower_wall_right_sf_g,
            bcu_cavity_left_g, bcu_cavity_botm_g, bcu_cavity_right_g]

        bcp_outlet_g = DirichletBC(P, Constant(0), 
            self.boundaries.loc['outlet'].subdomain)
        bcpg = [bcp_outlet_g]

        self.bc_ipcs = {'bcu': bcug, 'bcp': bcpg}


        ## order 1
        f = self.actuator_expression
        # velocity prediction
        F1g = dot((u - u_n) / k, v)*dx \
           + dot( dot(u_n, nabla_grad(u_n)), v)*dx \
           + inner(iRe*nabla_grad(u), nabla_grad(v))*dx \
           - inner(p_n*II, nabla_grad(v))*dx \
           - dot(f, v)*dx
        a1g = lhs(F1g)
        L1g = rhs(F1g)
        # pressure correction
        a2g = dot(nabla_grad(p), nabla_grad(q))*dx
        L2g = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx
        # velocity correction
        a3g = dot(u, v)*dx
        L3g = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx
        # system assembler
        # Note: useful for 2 reasons
        #  1) do not pass matrices all the way
        #  2) only (?) way to preallocate & assemble in preallocated RHS
        sysAssmb1 = [SystemAssembler(a1g, L1g, self.bc_ipcs['bcu']),
                     SystemAssembler(a2g, L2g, self.bc_ipcs['bcp']),
                     SystemAssembler(a3g, L3g, self.bc_ipcs['bcu'])]
        AA1 = [Matrix() for i in range(3)] # equiv to PETScMatrix()
        # Make assemblers & solvers
        solvers1 = self.make_solvers()
        for assmblr, A, solver in zip(sysAssmb1, AA1, solvers1):
            assmblr.assemble(A)
            solver.set_operator(A)

        ## order 2
        # velocity prediction
        F1g2 = dot((3*u - 4*u_n + u_nn) / (2*k), v)*dx \
           + 2*dot( dot(u_n, nabla_grad(u_n)), v)*dx \
           - dot( dot(u_nn, nabla_grad(u_nn)), v)*dx \
           + inner(iRe*nabla_grad(u), nabla_grad(v))*dx \
           - inner(p_n*II, nabla_grad(v))*dx \
           - dot(f, v)*dx
        a1g2 = lhs(F1g2)
        L1g2 = rhs(F1g2)
        # pressure correction
        a2g2 = dot(nabla_grad(p), nabla_grad(q))*dx
        L2g2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - 3/(2*k)*div(u_)*q*dx
        # velocity correction
        a3g2 = dot(u, v)*dx
        L3g2 = dot(u_, v)*dx - (2*k)/3*dot(nabla_grad(p_ - p_n), v)*dx
        # system assembler 2
        # "this is very pythonic"
        #sysAssmb2 = [SystemAssembler(a, L, self.bc_ipcs[bcn]) 
        #    for a, L, bcn in zip([a1g2, a2g2, a3g2], 
        #                         [L1g2, L2g2, L3g2], 
        #                         ['bcu', 'bcp', 'bcu'])]
        # "this is not very pythonic" but nvm
        sysAssmb2 = [SystemAssembler(a1g2, L1g2, self.bc_ipcs['bcu']),
                     SystemAssembler(a2g2, L2g2, self.bc_ipcs['bcp']),
                     SystemAssembler(a3g2, L3g2, self.bc_ipcs['bcu'])]
        AA2 = [Matrix() for i in range(3)] # equiv to PETScMatrix()
        # Make assemblers & solvers
        solvers2 = self.make_solvers() 
        for assmblr, A, solver in zip(sysAssmb2, AA2, solvers2):
            assmblr.assemble(A)
            solver.set_operator(A)

        # pre-allocate rhs
        #b1 = PETScVector(MPI.comm_world, V.dim())
        #b2 = PETScVector(MPI.comm_world, P.dim())
        #b3 = PETScVector(MPI.comm_world, V.dim())
        # PETScVector does not go parallel well with FEniCS solvers >>> Vector()
        self.bs = [Vector() for i in range(3)]
        # following only has an impact on 1st iteration and that's all
        #self.bs = [Vector(MPI.comm_world, u_.vector().local_size()),
        #           Vector(MPI.comm_world, p_.vector().local_size()), 
        #           Vector(MPI.comm_world, u_.vector().local_size())]
        self.assemblers = {1: sysAssmb1, 2: sysAssmb2}
        self.solvers =    {1: solvers1,  2: solvers2}
        self.u_ = u_
        self.p_ = p_
        self.u_n = u_n
        self.u_nn = u_nn
        self.p_n = p_n

        # Compute things on x(0) 
        fa = FunctionAssigner(self.W, [self.V, self.P])
        up_n = Function(self.W)
        fa.assign(up_n, [u_n, p_n])
        self.y_meas0 = self.make_measurement(mixed_field=up_n) 
        self.y_meas = self.y_meas0 
        # not valid in perturbations formulation
        cl1, cd1 = self.compute_force_coefficients(u_n, p_n)

        # Make time series pd.DataFrame
        y_meas_str = ['y_meas_'+str(i+1) for i in range(self.sensor_nr)]
        colnames = ['time', 'u_ctrl'] + y_meas_str + ['u_norm', 'p_norm', 'dE', 'cl', 'cd', 'runtime']
        empty_data = np.zeros((self.num_steps+1, len(colnames)))
        ts1d = pd.DataFrame(columns=colnames, data=empty_data)
        u_ctrl = Constant(0)
        ts1d.loc[0, 'time'] = self.Tstart
        #ts1d.loc[0, 'y_meas'] = self.y_meas0 
        # replace line above for several measurements
        self.assign_measurement_to_dataframe(df=ts1d, y_meas=self.y_meas0, index=0) 
        ts1d.loc[0, 'cl'], ts1d.loc[0, 'cd'] = cl1, cd1
        ts1d.loc[0, 'u_norm'] = 0 # norm(u_n, norm_type='L2', mesh=self.mesh)
        ts1d.loc[0, 'p_norm'] = 0 # norm(p_n, norm_type='L2', mesh=self.mesh)
        if self.compute_norms:
            #import pdb
            #pdb.set_trace()
            dEb = self.compute_energy(full=True, diff=True, normalize=True)
            #self.u_restrict.vector()[:] = self.u_n.vector()[:] - self.u0.vector()[:]
            #dEb = norm(self.u_restrict, norm_type='L2', mesh=self.mesh) / self.Eb
        else:
            dEb = 0
        ts1d.loc[0, 'dE'] = dEb 
        self.timeseries = ts1d


    def step(self, u_ctrl):
        '''Update state x(t)->x(t+dt) with control u_ctrl(t) and output y(t)=h(x(t), u_ctrl(t))
        This function usually goes in an external time loop
        It also logs some data for post-processing'''
        # measure runtime
        t0i = time.time()
        
        # assign control u(t)
        self.actuator_expression.ampl = u_ctrl
        
        # solve once
        u_ = self.u_
        p_ = self.p_
        u_n = self.u_n
        u_nn = self.u_nn
        p_n = self.p_n

        assemblers = self.assemblers[self.order]
        solvers = self.solvers[self.order]

        # ugly try/catch for now
        if not self.throw_error: # used for optimization -> return error code
            try:
                # Solve 3 steps
                for assmbl, solver, bb, upsol in zip(assemblers, solvers, self.bs, (u_, p_, u_)):
                    assmbl.assemble(bb) # assemble rhs
                    ret = solver.solve(upsol.vector(), bb) # solve Ax=b
            except RuntimeError:
                # Usually Krylov solver exploding return a RuntimeError
                # See: error_on_nonconvergence (but need to catch error somehow)
                print('Error solving system --- Exiting step()...')
                return -1 # -1 is error code
        else: # used for debugging -> show error message
           # Solve 3 steps
           #i = 0
           for assmbl, solver, bb, upsol in zip(assemblers, solvers, self.bs, (u_, p_, u_)):
               t0 = time.time()
               assmbl.assemble(bb) # assemble rhs
               tassemble = time.time() - t0
               t0 = time.time()
               ret = solver.solve(upsol.vector(), bb) # solve Ax=b
               tsolve = time.time() - t0

               #i += 1
               #self.alltimes[i] += tsolve
               #i += 1
               #self.alltimes.loc['assemble'][str(i)] += tassemble
               #self.alltimes.loc['solve'][str(i)] += tsolve
               # here: instead of assembling b everytime (especially step 1):
               # do: assemble(part of b) + u @ Dx @ b 
               # so: need to define b_partial that does not take into account convection
               # where the last part is <u, grad u> = convection term
                
        # at this point:
        # x(t+dt) = u_
        # x(t) = u_n
        # x(t-dt) = u_nn
        # do not move these lines after the assignment

        #if self.compute_norms:
        #    xdot = Function(self.V)
        #    xdot.vector()[:] = (3*u_.vector()[:] - 4*u_n.vector()[:] + u_nn.vector()[:])/(2*self.dt)
        #    normxdot = norm(xdot, norm_type='L2', mesh=self.mesh)
        #    if not hasattr(self, 'normxdot'):
        #        self.normxdot = []
        #    self.normxdot.append(normxdot)

        # Assign next
        u_nn.assign(u_n) # x(t-dt)
        u_n.assign(u_)   # x(t)
        p_n.assign(p_)   

        # Update time
        self.iter += 1
        self.t = self.Tstart + (self.iter)*self.dt # better accuracy than t+=dt
        
        # Assign to self
        self.u_ = u_
        self.p_ = p_
        self.u_n = u_n
        self.u_nn = u_nn
        self.p_n = p_n
        
        # Goto order 2 next time
        self.order = 2
        
        # Measurement of y(t+dt)
        self.y_meas = self.make_measurement()

        # Log timeseries
        cl, cd = 1, 2 #self.compute_force_coefficients(u_, p_)
        if self.compute_norms:
            dE = self.compute_energy(full=True, diff=True, normalize=True)
            #flu.MpiUtils.check_process_rank()
            #print('i am calculating norm: ', dE)
            #print('base flow energy is: ', self.Eb)
        else:
            dE = -1
        
        # Runtime measurement & display
        # runtime does not take into account anything that is below tfi
        # so do not cheat by computing things after (like norms or cl, cd)
        tfi = time.time()
        if self.verbose:
            self.print_progress(runtime=tfi-t0i)
        self.log_timeseries(u_ctrl=u_ctrl,
                            y_meas=self.y_meas,
                            norm_u = 0, # norm(u_, norm_type='L2', mesh=self.mesh),
                            norm_p = 0, # norm(p_, norm_type='L2', mesh=self.mesh),
                            dE = dE,
                            cl=cl, cd=cd,
                            t=self.t, runtime=tfi-t0i)
        
        # Save
        if self.save_every and not self.iter%self.save_every:
            if self.verbose:
                print("saving to files %s" %(self.savedir0))
            flu.write_xdmf(self.paths['u_restart'], u_n, "u", 
                       time_step=self.t, append=True, write_mesh=False)
            flu.write_xdmf(self.paths['uprev_restart'], u_nn, "u_n", 
                       time_step=self.t, append=True, write_mesh=False)
            flu.write_xdmf(self.paths['p_restart'], p_n, "p", 
                       time_step=self.t, append=True, write_mesh=False) 
            self.write_timeseries()

        return ret

    
    def log_timeseries(self, u_ctrl, y_meas, norm_u, norm_p, dE, cl, cd, t, runtime):
        '''Fill timeseries table with data'''
        self.timeseries.loc[self.iter-1, 'u_ctrl'] = u_ctrl # careful here: log the command that was applied at time t (iter-1) to get time t+dt (iter)
        #self.timeseries.loc[self.iter, 'y_meas'] = y_meas
        # replace above line for several measurements
        self.assign_measurement_to_dataframe(df=self.timeseries, y_meas=y_meas, index=self.iter)
        self.timeseries.loc[self.iter, 'u_norm'] = norm_u
        self.timeseries.loc[self.iter, 'p_norm'] = norm_p
        self.timeseries.loc[self.iter, 'dE'] = dE
        self.timeseries.loc[self.iter, 'cl'], self.timeseries.loc[self.iter, 'cd'] = cl, cd
        self.timeseries.loc[self.iter, 'time'] = t
        self.timeseries.loc[self.iter, 'runtime'] = runtime


    def print_progress(self, runtime):
        '''Single line to print progress'''
        print('--- iter: %5d/%5d --- time: %3.3f/%3.2f --- elapsed %5.5f ---' 
              %(self.iter, self.num_steps, self.t, self.Tf+self.Tstart,  runtime))


    def step_perturbation(self, u_ctrl=0.0, shift=0.0, NL=True):
        '''Simulate system with perturbation formulation,
        possibly an actuation value, and a shift
        initial_up may be set as self.get_B() to compute impulse response'''
        iRe = Constant(1/self.Re)
        k = Constant(self.dt)
    
        v, q = TestFunctions(self.W)
        up = TrialFunction(self.W)
        u, p = split(up)
        up_ = Function(self.W)
        u_, p_ = split(up_)
        u0 = self.u0
        
        if NL: # nonlinear
            b0_1 = 1 # order 1 
            b0_2, b1_2 = 2, -1 # order 2
        else:  # linear, remove (u'.nabla)(u')
            b0_1 = b0_2 = b1_2 = 0

        # init with self.attr (from init_time_stepping)
        u_nn = self.u_nn
        u_n = self.u_n
        p_n = self.p_n

        # This step is handled with init_time_stepping for IPCS formulation
        if not hasattr(self, 'assemblers_p'):# make forms
            if self.verbose:
                print('Perturbations forms DO NOT exist: create...')
            #if self.perturb_initial_state:
            #    u_n = self.get_div0_u()
            #    u_nn = u_n.copy(deepcopy=True)
            #else:
            #    if initial_up is None:
            #        u_n.vector().zero()
            #        u_nn.vector().zero()
            #    # else keep as defined by initial_up
            f = self.actuator_expression
            shift = Constant(shift)
            # 1st order integration
            F1 = dot((u - u_n) / k, v)*dx \
               + dot( dot(u0, nabla_grad(u)), v)*dx \
               + dot( dot(u, nabla_grad(u0)), v)*dx \
               + iRe*inner(nabla_grad(u), nabla_grad(v))*dx \
               + Constant(b0_1)*dot( dot(u_n, nabla_grad(u_n)), v)*dx \
               - p*div(v)*dx \
               - div(u)*q*dx \
               - shift*dot(u,v)*dx \
               - dot(f, v)*dx

            # 2nd order integration
            F2 = dot((3*u - 4*u_n + u_nn) / (2*k), v)*dx \
               + dot( dot(u0, nabla_grad(u)), v)*dx \
               + dot( dot(u, nabla_grad(u0)), v)*dx \
               + iRe*inner(nabla_grad(u), nabla_grad(v))*dx \
               + Constant(b0_2)*dot( dot(u_n, nabla_grad(u_n)), v)*dx \
               + Constant(b1_2)*dot( dot(u_nn, nabla_grad(u_nn)), v)*dx \
               - p*div(v)*dx \
               - div(u)*q*dx \
               - shift*dot(u,v)*dx \
               - dot(f, v)*dx

            # Extract
            a1 = lhs(F1)
            L1 = rhs(F1)
            a2 = lhs(F2)
            L2 = rhs(F2)
        
            sysAssmb1 = SystemAssembler(a1, L1, self.bc_p['bcu'])
            sysAssmb2 = SystemAssembler(a2, L2, self.bc_p['bcu'])
            Ap1, Ap2 = Matrix(), Matrix()

            S = [LUSolver('mumps') for i in range(2)]
            for assemblr, solver, A in zip([sysAssmb1, sysAssmb2], S, [Ap1, Ap2]):
                assemblr.assemble(A)
                solver.set_operator(A)

            self.bs_p = Vector() # create rhs
            self.assemblers_p = {1: sysAssmb1, 2: sysAssmb2}
            self.solvers_p = {1: S[0], 2: S[1]}

            # save perturbation and full solution
            self.u_full = Function(self.V)
            self.u_n_full = Function(self.V)
            self.p_full = Function(self.P)
        
        # time
        t0i = time.time()
        
        # control
        self.actuator_expression.ampl = u_ctrl
        
        # Assign system of eqs
        assembler = self.assemblers_p[self.order]
        solver = self.solvers_p[self.order]

        if not self.throw_error: # used for optimization -> return error code
            try:
                assembler.assemble(self.bs_p) # assemble rhs
                solver.solve(up_.vector(), self.bs_p) # solve Ax=b
                u_, p_ = up_.split(deepcopy=True)
                # test if nan here?
                #pdb.set_trace()
            except RuntimeError:
                # Usually Krylov solver exploding return a RuntimeError
                # See: error_on_nonconvergence (but need to catch error somehow)
                print('Error solving system --- Exiting step()...')
                return -1 # -1 is error code
        else: # used for debugging -> show error message
            #pdb.set_trace()
            assembler.assemble(self.bs_p) # assemble rhs
            solver.solve(up_.vector(), self.bs_p) # solve Ax=b
            u_, p_ = up_.split(deepcopy=True)

        # Assign new
        u_nn.assign(u_n)
        u_n.assign(u_)
        p_n.assign(p_)

        # Update time
        self.iter += 1
        self.t = self.Tstart + (self.iter)*self.dt # better accuracy than t+=dt
        
        # Assign to self
        self.u_ = u_
        self.p_ = p_
        self.u_n = u_n
        self.u_nn = u_nn
        self.p_n = p_n
        self.up_ = up_
        
        # Goto order 2 next time
        self.order = 2
     
        # Measurement --> on perturbation field = y_p
        self.y_meas = self.make_measurement(mixed_field=up_)
        
        tfi = time.time()
        if self.verbose and (not self.iter%self.verbose):
            self.print_progress(runtime=tfi-t0i)

        # Log timeseries
        cl, cd = 1, 2 #self.compute_force_coefficients(u_, p_)
        # Be careful: cl, cd, norms etc. are in perturbation formulation (miss u0, p0)
        # perturbation energy wrt base flow, here u_ = u_pert
        if self.compute_norms:
            dE = self.compute_energy(full=True, diff=True, normalize=True)
            #dE = norm(self.u_, norm_type='L2', mesh=self.mesh) / self.Eb
        else:
            dE = -1
            
        ##################################### here if dE is inf or nan, exit
        if not self.throw_error and (np.isinf(dE) or np.isnan(dE)):
            print('Error solving system --- Exiting step()...')
            return -1
        ####################################################################

        self.log_timeseries(u_ctrl=u_ctrl,
                            y_meas=self.y_meas,
                            norm_u = 0, # norm(u_, norm_type='L2', mesh=self.mesh),
                            norm_p = 0, # norm(p_, norm_type='L2', mesh=self.mesh),
                            dE = dE,
                            cl=cl, cd=cd,
                            t=self.t, runtime=tfi-t0i)
        # Save
        if self.save_every and not self.iter%self.save_every:
            self.u_full.vector()[:] = u_n.vector()[:] + self.u0.vector()[:]
            self.u_n_full.vector()[:] = u_nn.vector()[:] + self.u0.vector()[:]
            self.p_full.vector()[:] = p_n.vector()[:] + self.p0.vector()[:]
            if self.verbose:
                print("saving to files %s" %(self.savedir0))
            flu.write_xdmf(self.paths['u_restart'], self.u_full, "u", 
                       time_step=self.t, append=True, write_mesh=False)
            flu.write_xdmf(self.paths['uprev_restart'], self.u_n_full, "u_n", 
                       time_step=self.t, append=True, write_mesh=False)
            flu.write_xdmf(self.paths['p_restart'], self.p_full, "p", 
                       time_step=self.t, append=True, write_mesh=False)
            # this is asynchronous and calls process 0?
            self.write_timeseries()

        return 0


    def prepare_restart(self, Trestart):
        '''Prepare restart: set Tstart, redefine paths & files'''
        self.Tstart = Trestart
        self.define_paths()
        self.init_time_stepping()


    def make_solvers(self):
        '''Define solvers'''
        if self.equations == 'ipcs':
            if self.solver_type == 'Krylov':
                S1 = KrylovSolver('bicgstab', 'jacobi')
                S2 = KrylovSolver('bicgstab', 'petsc_amg') # pressure solver needs to be good
                S3 = KrylovSolver('bicgstab', 'jacobi')
                for s in [S1, S2, S3]:
                    sparam = s.parameters
                    sparam['nonzero_initial_guess'] = True
                    sparam['absolute_tolerance'] = 1e-8
                    sparam['relative_tolerance'] = 1e-8
                    sparam['error_on_nonconvergence'] = True # error + catch
            else:
                S1 = LUSolver('mumps')
                S2 = LUSolver('mumps')
                S3 = LUSolver('mumps')
            return [S1, S2, S3]
        else:
            return LUSolver('mumps')


    def make_measurement(self, field=None, mixed_field=None):
        '''Perform measurement and assign'''
        ns = self.sensor_nr
        y_meas = np.zeros((ns, ))

        ## for measuring: integrate dy(u) on sensor surface
        #yy = assemble(self.u_[0].dx(1) * ds_sensor(SENSOR_IDX))
        # or in another function: yy=assemble(up[0].dx(1) * self.ds_sensor(self.boundaries.loc['sensor'].idx))
        ds_sensor = self.ds_sensor
        SENSOR_IDX = int(self.boundaries.loc['sensor'].idx)

        # TODO
        # good thing, the syntax is the same whatever the field form!
        # Note 06/09: I don't know what that means and think this is false

        for isensor in range(ns):
            # field (u,v) is given
            if field is not None:
                # eval on field[0]
                ff = field
                #print('found field (maybe no u0)')
            else:
                # no field (u,v,p) is given
                if mixed_field is not None:
                    ff = mixed_field
                    #ff = self.u_
                    #print('found mixed field')
                else:
                # some field (u,v,p) is given
                    # eval on u_
                    ff = self.u_
                    #ff = flu.projectm(self.u_ + self.u0, self.V)
                    #print('found u_ + u0')
                    # eval on mixed_field[0][0]
            y_meas_i = assemble(ff.dx(1)[0] * ds_sensor(SENSOR_IDX)) 
            # WARNING
            # BELOW DOES NOT SEEM TO BE WORKING
            # FOR SOME UNKNOWN F-ING REASON 
            #y_meas_i = assemble(ff[0].dx(1) * ds_sensor(SENSOR_IDX)) 
            y_meas[isensor] = y_meas_i
        #print('measurement: ', y_meas_i)
        #import pdb
        #pdb.set_trace()
        return y_meas
    
    #vx = project(u.dx(1), V, solver_type='mumps')
    #vy = project(-u.dx(0), V, solver_type='mumps')
    #vvec = project(as_vector([vx, vy]), W, solver_type='mumps')

    #ds = Measure("ds", domain=mesh, subdomain_data=subdomains)
    #GammaP = ds(5)
    #n = FacetNormal(mesh)
    #tangent = as_vector([n[1], -n[0]])
    #L1 = (dot(vvec, tangent))*GammaP

    def assign_measurement_to_dataframe(self, df, y_meas, index):
        '''Assign measurement (array y_meas) to DataFrame at index
        Essentially convert array (y_meas) to separate columns (y_meas_i)'''
        y_meas_str = self.make_y_dataframe_column_name()
        for i_meas, name_meas in enumerate(y_meas_str):
            df.loc[index, name_meas] = y_meas[i_meas]  
    

    def make_y_dataframe_column_name(self):
        '''Return column names of different measurements y_meas_i'''
        return ['y_meas_'+str(i+1) for i in range(self.sensor_nr)]
        

    def write_timeseries(self):
        '''Write pandas DataFrame to file'''
        if flu.MpiUtils.get_rank() == 0:
            #zipfile = '.zip' if self.compress_csv else ''
            self.timeseries.to_csv(self.paths['timeseries'], sep=',', index=False)


    def create_indicator_function(self):
            # define subdomain by hand
            subdomain_str = 'x[0]<=10 && x[0]>=-2 && x[1]<=3 && x[1]>=-3'
            subdomain = CompiledSubDomain(subdomain_str)

            #class IndicatorFunction(flo.UserExpression):
            #    def __init__(self, subdomain, **kwargs):
            #        self.subdomain = subdomain
            #        super(IndicatorFunction, self).__init__(**kwargs)
            #    def eval(self, values, x):
            #        values[0] = 0
            #        values[1] = 0
            #        if self.subdomain.inside(x, True):
            #            values[0] = 1
            #            values[1] = 1
            #    def value_shape(self):
            #        return (2,)
            ## interpreted 
            #IF = IndicatorFunction(subdomain)

            # compiled
            IF = Expression([subdomain_str]*2, degree=0, element=self.V.ufl_element())
            # then project on V
            IF = flu.projectm(IF, V=self.V)
            self.IF = IF


    def compute_energy(self, field=None, full=True, diff=False, normalize=False):
        '''Compute energy of flow
            on full/restricted domain      (default:full=True)
            minus base flow                (default:diff=False)
            normalized by base flow energy (default:normalize=False)'''
        if field is not None:
            u = field
        else:
            u  = self.u_

        # Initialize at current state
        self.u_restrict.vector()[:] = u.vector()[:]
        # Add/remove steady
        if self.perturbations and not diff: 
            # perturbations + not diff >> add base flow
            self.u_restrict.vector()[:] += + self.u0.vector()[:]
        if not self.perturbations and diff:
            # not perturbations + diff >> remove base flow
            self.u_restrict.vector()[:] += - self.u0.vector()[:]
        # Apply restriction
        if not full:
            self.u_restrict.vector()[:] *= self.IF.vector()[:]
        # Compute energy
        # E = 1/2 xT Q x = 1/2 norm_Q(x)**2
        dE = 1/2 * norm(self.u_restrict, norm_type='L2', mesh=self.mesh)**2
        # Normalize
        if normalize:
            if full:
                dE /= self.Eb
            else:
                dE /= self.Eb_r
        return dE

    
    def get_Dxy(self):
        '''Get derivation matrices Dx, Dy in V space
        such that Dx*u = u.dx(0), Dy*u = u.dx(1)'''
        u = TrialFunction(self.V)
        ut = TestFunction(self.V)
        Dx = assemble(inner(u.dx(0), ut)*self.dx)
        Dy = assemble(inner(u.dx(1), ut)*self.dx)
        return Dx, Dy


    def get_A(self, perturbations=True, shift=0.0, timeit=True, up_0=None):
        '''Get state-space dynamic matrix A around some state up_0'''
        if timeit:
            print('Computing jacobian A...')
            t0 = time.time()

        Jac = PETScMatrix()
        v, q = TestFunctions(self.W)
        iRe = Constant(1/self.Re)
        shift = Constant(shift)
        self.actuator_expression.ampl = 0.0 

        if up_0 is None:
            up_ = self.up0 # base flow
        else:
            up_ = up_0
        u_, p_ = up_.split()


        if perturbations: # perturbation equations linearized
            up = TrialFunction(self.W)
            u, p = split(up)
            #u0 = self.u0
            dF0 = -dot( dot(u_, nabla_grad(u)), v)*dx \
               - dot( dot(u, nabla_grad(u_)), v)*dx \
               - iRe*inner(nabla_grad(u), nabla_grad(v))*dx \
               + p*div(v)*dx \
               + div(u)*q*dx \
               - shift*dot(u,v)*dx  # sum u, v but not p
            bcu = self.bc_p['bcu'] 
            bcs = bcu
        else: # full ns + derivative
            up_ = self.up0
            u_, p_ = split(up_)
            F0 = - dot(dot(u_, nabla_grad(u_)), v)*dx \
                - iRe*inner(nabla_grad(u_), nabla_grad(v))*dx \
                + p_*div(v)*dx \
                + q*div(u_)*dx \
                - shift*dot(u_,v)*dx
            # prepare derivation 
            du = TrialFunction(self.W)
            dF0 = derivative(F0, up_, du=du)
            ## shift
            #dF0 = dF0 - shift*dot(u_,v)*dx
            # bcs
            bcs = self.bc['bcu']
        
        assemble(dF0, tensor=Jac)
        [bc.apply(Jac) for bc in bcs]
        
        if timeit:
            print('Elapsed time: ', time.time() - t0)

        return Jac
        
        #if not hasattr(self, 'F0'):
        #    self.make_form_mixed_steady()
        
        #R = action(self.F0, self.up0) ## what is this?
        #DR = derivative(R, self.up0)
        # for lumping maybe
        #self.DR = DR


    def get_B(self, export=False, timeit=True):
        '''Get actuation matrix B'''
        print('Computing actuation matrix B...')

        if timeit:
            t0 = time.time()

        # for an exponential actuator -> just evaluate actuator_exp on every coordinate, kinda?
        # for a boundary actuator -> evaluate actuator on boundary
        actuator_ampl_old = self.actuator_expression.ampl
        self.actuator_expression.ampl = 1.0

        # Method 1 
        # restriction of actuation of boundary
        #class RestrictFunction(UserExpression):
        #    def __init__(self, boundary, fun, **kwargs):
        #        self.boundary = boundary
        #        self.fun = fun
        #        super(RestrictFunction, self).__init__(**kwargs)
        #    def eval(self, values, x):
        #        values[0] = 0
        #        values[1] = 0
        #        values[2] = 0
        #        if self.boundary.inside(x, True):
        #            evalval = self.fun(x)
        #            values[0] = evalval[0]
        #            values[1] = evalval[1]        
        #    def value_shape(self):
        #        return (3,)

        #Bi = []
        #for actuator_name in ['actuator_up', 'actuator_lo']:
        #    actuator_restricted = RestrictFunction(boundary=self.boundaries.loc[actuator_name].subdomain, 
        #                                           fun=self.actuator_expression)
        #    actuator_restricted = interpolate(actuator_restricted, self.W)
        #    #actuator_restricted = flu.projectm(actuator_restricted, self.W)
        #    Bi.append(actuator_restricted)

        # Method 2
        # Projet actuator expression on W
        class ExpandFunctionSpace(UserExpression):
            '''Expand function from space [V1, V2] to [V1, V2, P]'''
            def __init__(self, fun, **kwargs):
                self.fun = fun
                super(ExpandFunctionSpace, self).__init__(**kwargs)
            def eval(self, values, x):
                evalval = self.fun(x)
                values[0] = evalval[0]
                values[1] = evalval[1]
                values[2] = 0
            def value_shape(self):
                return (3,)

        actuator_extended = ExpandFunctionSpace(fun=self.actuator_expression)
        actuator_extended = interpolate(actuator_extended, self.W)
        B_proj = flu.projectm(actuator_extended, self.W)
        B = B_proj.vector().get_local()

        # remove very small values (should be 0 but are not)
        B = flu.dense_to_sparse(B, eliminate_zeros=True, eliminate_under=1e-14).toarray()
        B = B.T # vertical B

        if export:
            fa = FunctionAssigner([self.V, self.P], self.W)
            vv = Function(self.V)
            pp = Function(self.P)
            ww = Function(self.W)
            ww.assign(B_proj)
            fa.assign([vv, pp], ww)
            flu.write_xdmf('B.xdmf', vv, 'B')
    
        self.actuator_expression.ampl = actuator_ampl_old
   
        if timeit:
            print('Elapsed time: ', time.time() - t0)

        return B


    def get_C(self, timeit=True, check=False, verbose=False):
        '''Get measurement matrix C'''
        # Solution to make it faster:
        # localize the region of dofs where C is going to be nonzero
        # and only account for dofs in this region
        print('Computing measurement matrix C...')
        
        if timeit:
            t0 = time.time()

        # Initialize
        fspace = self.W # function space
        uvp = Function(fspace) # function to store C
        uvp_vec = uvp.vector() # as vector
        ndof = fspace.dim() # size of C
        ns = self.sensor_nr
        C = np.zeros((ns, ndof))

        dofmap = fspace.dofmap() # indices of dofs
        dofmap_x = fspace.tabulate_dof_coordinates() # coordinates of dofs

        # Box that encapsulates all dofs on sensor
        margin = 0.05
        xmin = 1 - margin 
        xmax = 1.1 + margin
        ymin = 0 - margin
        ymax = 0 + margin
        xymin = np.array([xmin, ymin]).reshape(1, -1)
        xymax = np.array([xmax, ymax]).reshape(1, -1)
        # keep dofs with coordinates inside box
        dof_in_box = (np.greater_equal(dofmap_x, xymin) * np.less_equal(dofmap_x, xymax)).all(axis=1) 
        # retrieve said dof index
        dof_in_box_idx = np.array(dofmap.dofs())[dof_in_box]
        
        # Iteratively put each DOF at 1
        # And evaluate measurement on said DOF
        idof_old = 0
        ii = 0 # counter of the number of dofs evaluated
        for idof in dof_in_box_idx:
            ii+=1 
            if verbose and not ii%1000:
                    print('get_C::eval iter {0} - dof n°{1}/{2}'.format(ii, idof, ndof))
            # set field 1 at said dof
            uvp_vec[idof] = 1
            uvp_vec[idof_old] = 0
            idof_old = idof
            # retrieve coordinates
            #dof_x = dofmap_x[idof] # not needed for measurement
            # evaluate measurement
            C[:, idof] = self.make_measurement(mixed_field=uvp)

        # check:
        if check:
            for i in range(ns):
                sensor_types = dict(u=0, v=1, p=2)
                #print('True probe: ', self.up0(self.sensor_location[i])[sensor_types[self.sensor_type[0]]])
                # true probe would be make_measurement(...)
                print('\t with fun:', self.make_measurement(mixed_field=self.up0))
                print('\t with C@x: ', C[i] @ self.up0.vector().get_local())

        if timeit:
            print('Elapsed time: ', time.time() - t0)

        return C
 

    def get_matrices_lifting(self, A, C, Q):
        '''Return matrices A, B, C, Q resulting form lifting transform (Barbagallo et al. 2009)
        See get_Hw_lifting for details'''
        # Steady field with rho=1: S1

        print('Computing steady actuated field...')
        self.actuator_expression.ampl = 1.0
        #S1 = self.compute_steady_state_newton()
        S1 = self.compute_steady_state_picard(max_iter=25)
        S1v = S1.vector()
        
        # Q*S1 (as vector)
        sz = self.W.dim()
        QS1v = Vector(S1v.copy())
        QS1v.set_local(np.zeros(sz,))
        QS1v.apply('insert')
        Q.mult(S1v, QS1v) # QS1v = Q * S1v

        # Bl = [Q*S1; -1]
        Bl = np.hstack((QS1v.get_local(), -1)) # stack -1
        Bl = np.atleast_2d(Bl).T # as column

        # Cl = [C, 0]
        Cl = np.hstack((C, np.atleast_2d(0)))

        # Ql = diag(Q, 1)
        Qsp = flu.dense_to_sparse(Q)
        Qlsp = flu.spr.block_diag((Qsp, 1))

        # Al = diag(A, 0)
        Asp = flu.dense_to_sparse(A)
        Alsp = flu.spr.block_diag((Asp, 0))

        return Alsp, Bl, Cl, Qlsp


    def get_mass_matrix(self, sparse=False, volume=True):
        '''Compute the mass matrix associated to 
        spatial discretization'''
        up = TrialFunction(self.W)
        vq = TestFunction(self.W)

        M = PETScMatrix()
        # volume integral or surface integral (unused)
        dOmega = self.dx if volume else self.ds

        mf = sum([up[i]*vq[i] for i in range(2)])*dOmega  # sum u, v but not p
        assemble(mf, tensor=M)

        if sparse:
            return flu.dense_to_sparse(M)
        return M


    def get_block_identity(self, sparse=False):
        '''Compute the block-identity associated to 
        the time-continuous, space-continuous formulation: 
        E*dot(x) = A*x >>> E = blk(I, I, 0), 0 being on p dofs'''
        dof_idx = flu.get_subspace_dofs(self.W) 
        sz = self.W.dim()
        diagE = np.zeros(sz)
        diagE[np.hstack([dof_idx[kk] for kk in ['u', 'v']])] = 1.0
        E = spr.diags(diagE, 0)
        if sparse:
            return E
        # cast
        return flu.sparse_to_petscmat(E)


    def check_mass_matrix(self, up=0, vq=0, random=True):
        '''Given two vectors u, v (Functions on self.W),
        compute assemble(dot(u,v)*dx) v.s. u.local().T @ Q @ v.local()
        The result should be the same'''
        if random:
            print('Creating random vectors')
            def createrandomfun():
                up = Function(self.W)
                up.vector().set_local((np.random.randn(self.W.dim(), 1)))
                up.vector().apply('insert') 
                return up
            up = createrandomfun()
            vq = createrandomfun()

        fa = FunctionAssigner([self.V, self.P], self.W)
        u = Function(self.V) # velocity only
        p = Function(self.P)
        v = Function(self.V)
        q = Function(self.P)
        fa.assign([u, p], up)
        fa.assign([v, q], vq)

        # True integral of velocities
        d1 = assemble( dot(u, v)*self.dx)
        
        # Discretized dot product (scipy)
        Q = self.get_mass_matrix(sparse=True)
        d2 = up.vector().get_local().T @ Q @ vq.vector().get_local()
        ## Note: u.T @ Qv = (Qv).T @ u
        # d2 = (Q @ v.vector().get_local()).T @ u.vector().get_local()
        
        # Discretized dot product (petsc)
        QQ = self.get_mass_matrix(sparse=False)
        uu = Vector(up.vector())
        vv = Vector(vq.vector())
        ww = Vector(up.vector()) # intermediate result
        QQ.mult(vv, ww) # ww = QQ*vv
        d3 = uu.inner(ww) 
        
        return {'integral': d1, 'dot_scipy': d2, 'dot_petsc': d3}
###############################################################################
###############################################################################
############################ END CLASS DEFINITION #############################
###############################################################################
###############################################################################










###############################################################################
###############################################################################
############################     RUN EXAMPLE      #############################
###############################################################################
###############################################################################
if __name__=='__main__':
    t000 = time.time()
    
    print('Trying to instantiate FlowSolver...')
    params_flow={'Re': 7500.0, 
                 'uinf': 1.0, 
                 'd': 1.0,
                 'sensor_location': np.array([[1.1, 0.1]]), # sensor 
                 'sensor_type': ['v'], # u, v, p only >> reimplement make_measurement
                 'actuator_angular_size': 10, # actuator angular size
                 } 
    params_time={'dt': 0.0004, # in Sipp: 0.0004 
                 'Tstart': 0, 
                 'num_steps': 10, # 1e6 
                 'Tc': 1000,
                 } 
    params_save={'save_every': 100000, 
                 'save_every_old': 2000,
                 'savedir0': '/scratchm/wjussiau/fenics-results/cavity/',
                 'compute_norms': True}
    params_solver={'solver_type': 'Krylov', 
                   'equations': 'ipcs',
                   'throw_error': True,
                   'perturbations': True, ######
                   'NL': True, ################# NL=False only works with perturbations=True
                   'init_pert': 0} # initial perturbation amplitude, np.inf=impulse (sequential only?)
    # cav0
    params_mesh = {'genmesh': False,
                   'remesh': False,
                   'nx': 1,
                   'meshpath': '/stck/wjussiau/fenics-python/mesh/', 
                   'meshname': 'cavity_byhand_n200.xdmf', #'cavity_byhand_n200.xdmf',
                   'xinf': 2.5,
                   'xinfa': -1.2,
                   'yinf': 0.5,
                   'segments': 540,
                   }

    fs = FlowSolver(params_flow=params_flow,
                    params_time=params_time,
                    params_save=params_save,
                    params_solver=params_solver,
                    params_mesh=params_mesh,
                    verbose=True)
    #alltimes = pd.DataFrame(columns=['1', '2', '3'], index=['assemble', 'solve'], data=np.zeros((2,3))) 
    print('__init__(): successful!')

    print('Compute steady state...')
    u_ctrl_steady = 0.0
    fs.compute_steady_state(method='picard', max_iter=5, tol=1e-9, u_ctrl=u_ctrl_steady)
    fs.compute_steady_state(method='newton', max_iter=25, u_ctrl=u_ctrl_steady, initial_guess=fs.up0)
    #fs.load_steady_state(assign=True)

    print('Init time-stepping')
    fs.init_time_stepping()
   
    print('Step several times')
    y_steady = 0 if fs.perturbations else fs.y_meas_steady # reference measurement
    u_ctrl = 0
    u_ctrl0 = 1e-1
    tlen = 0.15
    tpeak = 1 
    for i in range(fs.num_steps):
        # open loop
        u_ctrl = u_ctrl0 * np.exp(-1/2*(fs.t-tpeak)**2/tlen**2)

        if fs.perturbations:
            fs.step_perturbation(u_ctrl=u_ctrl, NL=fs.NL, shift=0.0)
        else:
            fs.step(u_ctrl) # step and take measurement

    if fs.num_steps > 3:
        print('Total time is: ', time.time() - t000)
        print('Iteration 1 time     ---', fs.timeseries.loc[1, 'runtime'])
        print('Iteration 2 time     ---', fs.timeseries.loc[2, 'runtime'])
        print('Mean iteration time  ---', np.mean(fs.timeseries.loc[3:, 'runtime']))
        print('Time/iter/dof        ---', np.mean(fs.timeseries.loc[3:, 'runtime'])/fs.W.dim())
    list_timings(TimingClear.clear, [TimingType.user])
    
    fs.write_timeseries()
    print(fs.timeseries)

## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------




