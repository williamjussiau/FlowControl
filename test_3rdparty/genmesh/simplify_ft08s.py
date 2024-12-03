"""
ft08s.py but all unused lines are removed
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
from fenicstools.Probe import Probe, Probes
from mpi4py import MPI as mpi 
import sympy as sp
import functools
from scipy import signal as ss

t000 = time.time()


# Options (high-level) ########################################################
# Steady-state or time-stepping
compute_steady_state = True # compute base flow (for perturbations or full-ns init) 
perturbations = False # use perturbation formulation
Re = 100
uinf = 1 
d = 1
r = d/2
nu = uinf*d/Re # dunnu touch

Tstart = 0 # 0 or >0 or -1 but messes up time scale
num_steps = 0
dt = 0.01# 0.02 
Tf = num_steps*dt # final time
Tc = 500 # start control at

# Files location
savedir0 = '/scratchm/wjussiau/fenics-results/cylinder_checkmesh/'
# dunnu touch below
file_extension = '' if Tstart==0 else '_restart'+str(Tstart).replace('.','')
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
filename_timeseries = savedir0 + 'timeseries1D' + file_extension_csv # all 1D timeseries in pandas
# extension is .xdmf if no restart
# else _restart(Tstart).xdmf
save_every_old = 100
save_every = 100 # can be 0 
##############################################################################





# Checks ######################################################################
class FlowSolver():
    pass

def run_simulation():
    print('coucou')


def check_process_rank():
    comm = mpi.COMM_WORLD
    ip = comm.Get_rank()
    print("================= Hello I am process ", ip)


def mpi4py_comm(comm):
    '''Get mpi4py communicator'''
    try:
        return comm.tompi4py()
    except AttributeError:
        return comm


def peval(f, x):
    '''Parallel synced eval'''
    try:
        yloc = f(x)
    except RuntimeError:
        yloc = np.inf*np.ones(f.value_shape())
    comm = mpi4py_comm(f.function_space().mesh().mpi_comm())
    yglob = np.zeros_like(yloc)
    comm.Allreduce(yloc, yglob, op=mpi.MIN)
    return yglob


def peval2(f, x):
    '''Parallel synced eval, v2'''
    comm = f.function_space().mesh().mpi_comm()
    if comm.size == 1:
        return f(*x)

    # Find whether the point lies on the partition of the mesh local
    # to this process, and evaulate u(x)
    cell, distance = mesh.bounding_box_tree().compute_closest_entity(Point(*x))
    f_eval = f(*x) if distance < DOLFIN_EPS else None
    # Gather the results on process 0
    comm = mesh.mpi_comm()
    computed_f = comm.gather(f_eval, root=0)
    # Verify the results on process 0 to ensure we see the same value
    # on a process boundary
    if comm.rank == 0:
        global_f_evals = np.array([y for y in computed_f if y is not None], dtype=np.double)
        assert np.all(np.abs(global_f_evals[0] - global_f_evals) < 1e-9)
        computed_f = global_f_evals[0]
    else:
        computed_f = None
    # Broadcast the verified result to all processes
    computed_f = comm.bcast(computed_f, root=0)
    return computed_f


def set_omp_num_threads(): 
    '''Memo for getting/setting OMP_NUM_THREADS'''
    try:
        print('nb threads was: ', os.environ['OMP_NUM_THREADS'])
    except:
        os.environ['OMP_NUM_THREADS'] = '1'
    print('nb threads is: ', os.environ['OMP_NUM_THREADS'])


set_log_level(LogLevel.INFO) # DEBUG TRACE PROGRESS INFO
###############################################################################





# Utility #####################################################################
def stress_tensor(u, p):
    '''Compute stress tensor (for lift & drag)'''
    return nu*(grad(u) + grad(u).T) - p*Identity(len(u))
def compute_force_coefficients(u, p):
    '''Compute lift & drag coefficients'''
    sigma = stress_tensor(u, p)
    Fo = -dot(sigma, n)
    drag_sym = Fo[0]*ds(CYLINDER_IDX)
    lift_sym = Fo[1]*ds(CYLINDER_IDX)

    drag = assemble(drag_sym)
    lift = assemble(lift_sym)

    cd = drag/(1/2*uinf**2*d)
    cl = lift/(1/2*uinf**2*d)
    return cl, cd
def apply_fun(u, fun):
    '''Shortcut for applying numeric method to dolfin.Function'''
    return fun(u.vector().get_local())
def show_max(u, name):
    '''Display max of dolfin.Function'''
    print('Max of vector "%s" is : %f' %(name, apply_fun(u, np.max)))
def write_xdmf(filename, func, name, time_step=0., append=False, write_mesh=True):
    '''Shortcut to write XDMF file with options & context manager'''
    with XDMFFile(MPI.comm_world, filename) as ff:
        ff.parameters['rewrite_function_mesh'] = write_mesh
        ff.parameters['functions_share_mesh'] = not write_mesh
        ff.write_checkpoint(func, name, time_step=time_step,
                            encoding=XDMFFile.Encoding.HDF5,
                            append=append)
def read_xdmf(filename, func, name, counter=-1):
    '''Shortcut to read XDMF file with context manager'''
    with XDMFFile(MPI.comm_world, filename) as ff:
        ff.read_checkpoint(func, name=name, counter=counter)

projectm = functools.partial(project, solver_type='mumps')
###############################################################################






# Mesh ########################################################################
# Create mesh
genmesh = False
meshdir = './results/' #'/stck/wjussiau/fenics-python/mesh/' 
# dunnu touch below
readmesh = True
if genmesh:
    xinf = 20 # 20 # 20
    yinf = 8 # 5 # 8
    xinfa = -5 # -5 # -10
    nx = 32 

    meshname = 'cylinder_' + str(nx) + '.xdmf'
    meshpath = os.path.join(meshdir, meshname)
    if not os.path.exists(meshpath):
        print('Mesh does not exist @:', meshpath)
        print('-- Creating mesh...')
        channel = Rectangle(Point(xinfa, -yinf), Point(xinf, yinf))
        cyl = Circle(Point(0.0, 0.0), d/2, segments=360)
        domain = channel - cyl
        mesh = generate_mesh(domain, nx) # was 64
        with XDMFFile(MPI.comm_world, meshpath) as fm:
            fm.write(mesh)
        readmesh = False
else:
    xinf = 20
    yinf = 8
    xinfa = -10
    #meshname = 'cylinder_onesurf.xdmf'
    #meshname = 'cylinder_fine.xdmf'
    #meshname = 'cylinder_onesurf.xdmf'
    #meshname = 'cylinder_multizone_x20_xa10_y8_n5-3-1.xdmf'

    # the following should give the same result because visually equivalent!!!
    #meshname = 'cylinder_onesurf.xdmf'
    meshname = 'cylinder_onesurf_30-4.xdmf'

if readmesh:
    mesh = Mesh()
    meshpath = os.path.join(meshdir, meshname)
    print('Mesh exists @: ', meshpath)
    print('--- Reading mesh...')
    with XDMFFile(MPI.comm_world, meshpath) as fm:
        fm.read(mesh)

print('Mesh has: %d cells' %(mesh.num_cells()))


# Function spaces on mesh
Ve = VectorElement('CG', mesh.ufl_cell(), 2) # was 'P'
Pe = FiniteElement('CG', mesh.ufl_cell(), 1) # was 'P'
We = MixedElement([Ve, Pe])
V = FunctionSpace(mesh, Ve)
P = FunctionSpace(mesh, Pe)
W = FunctionSpace(mesh, We) 

# Define boundaries
boundary_markers = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
MESH_TOL = DOLFIN_EPS 

## Inlet
inlet = CompiledSubDomain('near(x[0], xinfa, MESH_TOL)', xinfa=xinfa, MESH_TOL=MESH_TOL)

## Outlet
outlet = CompiledSubDomain('near(x[0], xinf, MESH_TOL)', xinf=xinf, MESH_TOL=MESH_TOL)

## Walls
walls = CompiledSubDomain('near(x[1], -yinf, MESH_TOL) || near(x[1], yinf, MESH_TOL)', 
            yinf=yinf, MESH_TOL=MESH_TOL)

## Cylinder
close_to_cylinder = lambda x: between(x[0], (-1, 1)) and between(x[1], (-1, 1)) # slow af
theta0 = -0 * pi/180 # angular position of actuator with respect to vertical
###### warning with theta0 : velocity profile is not compatible with theta0 not 0 !!!
delta = 10 * pi/180 # angular size of acutator
get_theta = lambda x: ufl.atan_2(x[1], x[0])
theta_tol = 3*pi/180 # not needed so far
cone_ri = lambda theta: between(theta, (-pi/2+delta/2 - theta_tol, +pi/2-delta/2 + theta_tol))
cone_le = lambda theta: between(theta, (-pi, -pi/2-delta/2+theta_tol)) or between(theta, (pi/2+delta/2-theta_tol, pi))
cone_up = lambda theta: between(theta - pi/2, (-delta/2 - theta_tol, +delta/2 + theta_tol))
cone_lo = lambda theta: between(theta + pi/2, (-delta/2 - theta_tol, +delta/2 + theta_tol))

class boundary_cylinder(SubDomain):
    def inside(self, x, on_boundary):
        theta = get_theta(x)
        return on_boundary and close_to_cylinder(x) and (cone_le(theta) or cone_ri(theta))
cylinder = boundary_cylinder()

class boundary_cylinder_actuator_up(SubDomain):
    def inside(self, x, on_boundary):
        theta = get_theta(x)
        return on_boundary and close_to_cylinder(x) and cone_up(theta)# or cone_lo(theta))
actuator_up = boundary_cylinder_actuator_up()

class boundary_cylinder_actuator_lo(SubDomain):
    def inside(self, x, on_boundary):
        theta = get_theta(x)
        return on_boundary and close_to_cylinder(x) and cone_lo(theta)
actuator_lo = boundary_cylinder_actuator_lo()




# Sensor definition ###########################################################
xs = np.array([1.5, 0.0]) # 1.5 0.5
probe = Probes(xs, W)
###############################################################################




# Actuator definition ###########################################################
actuator_bc = Expression(['0','ampl*exp(-0.5 * x[0]*x[0] / den)'], 
    element=Ve, ampl=1, den=(r*sin(delta/2)/5)**2)
###############################################################################



# Boundary conditions #########################################################
INLET_IDX = 0
OUTLET_IDX = 1
WALLS_IDX = 2
CYLINDER_IDX = 3
CYLINDER_ACTUATOR_UP_IDX = 5
CYLINDER_ACTUATOR_LO_IDX = 6

inlet.mark(boundary_markers, INLET_IDX)
outlet.mark(boundary_markers, OUTLET_IDX)
walls.mark(boundary_markers, WALLS_IDX)
cylinder.mark(boundary_markers, CYLINDER_IDX)
actuator_up.mark(boundary_markers, CYLINDER_ACTUATOR_UP_IDX)
actuator_lo.mark(boundary_markers, CYLINDER_ACTUATOR_LO_IDX)

# Dirichlet BC
inlet_tau = 100*dt
class inlet_velocity(UserExpression):
    def eval(self, value, x):
        value[0] = uinf * (1 - exp(-self.t / self.tau))
        value[1] = 0.0
    def value_shape(self):
        return (2,)
inlet_expr = inlet_velocity(element=Ve)
inlet_expr.tau = inlet_tau
bcu_inlet = DirichletBC(W.sub(0), Constant((uinf, 0)), inlet)
bcu_walls = DirichletBC(W.sub(0).sub(1), Constant(0), walls)
bcu_cylinder = DirichletBC(W.sub(0), Constant((0, 0)), cylinder)
bcu_actuation_up = DirichletBC(W.sub(0), Constant((0, 0)), actuator_up)
bcu_actuation_lo = DirichletBC(W.sub(0), Constant((0, 0)), actuator_lo)
bcu = [bcu_inlet, bcu_walls, bcu_cylinder, bcu_actuation_up, bcu_actuation_lo]


# Measures (e.g. subscript ds(INLET_MARKER)) ##################################
ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
dx = Measure('dx', domain=mesh, subdomain_data=boundary_markers)
###############################################################################




# Trial and test functions ####################################################
v, q = TestFunctions(W)
up_ = Function(W)
u_, p_ = split(up_)
# for nonlinear solve
up = Function(W)
#u, p = split(up)
# for linear solve
u, p = TrialFunctions(W) # equiv: up = TrialFunction(W) then up[0] is u, up[1] is p

# Define expressions used in variational forms
n  = FacetNormal(mesh)
f  = Constant((0, 0)) # shape of actuation
uf = Constant(0) # amplitude of actuation
Re = Constant(Re)
iRe = Constant(1/Re)
II = Identity(2)
nu = Constant(nu)
k = Constant(dt)
##############################################################################






# Steady-state ################################################################
# Problem
F0 = dot(dot(u_, nabla_grad(u_)), v)*dx \
    + iRe*inner(nabla_grad(u_), nabla_grad(v))*dx \
    - p_*div(v)*dx \
    - q*div(u_)*dx
    # + inner(uf*f, v)*dx
# Param
nl_solver_param = {"newton_solver":
                        {
                        "linear_solver" : "mumps", 
                        "preconditioner" : "default",
                        "maximum_iterations" : 25,
                        }
                  }

if Tstart == 0 and compute_steady_state: # restart from 0
    # Solve
    solve(F0==0, up_, bcu, solver_parameters=nl_solver_param)

    # Extract & save
    u0, p0 = up_.split(deepcopy=True)
    u0 = projectm(v=u0, V=V, bcs=bcu, mesh=mesh)
    p0 = projectm(v=p0, V=P, bcs=bcu, mesh=mesh)

    if save_every:
        write_xdmf(filename_u0, u0, "u", time_step=0.0, append=False, write_mesh=True)
        write_xdmf(filename_p0, p0, "p", time_step=0.0, append=False, write_mesh=True)
    print('Stored base flow in: ', savedir0)
else:
    u0 = Function(V) 
    p0 = Function(P)
    read_xdmf(filename_u0, u0, "u")
    read_xdmf(filename_p0, p0, "p")

# Compute lift & drag
cl, cd = compute_force_coefficients(u0, p0)
print('Lift coefficient is: cl =', cl)
print('Drag coefficient is: cd =', cd)
###############################################################################


def picard_iteration(nip=10, tol=1e-14):
    '''Picard iteration for computing steady flow
    Should have a larger convergence radius than Newton method'''
    
    # for residual computation
    bcu_inlet0 = DirichletBC(W.sub(0), Constant((0, 0)), inlet)
    bcu0 = [bcu_inlet0, bcu_walls, bcu_cylinder, bcu_actuation_up, bcu_actuation_lo]
    
    # define forms
    up0 = Function(W)
    up1 = Function(W)

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    
    class initial_condition(UserExpression):
        def eval(self, value, x):
            value[0] = 1.0
            value[1] = 0.0
            value[2] = 0.0
        def value_shape(self):
            return (3,)
    
    up0.interpolate(initial_condition())
    u0 = as_vector((up0[0], up0[1]))
    u1 = as_vector((up1[0], up1[1]))

    ap = dot( dot(u0, nabla_grad(u)), v) * dx \
        + iRe*inner(nabla_grad(u), nabla_grad(v))*dx \
        - p*div(v)*dx \
        - q*div(u)*dx # steady lhs
    Lp = Constant(0)*inner(u0, v)*dx + Constant(0)*q*dx # zero rhs
    bp = assemble(Lp)

    solverp = LUSolver('mumps')

    for i in range(nip):
        Ap = assemble(ap)
        [bc.apply(Ap, bp) for bc in bcu]

        solverp.solve(Ap, up1.vector(), bp)

        up0.assign(up1)
        u, p = up1.split()

        # show_max(u, 'u')
        res = assemble(action(ap, up1))
        [bc.apply(res) for bc in bcu0]
        res_norm = norm(res)/sqrt(W.dim())
        print('Picard iteration: {0}/{1}, residual: {2}'.format(i+1, nip, res_norm))
        if res_norm < tol:
            print('Residual norm lower than tolerance {0}'.format(tol))
            break
    
    u1, p1 = up1.split(deepcopy=True)
    return u1, p1



# Initial perturbations ######################################################
def get_div0_u():
    '''Create velocity field with zero divergence'''
    PP = FunctionSpace(mesh, "CG", 2)
    QQ = FunctionSpace(mesh, "CG", 1)
    
    x0 = 5
    y0 = 0
    nsig = 5
    sigm = 0.5
    xm, ym = sp.symbols('x[0], x[1]')
    rr = (xm-x0)**2 + (ym-y0)**2
    fpsi = 0.25 * sp.exp(-1/2 * rr / sigm**2)
    #fpsi = sp.Piecewise(   (sp.exp(-1/2 * rr / sigm**2), rr <= nsig**2 * sigm**2), (0, True) )
    dfx = fpsi.diff(xm, 1)
    dfy = fpsi.diff(ym, 1)
    
    psi = Expression(sp.ccode(fpsi), element=PP.ufl_element())
    dfx_expr = Expression(sp.ccode(dfx), element=QQ.ufl_element())
    dfy_expr = Expression(sp.ccode(dfy), element=QQ.ufl_element())
    
    # psiproj = projectm(psi, PP)
   
    # Check
    # write_xdmf('psi.xdmf', psiproj, 'psi')
    # write_xdmf('psi_dx.xdmf', projectm(dfx_expr, QQ), 'psidx')
    # write_xdmf('psi_dy.xdmf', projectm(dfy_expr, QQ), 'psidy')
    
    # Make velocity field
    upsi = projectm(as_vector([dfy_expr, -dfx_expr]), V)
    return upsi
###############################################################################







# Time-stepping ###############################################################
t = Tstart
#inlet_expr.t = t

if Tstart == 0: 
    order = 1
    # assign steady state
    pert0 = get_div0_u()
    #pert0 = interpolate(localized_perturbation_u(element=Ve), V)
    u1 = u0.copy(deepcopy=True)
    u1.vector().set_local(u1.vector().get_local() + pert0.vector().get_local())
    u_n = projectm(v=u1, V=V, bcs=bcu)
    u_nn = u0.copy(deepcopy=True)
    p_n = interpolate(p0, P)

    # Flush file
    if save_every:
        write_xdmf(filename_u_restart, u_n, "u",
                   time_step=0., append=False, write_mesh=True)
        write_xdmf(filename_uprev_restart, u_nn, "u_n",
                   time_step=0., append=False, write_mesh=True)
        write_xdmf(filename_p_restart, p_n, "p",
                   time_step=0., append=False, write_mesh=True)
else:
    idxstart = -1 if(Tstart==-1) else int(np.floor(Tstart/dt/save_every_old)) 
    # replace by old_save_every
    order = 2
    # assign previous solution
    u_n = Function(V)
    read_xdmf(filename_u, u_n, "u", counter=idxstart)
    u_nn = Function(V)
    read_xdmf(filename_uprev, u_nn, "u_n", counter=idxstart)
    p_n = Function(P)

    write_xdmf(filename_u_restart, u_n, "u", time_step=Tstart, append=False, write_mesh=True)
    write_xdmf(filename_uprev_restart, u_nn, "u_n", time_step=Tstart, append=False, write_mesh=True)

    
print('Starting or restarting from time: ', Tstart, 
      ' with temporal scheme order: ', order)



colnames = ['time', 'u_ctrl', 'p_meas', 'u_norm', 'p_norm', 'cl', 'cd', 'runtime']
ts1d = pd.DataFrame(np.zeros((num_steps+1, len(colnames))), columns=colnames)

u_ctrl = Constant(0)
probe(p0)
ts1d.loc[0, 'p_meas'] = probe.array(0, component=0)  #p0(sensor_point['x'], sensor_point['y'])
ts1d.loc[0, 'cl'], ts1d.loc[0, 'cd'] = cl, cd
ts1d.loc[0, 'u_norm'] = norm(u0, norm_type='L2', mesh=mesh)
ts1d.loc[0, 'p_norm'] = norm(p0, norm_type='L2', mesh=mesh)





#################### IPCS #################################
bcu_inlet_g = DirichletBC(V, Constant((1, 0)), inlet)
bcu_walls_g = DirichletBC(V.sub(1), Constant(0), walls)
bcu_cylinder_g = DirichletBC(V, Constant((0, 0)), cylinder)
bcu_actuation_up_g = DirichletBC(V, actuator_bc, actuator_up)
bcu_actuation_lo_g = DirichletBC(V, actuator_bc, actuator_lo)
bcug = [bcu_inlet_g, bcu_walls_g, bcu_cylinder_g, bcu_actuation_up_g, bcu_actuation_lo_g]

bcp_outlet_g = DirichletBC(P, Constant(0), outlet)
bcpg = [bcp_outlet_g]

# Define trial and test functions
ug = TrialFunction(V)
vg = TestFunction(V)
pg = TrialFunction(P)
qg = TestFunction(P)

# Define functions for solutions at previous and current time steps
ug_  = Function(V)
pg_  = Function(P)
ug_n = u_n.copy(deepcopy=True)
pg_n = p_n.copy(deepcopy=True)

# Define expressions used in variational forms
U  = 0.5*(ug_n + ug)

# Define variational problem for step 1
F1g = dot((ug - ug_n) / k, vg)*dx \
   + dot( dot(ug_n, nabla_grad(ug_n)), vg)*dx \
   - dot(f, vg)*dx \
   + inner(iRe*nabla_grad(U), nabla_grad(vg))*dx \
   - inner(pg_n*II, nabla_grad(vg))*dx
a1g = lhs(F1g)
L1g = rhs(F1g)
# Define variational problem for step 2
a2g = dot(nabla_grad(pg), nabla_grad(qg))*dx
L2g = dot(nabla_grad(pg_n), nabla_grad(qg))*dx - (1/k)*div(ug_)*qg*dx
# Define variational problem for step 3
a3g = dot(ug, vg)*dx
L3g = dot(ug_, vg)*dx - k*dot(nabla_grad(pg_ - pg_n), vg)*dx

#if 0:
#    # Assemble matrices
#    A1g = PETScMatrix()
#    A2g = PETScMatrix()
#    A3g = PETScMatrix()
#    assemble(a1g, tensor=A1g)
#    assemble(a2g, tensor=A2g)
#    assemble(a3g, tensor=A3g)
#    
#    # Apply boundary conditions to matrices
#    [bc.apply(A1g) for bc in bcug]
#    [bc.apply(A2g) for bc in bcpg]

sysAssmb1 = [SystemAssembler(a1g, L1g, bcug),
             SystemAssembler(a2g, L2g, bcpg),
             SystemAssembler(a3g, L3g, bcug)]
AA1 = [PETScMatrix() for i in range(3)]
for assmblr, A in zip(sysAssmb1, AA1):
    assmblr.assemble(A)


################### Order 2 #######################
ug_nn = ug_n.copy(deepcopy=True)
# Define variational problem for step 1
F1g2 = dot((3*ug - 4*ug_n + ug_nn) / (2*k), vg)*dx \
   + 2*dot( dot(ug_n, nabla_grad(ug_n)), vg)*dx \
   - dot( dot(ug_nn, nabla_grad(ug_nn)), vg)*dx \
   - dot(f, vg)*dx \
   + inner(iRe*nabla_grad(ug), nabla_grad(vg))*dx \
   - inner(pg_n*II, nabla_grad(vg))*dx
a1g2 = lhs(F1g2)
L1g2 = rhs(F1g2)

# Define variational problem for step 2
a2g2 = dot(nabla_grad(pg), nabla_grad(qg))*dx
L2g2 = dot(nabla_grad(pg_n), nabla_grad(qg))*dx - 3/(2*k)*div(ug_)*qg*dx

# Define variational problem for step 3
a3g2 = dot(ug, vg)*dx
L3g2 = dot(ug_, vg)*dx - (2*k)/3*dot(nabla_grad(pg_ - pg_n), vg)*dx

A1g2 = PETScMatrix()
A2g2 = PETScMatrix()
A3g2 = PETScMatrix()
assemble(a1g2, tensor=A1g2)
assemble(a2g2, tensor=A2g2)
assemble(a3g2, tensor=A3g2)
# Apply boundary conditions to matrices
[bc.apply(A1g2) for bc in bcug]
[bc.apply(A2g2) for bc in bcpg]


# Create XDMF files for visualization output
savedir_g = '/scratchm/wjussiau/fenics-results/cylinder_ipcs/'
file_ug = savedir_g + 'ug.xdmf'
file_pg = savedir_g + 'pg.xdmf'
write_xdmf(file_ug, ug_n, 'ug', time_step=0, append=False, write_mesh=True)
write_xdmf(file_pg, pg_n, 'pg', time_step=0, append=False, write_mesh=True)


solver_type = 'Krylov'
if solver_type == 'Krylov':
    S1 = KrylovSolver('bicgstab', 'sor')# 'hypre_amg') # hypre_euclid
    S2 = KrylovSolver('bicgstab', 'hypre_amg')# 'hypre_amg') # hypre_euclid
    S3 = KrylovSolver('cg', 'sor')
    SS = [S1, S2, S3]
    # info(S1.parameters, True)
    for s in SS:
        sparam = s.parameters
        sparam['nonzero_initial_guess'] = True
        sparam['absolute_tolerance'] = 1e-8
        sparam['relative_tolerance'] = 1e-6
else:
    S1 = LUSolver('mumps')
    S2 = LUSolver('mumps')
    S3 = LUSolver('mumps')

for s, Amat in zip([S1, S2, S3], AA1):
    s.set_operator(Amat)



u_ = Function(V)
p_ = Function(P)
#################### IPCS #################################


p0m = peval(p0, xs)

## controller
kp = 10**0.560218037
ki = 10**-3.9749
kd = 0.0
Ts = dt
Td = 3*dt

num = [kd+kp*Td, kp+ki*Td, ki]
den = [Td, 1, 0]
tf_c = ss.TransferFunction(num, den)
ss_c = tf_c.to_ss()
ss_d = ss_c.to_discrete(dt)

x_ctrl = np.zeros((2,))
y_ctrl = np.zeros((1,))


b1g = PETScVector(MPI.comm_world, V.dim())
b2g = PETScVector(MPI.comm_world, P.dim())
b3g = PETScVector(MPI.comm_world, V.dim())

################ Loop here ##############
set_log_level(LogLevel.INFO)
for ii in range(num_steps):
    print('----- Computing step: %d/%d (time is: %.4f/%1.5f) -------' 
        % (ii+1, num_steps, t+dt, Tstart+Tf))
    t0 = time.time()
   
    actuator_bc.ampl = u_ctrl.values()[0]
    
    if 1:
        # IPCS
        if 1:#ii==0 or order==1:
            # Step 1: Tentative velocity step
            for assmbl, solver, bb, upsol in zip(sysAssmb1, [S1, S2, S3], [b1g, b2g, b3g], (ug_, pg_, ug_)):
                assmbl.assemble(bb)
                solver.solve(upsol.vector(), bb)

            #sysAssmb1[0].assemble(b1g)
            #S1.solve(ug_.vector(), b1g)
            ## Step 2: Pressure correction step
            #sysAssmb1[1].assemble(b2g)
            #S2.solve(pg_.vector(), b2g)
            ## Step 3: Velocity correction step
            #sysAssmb1[2].assemble(b3g)
            #S3.solve(ug_.vector(), b3g)
            order = 2
        else:
            # Step 1: Tentative velocity step
            ts1 = time.time()
            b1g = assemble(L1g2)
            [bc.apply(b1g) for bc in bcug]
            ct1 = S1.solve(A1g2, ug_.vector(), b1g)
            rt1 = time.time() - ts1
            # Step 2: Pressure correction step
            ts2 = time.time()
            b2g = assemble(L2g2)
            [bc.apply(b2g) for bc in bcpg]
            ct2 = S2.solve(A2g2, pg_.vector(), b2g)
            rt2 = time.time() - ts2
            # Step 3: Velocity correction step
            ts3 = time.time()
            b3g = assemble(L3g2)
            # no BC (or actuator ?)
            ct3 = S3.solve(A3g2, ug_.vector(), b3g)
            rt3 = time.time() - ts3

            print('Step 1: ', ct1, rt1)
            print('Step 2: ', ct2, rt2)
            print('Step 3: ', ct3, rt3)


        # Update previous solution
        ug_nn.assign(ug_n)
        ug_n.assign(ug_)
        pg_n.assign(pg_)

        u_.assign(ug_)
        p_.assign(pg_)
        u_n.assign(ug_n)
    ###################################################################

    # extract sensor data (used in next time step)
    tprobe = time.time()
    probe(p_)
    p_meas = MPI.comm_world.bcast(probe.array(ii+1, component=0), root=0)   # 0
    print('local probe is: ', p_meas, '\t ---- in: ', time.time() - tprobe)

    # Compute control (used in next time step)
    if t>=Tc:
        p_meas_sh = np.array([p_meas - p0m])
        x_ctrl = ss_d.A@x_ctrl + ss_d.B@p_meas_sh
        y_ctrl = ss_d.C@x_ctrl + ss_d.D@p_meas_sh
        u_ctrl.assign(y_ctrl)
        print('controlling flow with: ', y_ctrl)

    # Advance time
    t += dt
    #inlet_expr.t = t
    
    # Save
    if save_every and not (ii+1)%save_every:
        print("saving to files %s" %(savedir0))
        write_xdmf(filename_u_restart, u_, "u", time_step=t, append=True, write_mesh=False)
        write_xdmf(filename_uprev_restart, u_n, "u_n", time_step=t, append=True, write_mesh=False)
        write_xdmf(filename_p_restart, p_, "p", time_step=t, append=True, write_mesh=False)

    show_max(u_,  'u  ')

    # Assign new
    u_nn.assign(u_n)
    u_n.assign(u_)

    # Elapsed time
    t1 = time.time()
    print('elapsed time ****************************************************************** ', t1 - t0)
    print('')
    
    # Log 1D
            #+ norm(f)*norm(uf, mesh=mesh)
    ts1d.loc[ii+1, 'u_ctrl'] = u_ctrl.values()[0]
    ts1d.loc[ii+1, 'p_meas'] = p_meas 
    ts1d.loc[ii+1, 'u_norm'] = norm(u_, norm_type='L2', mesh=mesh)
    ts1d.loc[ii+1, 'p_norm'] = norm(p_, norm_type='L2', mesh=mesh)
    ts1d.loc[ii+1, 'cl'], ts1d.loc[ii+1, 'cd'] = compute_force_coefficients(u_, p_)
    ts1d.loc[ii+1, 'time'] = t
    ts1d.loc[ii+1, 'runtime'] = t1 - t0

list_timings(TimingClear.clear, [TimingType.wall, TimingType.user])
if num_steps > 3:
    print('Total time is: ', time.time() - t000)
    print('Iteration 1 time     ---', ts1d.loc[1, 'runtime'])
    print('Iteration 2 time     ---', ts1d.loc[2, 'runtime'])
    print('Mean iteration time  ---', np.mean(ts1d.loc[3:, 'runtime']))
    print('Time/iter/dof        ---', np.mean(ts1d.loc[3:, 'runtime'])/W.dim() )

# Write pandas DataFrame
if save_every and MPI.comm_world.Get_rank()==0:
    ts1d.to_csv(filename_timeseries, sep=',', index=False)



