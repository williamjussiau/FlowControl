'''Test function for dofmap coordinates in parallel
The objective is to transpose FlowSolver.get_B() to MPI'''

from dolfin import *
import numpy as np
import time
import sys

from mpi4py import MPI as mpi 


class MpiUtils():
    @staticmethod
    def get_rank():
        '''Access MPI rank in COMM WORLD'''
        return mpi.COMM_WORLD.Get_rank()


    @staticmethod
    def check_process_rank():
        '''Check process MPI rank'''
        comm = mpi.COMM_WORLD
        ip = comm.Get_rank()
        print("================= Hello I am process ", ip)
    

    @staticmethod
    def mpi4py_comm(comm):
        '''Get mpi4py communicator'''
        try:
            return comm.tompi4py()
        except AttributeError:
            return comm
    

    @staticmethod
    def peval(f, x):
        '''Parallel synced eval'''
        try:
            yloc = f(x)
        except RuntimeError:
            yloc = np.inf*np.ones(f.value_shape())
        comm = MpiUtils.mpi4py_comm(f.function_space().mesh().mpi_comm())
        yglob = np.zeros_like(yloc)
        comm.Allreduce(yloc, yglob, op=mpi.MIN)
        return yglob
 

if __name__ == '__main__':
    n3 = 5
    mesh = UnitSquareMesh(n3, n3)
    print('Make mesh of size: ', mesh.num_cells())
    # function space
    V = FunctionSpace(mesh, 'CG', 2)
    print('Make fspace of size: ', V.dim())
    # arbitrary expression
    u = Expression('100*(x[0]+x[1])', degree=2)
    print('Make expression')
    # mesh coordinates
    mesh_coords = mesh.coordinates()

    boundary = CompiledSubDomain('on_boundary and near(x[0], 0, DOLFIN_EPS)')
    bc = DirichletBC(V, Constant(0), boundary)
    boundary_idx = list(bc.get_boundary_values().keys())
    print('Make BC')
    print('dofs on BC:', boundary_idx)
    
    print('ownership range:', V.dofmap().ownership_range())
    dof_x = V.tabulate_dof_coordinates()
    print('dof_x is of size:', len(dof_x))
    dofmap = V.dofmap()
    dofs = dofmap.dofs()
    print('dof idx is of size:', len(dofs))

    #print('I own the folling dofs: %a, with coordinates: %a' %(dofs, dof_x))

    # Method 1 (with restriction)
    class RestrictedB(UserExpression):
        def __init__(self, boundary, fun, **kwargs):
            self.boundary = boundary
            self.fun = fun
            super(RestrictedB, self).__init__(**kwargs)
        def eval(self, values, x):
            if self.boundary.inside(x, True):
                values[0] = self.fun(x)
                #values[1] = 1
            else:
                values[0] = 0.0
                #values[1] = 0.0
        def value_shape(self):
            return ()

    rb = RestrictedB(boundary=boundary, fun=u)
    rb_vec = interpolate(rb, V).vector().get_local()

    # Method 2 (with loop)
    B = np.zeros( (V.dim(),), dtype=np.float64)
    print('B is of size:', len(B))
    for idof, coord in enumerate(dof_x):
        #print('helo iter %d with coord %a' %(idof, coord))
        # find if on boundary
        is_on_boundary = boundary.inside(coord, True)
        if is_on_boundary:
            print('dof is on boundary with coord: %a' %(coord))
            print('index: loop, local, global: ', 
                idof, 
                dofs[idof],)
            B[idof] = u(coord)
    
    np.set_printoptions(suppress=True)
    print(B)
    print(rb_vec)
    print('')





    # Now with a MixedElement FunctionSpace and a complicated Function to be evaluated
    # fspace = (V(2), P(1))
    # function returns a vector of size 2
    print('==========================================')
    for i in range(5):
        print(':::::')
    print('==========================================')

    n3 = 20 
    # mesh
    mesh = UnitSquareMesh(n3, n3)
    # function space
    Ve = VectorElement('CG', mesh.ufl_cell(), 2)
    Pe = FiniteElement('CG', mesh.ufl_cell(), 1) 
    We = MixedElement([Ve, Pe])
    W = FunctionSpace(mesh, We) 
    V = FunctionSpace(mesh, Ve) 
    P = FunctionSpace(mesh, Pe) 
    # expression (actuator)
    actu = Expression(['0.0', '1.0'], degree=0)
    # bc
    boundary = CompiledSubDomain('on_boundary and near(x[0], 0, DOLFIN_EPS)')
    boundary_idx = []
    for fspace in [W.sub(0).sub(0), W.sub(0).sub(1), W.sub(1)]:
        bc = DirichletBC(fspace, Constant(0), boundary)
        boundary_idx += list(bc.get_boundary_values().keys())
    print('dofs on BC:', boundary_idx)

    # Method 1 
    # restriction of actuation of boundary
    class RestrictedB(UserExpression):
        def __init__(self, boundary, fun, **kwargs):
            self.boundary = boundary
            self.fun = fun
            super(RestrictedB, self).__init__(**kwargs)
        def eval(self, values, x):
            values[0] = 0
            values[1] = 0
            values[2] = 0
            if self.boundary.inside(x, True):
                evalval = self.fun(x)
                values[0] = evalval[0]
                values[1] = evalval[1]        
        def value_shape(self):
            return (3,)

    rb3 = interpolate(RestrictedB(boundary=boundary, fun=actu), W)
    
    # this is supposedly B
    rb3_vec = interpolate(rb3, W).vector().get_local()

    # Method 2 (with loop)
    dof_x = W.tabulate_dof_coordinates() 
    dof = W.dofmap().dofs()
    B3 = np.zeros( (W.dim(),), dtype=np.float64)
    print('B3 is of size:', len(B3))

    def get_subspace_dofs(V):
        def get_dofs(V): return np.array(V.dofmap().dofs(), dtype=int)
        subspace_dof = {'u': get_dofs(W.sub(0).sub(0)),
                        'v': get_dofs(W.sub(0).sub(1)),
                        'p': get_dofs(W.sub(1))}
        return subspace_dof
    subspace_dof = get_subspace_dofs(W)

    #for idof, coord in enumerate(dof_x):
    #    is_on_boundary = boundary.inside(coord, True)

    #    nrdof = dof[idof] 
    #    if nrdof in subspace_dof['u']:
    #        dimension_index = 0
    #    elif nrdof in subspace_dof['v']:
    #        dimension_index = 1
    #    else: # nrdof in subspace_dof['p']:
    #        dimension_index = 2

    #    if dimension_index in [0, 1]:
    #        actu_value = actu(coord)[dimension_index]
    #    else:
    #        actu_value = 0.0

    #    B3[nrdof] = actu_value 
    #B3fun = Function(W)
    #B3fun.vector().set_local(B3)
    
    # Export methods 1 & 2
    fa = FunctionAssigner([V, P], W)
    vv = Function(V)
    pp = Function(P)
    for i, fun in enumerate([rb3]):#, B3fun]):
        ww = Function(W)
        ww.assign(fun)
        fa.assign([vv, pp], ww)
    
        file_v = File('vv_m' + str(i) + '.pvd')
        file_v << vv

    # Method 1 works, method 2 does work in parallel and gives wrong result

#    sys.exit()
#    ################################################################



  # Here what Id like to get is a returned numpy array containing local values 
  # of u evaluated at each coordinate.
  # Is there
  # any good method to this?

    # Objective: evaluate function at each coordinate, provided a coordinate list
    # Do we have a coordinate list?


#
#
#    print('main::size of mesh coords:', len(mesh_coords))
#
#    start = time.time()
#    # tabulate coordinates of dofs on this proc
#    x = V.tabulate_dof_coordinates()
#    
#    print('main::size of fspace:', V.dim())
#    print('main::size of dofmap:', len(x))
#
#    # get local version of expression
#    u_arr = u.vector().get_local()
#
#    print('main::size of expression:', len(u_arr))
#
#    # what is this???
#    # dof coordinates == mesh vertex coordinates ?
#    dof_vertex_indices = np.flatnonzero(np.isin(V.tabulate_dof_coordinates()
#                                                    ,mesh_coords).all(axis=1))
#
#    print('main::size of dof vertex idx:', len(dof_vertex_indices))
#
#    # create local vector
#    vals = np.zeros(len(dof_vertex_indices))
#
#    # fill local vector with expression
#    for i,j  in enumerate(dof_vertex_indices):
#        vals[i] = u_arr[j]
#    end = time.time()
#
#    print("Time dof coords: {0:.2f}".format(end-start))
#
#    # Use same ordering of coordinates for all methods
#    coords = x[dof_vertex_indices]
#    
#    start = time.time()
#    outputs_2 = [u(coord) for coord in coords]
#    end = time.time()
#    print('Time list comprehension:', end - start)
#
#    start = time.time()
#    outputs_3 = list()
#    for coord in coords:
#        outputs_3.append(u(coord))
#    print('Time loop:', time.time() - start)
#
#    print(np.linalg.norm( np.array(outputs_2)-vals))
#    print(np.linalg.norm(np.array(outputs_2)-np.array(outputs_3)))
#
#
#
#
#
#
#
#
#
#
#
#
