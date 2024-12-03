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
 

    @staticmethod
    def peval2(f, x):
        '''Parallel synced eval, v2'''
        comm = f.function_space().mesh().mpi_comm()
        if comm.size == 1:
            return f(*x)
        # Find whether the point lies on the partition of the mesh local
        # to this process, and evaluate u(x)
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
 

if __name__ == '__main__':
    comm = MPI.comm_world
    size = comm.Get_size()
    rank = comm.Get_rank()
    MpiUtils.check_process_rank()
    print('Pool size:', size)

    sendbuf = np.zeros(100, dtype='i') + rank
#    print('Local array:', sendbuf)
    recvbuf = None
    if rank == 0:
        recvbuf = np.empty([size, 100], dtype='i')
    
    comm.Gather(sendbuf, recvbuf, root=0)

    if rank == 0:
        for i in range(size):
            assert np.allclose(recvbuf[i,:], i)
#        print('Gathered array:', recvbuf)


    print('-'*80)
    n3 = 2
    mesh = UnitSquareMesh(n3, n3)
    V = FunctionSpace(mesh, 'CG', 2)
    dofmap = V.dofmap()
    
    uexp = Expression('1*(sin(x[0])+x[1]*x[1])', degree=2)
    u0 = project(uexp, V)

    fspace = V
    ndof = fspace.dim()
    u = Function(fspace)
    u_vec = u.vector()
    xs = [0.2, 0.4]
    measurement_fun = lambda u: MpiUtils.peval(u, xs) 

    dofs_local = dofmap.dofs()
    idof_old = 0
    print('Local dofmap is:', dofmap.dofs())
    print('Size of u is:', len(u_vec.get_local()))
    ndof_local = len(dofs_local)
    C_local_red = np.zeros(ndof)
    C_local_gat = np.zeros(ndof_local)

    for i, idof in enumerate(dofs_local):
        #print(u_vec.get_local())
       # ei = np.zeros(ndof_local, dtype=float)
       # ei[i] = 1.0
       # u_vec.set_local(ei)

        #u_vec_global = comm.allgather(u_vec.get_local())
        #print(u_vec.get_local())

        C_local_gat[i] = idof
        C_local_red[idof] = idof #1*measurement_fun(u)
       
       # print('in loop', measurement_fun(u0))

    #ufun_global = comm.allgather(u_vec.get_local())
    #print('ufun global is:', ufun_global)
    #comm.Reduce(u_vec.get_local(), uvec_global, root=0)
    #print('u vector global is:', uvec_global)
    
    # Reduce results (with sum)
    C_red = comm.reduce(C_local_red, root=0)
    print('my local C to be reduced is:', C_local_red)
    print('C reduced is:', C_red)

    # Gather results
   # C_gat = comm.gather(C_local_gat, root=0)
   # print('C gathered is:', C_gat)
    
    if rank == 0:
        print('In process 0 (root) **********************' )
    #    print('Gathered array on root:', C_gat)
        print('Reduced array on root:', C_red)
   
    C = C_red
    print('True measurement: ', measurement_fun(u0))
    if rank == 0:
        print( '\t with C: ', C @ u0.vector().gather_on_zero() )
    
    
    print('Finished.') 
    print('')

##    n3 = 10
##    mesh = UnitSquareMesh(n3, n3)
##    V = FunctionSpace(mesh, 'CG', 2)
##    dofmap = V.dofmap()
##    
##    uexp = Expression('1*(sin(x[0])+x[1]*x[1])', degree=2)
##    u0 = interpolate(uexp, V)
##
##    fspace = V
##    ndof = fspace.dim()
##    u = Function(fspace)
##    u_vec = u.vector()
##    xs = [0.2, 0.4]
##    measurement_fun = lambda u: u(xs) 
##
##    C = np.zeros(ndof)
##    idof_old = 0
##    for idof in dofmap.dofs():
##        u_vec[idof] = 1
##        if idof_old>0:
##            u_vec[idof_old] = 0
##        idof_old = idof
##        mu = measurement_fun(u)
##        C[idof] = mu 
##
##
##    print('True measurement: ', measurement_fun(u0))
##    print('\t with C: ', C @ u0.vector().get_local())
    
#
#
#
