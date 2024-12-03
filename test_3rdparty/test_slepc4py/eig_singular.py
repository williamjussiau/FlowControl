### Generalized eigenvalue problem with singular RHS ###

import time
import numpy as np
import scipy as scp
import scipy.linalg as la
import scipy.sparse as spr
import scipy.sparse.linalg as spr_la
import petsc4py
from petsc4py import PETSc
from slepc4py import SLEPc
import pdb

def dense_to_sparse(A, eliminate_zeros=True):
    '''Cast PETSc or dolfin Matrix to scipy.sparse
    (Misleading name because A is not dense)'''
    import scipy.sparse as spr
    from petsc4py import PETSc

    if spr.issparse(A):
        #print('A is already sparse; exiting')
        if eliminate_zeros:
            A.eliminate_zeros()
        return A

    if isinstance(A, np.ndarray):
        A = spr.csr_matrix(A)
        if eliminate_zeros:
            A.eliminate_zeros()
        return A
    
    if not isinstance(A, PETSc.Mat):
        A = as_backend_type(A).mat()
    
    Ac, As, Ar = A.getValuesCSR()
    Acsr = spr.csr_matrix((Ar, As, Ac))
    if eliminate_zeros:
        Acsr.eliminate_zeros()
    return Acsr


def sparse_to_petscmat(A):
    '''Cast scipy.sparse matrix A to PETSc.Matrix()
    A should be scipy.sparse.xxx and square'''
    A = A.tocsr() 
    Acsr = (A.indptr, A.indices, A.data)
    from petsc4py import PETSc
    Amat = PETSc.Mat()
    Amat.createAIJWithArrays(size=A.shape, csr=Acsr)#, comm=MPI.comm_world)
    Amat.assemblyBegin()
    Amat.assemblyEnd()
    return Amat


def array_to_petscmat(A, eliminate=True, eliminate_zeros=True):
    '''Cast np array A to PETSc.Matrix()
    Warning: A should be of reasonable size to be stored as dense (in numpy)'''
    return sparse_to_petscmat(dense_to_sparse(A, eliminate_zeros))


def get_mat_vp_slepc(A, B=None, n=2, DEBUG=False, target=None, 
                     return_eigensolver=False, verbose=False,
                     eps_type=SLEPc.EPS.Type.KRYLOVSCHUR,
                     precond_type=PETSc.PC.Type.NONE,
                     ksp_type=PETSc.KSP.Type.GMRES):
    '''Get n eigenvales of matrix A
    Possibly with generalized problem (A, B).
    Direct SLEPc backend.
    A::dolfin.cpp.la.PETScMatrix'''
    import petsc4py
    from petsc4py import PETSc
    from slepc4py import SLEPc
    
    #if not isinstance(A, petsc4py.PETSc.Mat):
    #    A = A.mat()
    #if not isinstance(B, petsc4py.PETSc.Mat) and B is not None:
    #    B = B.mat()
    
    # Create problem
    eigensolver = SLEPc.EPS().create()
    eigensolver.setType(eps_type) # default is krylovschur
    #eigensolver.setType(eigensolver.Type.POWER) # default is krylovschur
    #eigensolver.setType(eigensolver.Type.ARNOLDI) # default is krylovschur

    # Problem types:
    #   HEP = Hermitian Eigenvalue Problem
    #   NHEP = Non-Hermitian EP
    #   GHEP = Generalized HEP
    #   GHIEP = Generalized Hermitian-indefinite EP
    #   GNHEP = Generalized NHEP
    #   PGNHEP = Positive semi-definite B + GNHEP
    # Our problem is most likely not hermitian at all
    # Should be PGNHEP though
    if B is None:
        pbtype = eigensolver.ProblemType.NHEP
    else:
        pbtype = eigensolver.ProblemType.GNHEP
    eigensolver.setProblemType(pbtype) 

    # Spectral transforms:
    #   SHIFT = shift of origin: inv(B)*A + sigma*I
    #   SINVERT = shift and invert: inv(A-sigma*B)*B
    #   CAYLEY = Cayley: inv(A-sigma*B)*(A+v*B)
    #   PRECOND = preconditioner: inv(K)=approx inv(A-sigma*B)
    # see https://slepc.upv.es/handson/handson3.html
    st = SLEPc.ST().create()
    st.setType(SLEPc.ST.Type.SINVERT)
    if target is not None:
        st.setShift(target)
    # we might have to set ST.KSP
    ksp = PETSc.KSP().create()
    ksp.setType(ksp.Type.GMRES)  # MINRES GMRES CGS CGNE
    #ksp.setGMRESRestart(1000)

    pc = PETSc.PC().create()
    pc.setType(precond_type) # NONE works
    # FIELDSPLIT? need block matrix
    #pc.setFieldSplitType(pc.SchurFactType.UPPER) 
    ksp.setPC(pc)
    
    #ksp set null space?

    st.setKSP(ksp)
    # yes: this solves problems related to LU first, but then we have problems with ILU precond
    # precond: solved by setting pc

    eigensolver.setST(st)

    # Target
    if target is not None:
        eigensolver.setTarget(target)
        eigensolver.setWhichEigenpairs(eigensolver.Which.TARGET_REAL)

    # Tolerances
    eigensolver.setTolerances(1e-3, 1000)
    eigensolver.setOperators(A, B)
    eigensolver.setDimensions(n, mpd=PETSc.DECIDE)
    
    if verbose:
        def monitor(eps, its, nconv, eig, err):
            # eps = SLEPc.EPS
            print('--- monitor ---')
            print('nit: ', its)
            print('nconv: ', nconv)
            #print('eig: ', eig)
            #print('err: ', err)
        eigensolver.setMonitor(monitor)


    if DEBUG:
        try:
            eigensolver.solve()
        except:
            print('was error')
            return eigensolver
    else:
        eigensolver.solve()

    sz = A.size[0] 
    
    nconv =  eigensolver.getConverged()
    niters = eigensolver.getIterationNumber()
    
    n = nconv
    valp = np.zeros(n, dtype=complex)
    vecp = np.zeros((sz, n), dtype=complex)
    vecp_re = np.zeros((sz, n), dtype=float)
    vecp_im = np.zeros((sz, n), dtype=float)

    vr = A.createVecRight()
    vi = A.createVecRight()

    for i in range(nconv):
        valp[i] = eigensolver.getEigenpair(i, Vr=vr, Vi=vi)
        vecp_re[:, i] = vr.array
        vecp_im[:, i] = vi.array
        if verbose:
            print('Eigenvalue %2d is: %f+i %f' % (i+1, np.real(valp[i]), np.imag(valp[i])))

    vecp = vecp_re + 1j*vecp_im

    ans = (valp, vecp)
    if return_eigensolver or DEBUG:
        ans = (ans, eigensolver)

    return ans



def make_mat_to_test_slepc(view=False, singular=False, neigpairs=3, density_B=1.0, rand=False):
    sz = 2*neigpairs
    if neigpairs==3:
        # known problem
        re = [0.2, -0.5, -0.7] # real part of eigenval
        im = [0.25, 0.8,  0.0] # imag part of eigenval
    else:
        if rand:
            re = np.random.randn(neigpairs) # real part of eigenval
            im = np.random.randn(neigpairs) # imag part of eigenval
        else:
            eigstep = 0.1 
            re=np.arange(start=-neigpairs*eigstep, stop=0, step=eigstep)
            im=np.arange(start=-neigpairs*eigstep, stop=0, step=eigstep)
            # eigenvalues from -10 to -1 (both re and im)
            #re = np.linspace(-10, -1, neigpairs)
            #im = np.linspace(-10, -1, neigpairs)

    mka = lambda a, b: [[a, -b], [b, a]] # make 2x2 matrix with given conj eigenval

    # slow because of reallocation
    #a = mka(re[0], im[0])
    #for i in range(1, neigpairs): # make block matrix with a+1jb as eigenval
    #    a = la.block_diag(a, mka(re[i], im[i])) 
    # hopefully faster
    A = np.zeros((sz, sz))
    for i in range(neigpairs):
        j = 2*i
        A[j:j+2, j:j+2] = mka(re[i], im[i]) # i+2 excluded

    # make b in numpy to set values easily, then convert
    B = np.eye(sz)
    if singular: # makes b non invertible
        if neigpairs==3:
            B[1, 1] = 0
            B[2, 2] = 0 
        else:
            # number of zeros in B
            nrzeros = int(sz * (1-density_B))
            if rand:
                # where zeros are located
                wherezeros = np.random.permutation(sz)[:nrzeros]
            else:
                wherezeros = np.arange(nrzeros)
            B[wherezeros, wherezeros] = 0

    if view:
        return A, B
    else:
        return array_to_petscmat(A), array_to_petscmat(B)


def load_mat_from_file(mat_type='fem'):
    if mat_type=='fem':
        Afile = 'A_sparse.npz'
        Qfile = 'Q_sparse.npz'
    else:
        Afile = 'A_rand.npz'
        Qfile = 'B_rand.npz'
    return spr.load_npz(Afile), spr.load_npz(Qfile)


def geig_singular(A, B, n=2, DEBUG=False, target=None, solve_dense=False):
    '''Get n eigenvales of generalized problem (A, B)
    Where B is a singular matrix (hence no inv(B))
    Direct SLEPc backend.'''
   
    # Setup problem
    # works with classic matrices for now, not PETSc
    s = 1.2
    nn = A.shape[0]
    sznp = nn-np.linalg.matrix_rank(B)
    C = np.ones((nn-sznp, nn-sznp))

    At, Bt = augment_matrices(A, B, s, C)
    #print('matrices A, B are: \n', At, '\n', Bt)

    if solve_dense:
        Dt, Vt = la.eig(At, Bt)
    else:
        At = dense_to_sparse(At)
        Bt = dense_to_sparse(Bt)
        if target is None:
            target = 0.0
        Asb = At - target*Bt
        Asb = Asb.tocsc()
        LU = spr_la.splu(Asb)
        OPinv = spr_la.LinearOperator(matvec=lambda x: LU.solve(x), shape=Asb.shape)
        OPinv = spr_la.LinearOperator(matvec=lambda x: spr_la.minres(Asb, x, tol=1e-5)[0], shape=Asb.shape) 
        Dt, Vt = spr_la.eigs(A=At, k=n, M=Bt, tol=0, sigma=target, OPinv=OPinv)
        print('Embedded sparse eig: ', Dt)

    #pdb.set_trace()
    
    Vt_zero = Vt[-sznp:, :]
    EPS = np.finfo(float).eps
    idx_true_eig = np.all(np.abs(Vt_zero) < EPS*1e3, axis=0) 
    
    true_eigval = Dt[idx_true_eig]
    true_eigvec = Vt[:-sznp, idx_true_eig]

    # Then solve
    # maybe with get_mat_vp***
    return true_eigval, true_eigvec


def augment_matrices(A, B, s, C):
    '''Augment matrices to make eigenproblem nonsingular
    Inputs are dense matrices [A, B, C] and scalar [s]'''
    N = la.null_space(B)
    M = la.null_space(B.T).T
    sznp = N.shape[1]

    s = 1.2
    MA = M @ A
    AN = A @ N
    C = np.ones((MA.shape[0], AN.shape[1]))

    Aa = np.block([[A, s*AN], [MA, s*C]])
    Ba = np.block([[B, AN], [MA, C]])

    return Aa, Ba


def get_all_enum_petsc_slepc(enum):
    '''Get every possible value of enum in PETSc/SLEPc subpackage
    e.g. PETSc.KSP.Type, PETSc.PC.Type... ''' 
    allenum = enum.__dict__.copy()
    keys_to_pop = []
    for key, value in allenum.items():
        if key[0]=='_':
            keys_to_pop.append(key)
    for key in keys_to_pop:
        allenum.pop(key)
    return allenum


# Run
if __name__=='__main__':
    ##Av, Bv = make_mat_to_test_slepc(view=True, singular=False)
    ##A = dense_to_sparse(Av)
    ##B = dense_to_sparse(Bv)

    ##k = 4
    ##D = spr_la.eigs(A=A, k=k, M=B, sigma=None, Minv=None, OPinv=None, return_eigenvectors=False) 

    ##print('Eigenvalues are:')
    ##print(D)
    ###print('Eigenvectors are:')
    ###print(V)

    ##k=6
    ##Ad, Bd = make_mat_to_test_slepc(view=True, singular=True)

    ##print('----------------------------------- begin embedded fn')
    ##solve_dense = True
    ##sigma = 0.0
    ##eva, eve = geig_singular(Ad, Bd, n=k, solve_dense=solve_dense, target=sigma)
    ##print('solving with dense version: ', solve_dense)
    ##print('eigenvalues: \n', eva)
    ###print(eve)
    ##
    ##Z = Ad@eve - Bd@eve@np.diag(eva)
    ###print('residual: \n', Z)
    ##print('----------------------------------- end')
   


    #print('----- converting to sparse ----- begin')

    ##n = 4 # size of problem
    ##re = [0.2, -0.5] # real part of eigenval
    ##im = [0.25, 0.8] # imag part of eigenval
    ##mka = lambda a, b: [[a, -b], [b, a]] # make 2x2 matrix with given conj eigenval
    ##a = mka(re[0], im[0])
    ##for i in range(1, n//2): # make block matrix with a+1jb as eigenval
    ##    a = la.block_diag(a, mka(re[i], im[i])) 
    ### make b in numpy to set values easily, then convert
    ##b = np.eye(n)
    ##singular = True
    ##if singular: # makes b non invertible
    ##    b[1, 1] = 0
    ##    b[2, 2] = 0 
    ##Ad = a
    ##Bd = b

    #As = dense_to_sparse(Ad)
    #Bs = dense_to_sparse(Bd)
    #
    #sigma = 0.
    #II = spr.eye(As.shape[0])

    #rhs = Bs # II

    #Asb = As - sigma*rhs
    #Asb = Asb.tocsc()
    #LU = spr_la.splu(Asb)

    ## should not be specified if sigma is not None
    ## Minv = spr_la.LinearOperator(matvec=lambda x: spr_la.minres(Bs, x, tol=1e-5)[0], shape=Bs.shape)
    #
    #OPinv = spr_la.LinearOperator(matvec=lambda x: LU.solve(x), shape=Asb.shape)     
    ###OPinv = spr_la.LinearOperator(matvec=lambda x: spr_la.minres(Asb, x, tol=1e-5)[0], shape=Asb.shape) 

    ##############Ds, Vs = spr_la.eigs(A=spr_la.aslinearoperator(As), k=k, M=Bs, sigma=sigma, which='LM', OPinv=OPinv)

    ### try: M = None, OPinv = A_as_LU.solve(B @ x)
    ##ALU = spr_la.splu(As.tocsc()) 
    ### this solves: Ax=Bb with virtually: inv(B)*Ax=b
    ##OPinv_AB = spr_la.LinearOperator(matvec=lambda x: ALU.solve(rhs@x), shape=Asb.shape)     
    ##Ds, Vs = spr_la.eigs(A=As, k=k, M=None, sigma=sigma, OPinv=OPinv_AB, which='LM')
    #
    ##Ds_shifted = Ds - sigma

    ##print('eigenvalues: \n', Ds)
    ##print('eigenvalues (shifted back): \n', Ds_shifted)
    ##print('eigenvectors: \n', Vs)

    #print('----- scipy.sparse ----- end')

    
    print('----- using slepc ----- begin')

    print('...............................')
    print('Small matrix')
    nit = 1
    dt = 0
    for i in range(nit):
        t0 = time.time()
        Ad, Bd = make_mat_to_test_slepc(view=True, singular=True)
        A = array_to_petscmat(Ad)
        B = array_to_petscmat(Bd)
        eigz, eigensolver = get_mat_vp_slepc(A, B, target=0.1, n=4, DEBUG=False, verbose=False,
            return_eigensolver=True)
        dt += time.time() - t0
    print('Elapsed (avg on %d iter): %f' %(nit, dt/nit))
    print(eigz[0][:,None])
    print('...............................')


    # Test routines with random sparse matrices
    if 1:
        print('')
        tc = time.time()
        AA, BB = make_mat_to_test_slepc(view=True, singular=True, 
            neigpairs=50, density_B=7/10, rand=False)
        #AA, BB = load_mat_from_file('rand')
        print('...............................')
        print('Random matrices of size: %i' %(BB.shape[0]))
        print('Create - elapsed: %f' %(time.time() - tc))
        
        
        # BENCHMARK
        if 0:
            print('...............................')
            print('Benchmark')
            # Get all preconditioners available
            all_pc = get_all_enum_petsc_slepc(PETSc.PC.Type)
            pc_except = ['MAT','ICC', 'NONE', 'DEFLATION']
            for pc in pc_except:
                all_pc.pop(pc)
            # We should do: for all pair (solver, pc), try eig solve, log time elapsed 
            pc_time = all_pc.copy()
            for pc_key, pc_name in all_pc.items():
                try:
                    t0 = time.time()
                    eigz, eigensolver = get_mat_vp_slepc(array_to_petscmat(AA), array_to_petscmat(BB),
                        target=-0.1, n=4, return_eigensolver=True, precond_type=pc_name, verbose=False)
                    tpc = time.time() - t0
                    print('PC success: %s --- %f' %(pc_name, tpc))
                except:
                    print('PC failed:  %s ' %(pc_name))
                    tpc = np.inf
                pc_time[pc_key] = tpc


            # Get all solvers available
            all_solvers = get_all_enum_petsc_slepc(SLEPc.EPS.Type)
            solvers_except = ['POWER', 'SUBSPACE', 'ARNOLDI']
            for solver in solvers_except:
                all_solvers.pop(solver)
            solver_time = all_solvers.copy()
            for solver_key, solver_name in all_solvers.items():
                try:
                    t0 = time.time()
                    eigz, eigensolver = get_mat_vp_slepc(array_to_petscmat(AA), array_to_petscmat(BB),
                        target=-0.1, n=4, return_eigensolver=True, precond_type=PETSc.PC.Type.NONE, verbose=False,
                        eps_type=solver_name)
                    tsl = time.time() - t0
                    print('Solver success: %s --- %f' %(solver_name, tsl))
                except:
                    print('Solver failed: %s' %(solver_name))
                    tsl = np.inf
                solver_time[solver_key] = tsl
            print('...............................')
        
        # BEST
        # Try best + best
        print('...............................')
        print('Best + best')
        t0b = time.time()
        eigz, eigensolver = get_mat_vp_slepc(array_to_petscmat(AA), array_to_petscmat(BB),
                target=-2.103, n=4, return_eigensolver=True,  verbose=False,
                precond_type=PETSc.PC.Type.HYPRE,
                eps_type=SLEPc.EPS.Type.KRYLOVSCHUR)
        print('Best + best: %f ' %(time.time() - t0b) )
        Z = AA@eigz[1] - BB@eigz[1]@np.diag(eigz[0])
        print('Sanity check: ', np.linalg.norm(Z))
        print('...............................')
    
    # True FEM matrices
    if 1:
        print('...............................')
        print('Start with FEM matrices')
        t0 = time.time()
        AA, BB = load_mat_from_file()
        # make empty entries -> 0
        #AA.setdiag(AA.diagonal())
        #BB.setdiag(BB.diagonal())
        # solve
        eigz, eigensolver = get_mat_vp_slepc(
            array_to_petscmat(AA, eliminate_zeros=True), 
            array_to_petscmat(BB, eliminate_zeros=True),
            target=-0.2, 
            n=2, 
            return_eigensolver=True, 
            verbose=True,
            precond_type=PETSc.PC.Type.NONE, 
            eps_type=SLEPc.EPS.Type.KRYLOVSCHUR)
        print('Big matrix - elapsed: %f' %(time.time() - t0))
        print('...............................')
   


    print('...............................')
    print('Summary .......................')
    st = eigensolver.getST()
    ksp = st.getKSP()
    pc = ksp.getPC()

    Print = PETSc.Sys.Print
    its = eigensolver.getIterationNumber()
    Print( "Number of iterations of the method: %d" % its )
 
    eps_type = eigensolver.getType()
    Print( "Solution method: %s" % eps_type )
 
    nev, ncv, mpd = eigensolver.getDimensions()
    Print( "Number of requested eigenvalues: %d" % nev )
 
    tol, maxit = eigensolver.getTolerances()
    Print( "Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit) )
 
    nconv = eigensolver.getConverged()
    Print( "Number of converged eigenpairs %d" % nconv )
     
    Print("Overview of eigenvalues: \n ", eigz[0][:,None])




