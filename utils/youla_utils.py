# -*- coding: utf-8 -*-
"""
Utilitary function for controllers & Youla parametrization
"""

import numpy as np
import control
import control.matlab as cmat
import scipy.io as sio
import scipy.linalg as la
#import scipy as scp
import scipy.signal as sig
import warnings
#import pdb
#try:
#    import matlab.engine
#    eng = matlab.engine.start_matlab()
#    eng.addpath(r'/scratchm/wjussiau/fenics-python/cavity/matlab/', nargout=0)
#except OSError: #ModuleNotFoundError:
#    print('Matlab not installed -- skipping import')


def youla(G, K0, Q):
    '''Return controller K parametrized with Q using Youla formula
    G = plant to be stabilized
    K0 stabilizes G initially (can be 0)
    Q = Youla parameter
    Feedback (+)'''
    Gstab = G.feedback(other=K0, sign=+1)
    #Psi = [1, 0; 1, -Gstab]
    Psi = build_block_Psi(Gstab)

    Kq = Psi.lft(Q)
    K = K0 + Kq
    return K


def build_block_Psi(G):
    '''Build block function Psi=[1, 0; 1, -G]
    If G is SIMO: Psi=[zeros(1,ny), 1; eye(ny), -G]
    '''
    ny = G.noutputs

    ## By hand
    ## Psi.A = G.A
    #PsiA = G.A
    ## Psi.B = [0, 0... B]
    #PsiB = np.hstack((np.zeros((G.B.shape[0], ny)), G.B)) 
    ## Psi.C = [0; C]
    #PsiC = np.vstack((np.zeros((1, G.C.shape[1])), -G.C)) 
    ## Psi.D = [0, 0..., 1; In, D]
    #PsiD = np.block([[np.zeros((1, ny)), np.array(1.0)], [np.eye(ny), -G.D]]) 
    #Psi = control.StateSpace(PsiA, PsiB, PsiC, PsiD)

    ## With stacking functions
    O1 = control.StateSpace([], [], [], 1) # 1 ss 
    Z1 = control.StateSpace([], [], [], np.zeros((1, ny))) # 0 ss
    E1 = control.StateSpace([], [], [], np.eye(ny)) # eye ss
    Psi = ss_vstack(ss_hstack(Z1, O1), ss_hstack(E1, -G))

    return Psi


def basis_laguerre_canonical(p, N):
    '''Return first N vectors of Laguerre basis with pole p
    Basis is: phi_i(s) = sqrt(2p) * (s-p)^i-1 / (s+p)^i with p>0
    Phi(s) = [phi_1(s),... phi_N(s)]'''
    s = control.TransferFunction([1, 0],[1])
    Phi = np.zeros((N,), dtype=control.TransferFunction)
    L_i = 1/(s+p)
    for i in range(N):
        Phi[i] = L_i
        L_i = L_i * (s-p)/(s+p)
    Phi = np.sqrt(2*p)*Phi
    return Phi


def basis_laguerre(p, theta):
    '''Parametrization of controller using strictly proper Laguerre basis
    Basis is: phi_i(s) = sqrt(2p) * (s-p)^i-1 / (s+p)^i with p>0
    Returned controller is Q(s) = sum_i(theta_i * phi_i(s))'''
    if type(theta) is int:
        N = 1
        warnings.warn('theta should be iterable, not int')
    else:
        N = len(theta)
    Phi = basis_laguerre_canonical(p, N)
    Q = sum(Phi * theta)
    return Q


def basis_laguerre_canonical_ss(p, N):
    '''Return first N elements of Laguerre basis with pole p>0'''
    a = p
    a_vec = np.hstack((-a, 2*a*(-1)**(np.arange(2, N+1))))
    #a_vec = np.hstack((-a, 2*a*(-1)**(np.arange(1, N))))
    a2 = np.triu(la.circulant(a_vec).T) # transpose to fit Matlab
    #b2 = np.eye(N) * (-1)**(np.arange(2,N+2)) 
    b2 = np.diag( (-1)**(np.arange(2,N+2)) )
    c2 = np.sqrt(2*a) * (-1)**(np.arange(2, N+2))
    d2 = np.zeros((1,N))
    Phi = control.StateSpace(a2, b2, c2, d2)
    return Phi


def basis_laguerre_ss(p, theta):
    '''Compute Q=sum(theta_i phi_i(s; p)) with phi the Laguerre basis'''
    if type(theta) is int:
        N = 1
        warnings.warn('theta should be iterable, not int')
    else:
        N = len(theta)
    Phi = basis_laguerre_canonical_ss(p, N)
    Q = Phi * np.atleast_2d(np.array(theta)).T
    return Q


def ssmult(G1, G2):
    '''Multiplication of MIMO SS: G = G1 x G2
    Such that y = (G1 x G2)u
    OBSOLETE: exists and works in control toolbox'''
    ZERO = np.zeros((G2.A.shape[0], G1.A.shape[1]))
    A = np.block([ [G1.A, G1.B@G2.C],
                   [ZERO, G2.A]])
    B = np.vstack((G1.B @ G2.D, G2.B))
    C = np.hstack((G1.C, G1.D @ G2.C))
    D = G1.D @ G2.D 
    return control.StateSpace(A, B, C, D)


def youla_laguerre(G, K0, p, theta, verbose=False):
    '''Compute Youla controller with respect to plant Ghat=feedback(G, K0)
    Youla parameter Q is projected onto the Laguerre basis and defined
    through the projection coefficients theta, such that Q=Theta^T*Phi(s)
    SISO ONLY
    For MIMO: use shortcut youla_laguerre_mimo'''
    ##############################################################
    ############ Bugs seem to have been   ########################
    ############ repaired                 ########################
    ##############################################################
    Gstab = G.feedback(other=K0, sign=+1)
    #Psi = [1, 0; 1, -Gstab]

    Psi = build_block_Psi(Gstab)

    # Build canonical Laguerre basis in SS
    if type(theta) is int:
        N = 1
        warnings.warn('theta should be iterable, not int')
    else:
        N = len(theta)
    a = p
    a_vec = np.hstack((-a, 2*a*(-1)**(np.arange(2, N+1))))
    a2 = np.triu(la.circulant(a_vec))
    b2 = np.eye(N)
    c2 = np.sqrt(2*a) * (-1)**(np.arange(2, N+2))
    d2 = np.zeros((1,N))
    Qf = control.StateSpace(a2, b2, c2, d2)
    
    #SS1 = control.StateSpace([], [], [], [1])
    Qf1 = cmat.append(1, Qf)
    # idk why but this seems better
    #Psif = ssmult(Psi, Qf1)
    Psif = Psi * Qf1
    ##############################################################
    
    # LFT
    N = len(theta)
    # either alternate sign of theta, or do it on B
    theta = theta * (-1)**(np.arange(N))
    ss_theta = control.StateSpace([], [], [], np.array([theta]).T)
    
    Kq = Psif.lft(ss_theta)
    K = K0 + Kq
    if verbose:
        print('\t Feedback(G, Ky, +1) is stable: ', isstablecl(G, K, +1)) 
    return K


def youla_laguerre_mimo(G, K0, p, theta, verbose=False):
    '''Youla in ~MIMO (SIMO plant G, MISO controller K)
    Could be extended to full MIMO plants
    p=[p1, p2, p3...] = np.array or list
    theta=[[theta11, theta12...], [theta21, theta22...]...] = np.array
     with Qi on each row
    '''
    nout = G.noutputs
    Q = basis_laguerre_ss(p=p[0], theta=theta[0,:])
    for i in range(1, nout): # from second output to last, stack
        Qi = basis_laguerre_ss(p=p[i], theta=theta[i,:])
        Q = ss_hstack(Q, Qi)
    K = youla(G, K0, Q)

    if verbose:
        print('\t Feedback(G, Ky, +1) is stable: ', isstablecl(G, K, +1)) 
    return K


def youla_laguerre_K00(G, K0, p, theta, check=False):
    '''Compute Youla regulator stabilizing G using base controller K0
    and Laguerre basis of transfer functions
    This function ensures that the resulting controller has K(0)=0
    SISO ONLY'''
    # The base formula is: K = K0 + Kq
    # where Kq = Q*inv(I+Gstab*Q), Gstab = feedback(G, K0, +1)
    # We want to ensure K(0) = 0, that is Kq(0) = -K0(0)
    # By the inversion theorem: Q(0) = -inv(I+K0(0)*Gstab(0))*K0(0)
    # We denote this Q(0) = beta0
    # Then we know Q(s) = sum(theta_i * phi_i(s))
    # So Q(0) = sqrt(2p)/p * sum(theta_i * (-1)^(0:n-1))
    # The condition is then: sum(...) = beta0*p/sqrt(2p) = alpha0
    # We denote the sum: J*Theta, with J = [1, -1, 1, -1...]
    # A solution Theta to J*Theta=alpha0 is Theta0 = J\alpha0 (pinv(J)*alpha0)
    # Every solution is written: Theta = Theta0 + ker(J) * y
    # Where y lies in R^{N-1}

    #N = len(theta)
    ## N = nparam

    ## Computing beta0
    #K00 = control.dcgain(K0)
    #Gstab = control.feedback(G, K0, +1)
    #G00 = control.dcgain(Gstab)
    #b0 = -K00 / (1+K00*G00)

    ##import pdb
    ##pdb.set_trace()

    ## Computing alpha0
    #a0 = b0 * np.sqrt(p/2)

    ## Computing all solutions
    ## J = [1, -1, 1...]
    #J = np.atleast_2d(np.ones((N+1,))) * (-1)**np.arange(0, N+1)
    ## equivalent to pinv(J) * a0
    #y0 = la.lstsq(J, [a0])[0]
    ## ker J
    #kerJ = la.null_space(J)
    ## all solutions @ theta 
    #y = y0 + kerJ @ theta

    ## Result
    ##K = youla(G=G, K0=K0, Q=basis_laguerre_ss(p=p, theta=y))
    Q00 = basis_laguerre_K00(G, K0, p, theta) 
    K = youla(G=G, K0=K0, Q=Q00)
    
    # Check
    if check:
        dcK = control.dcgain(K) # should be ~0
        print('DC gain of K (should be 0): ', dcK)

    return K


def basis_laguerre_K00(G, K0, p, theta):
    '''Compute Youla param Q00 ensuring K(0)=0 when plugging
    Q00 in classic Youla formulation with (G,K0)
    SISO ONLY'''
    # See youla_laguerre_K0 for details on the computation
    N = len(theta)

    # Computing beta0
    K00 = control.dcgain(K0)
    Gstab = control.feedback(G, K0, +1)
    G00 = control.dcgain(Gstab)
    b0 = -K00 / (1+K00*G00)

    # Computing alpha0
    a0 = b0 * np.sqrt(p/2)

    # Computing all solutions
    # J = [1, -1, 1...]
    J = np.atleast_2d(np.ones((N+1,)) * (-1)**np.arange(0, N+1))
    # equivalent to pinv(J) * a0

    #import pdb
    #pdb.set_trace()

    def lsq(A, b):
        return la.lstsq(A, b.reshape(-1))[0]
    y0 = lsq(J, a0)
    # ker J
    kerJ = la.null_space(J)
    # all solutions @ theta 
    y = y0 + kerJ @ theta
    # project theta on affine subspace with minimal norm
    #ybest = -lsq(kerJ, lsq(J, a0))

    # Result
    Q00 = basis_laguerre_ss(p=p, theta=y)

    return Q00    #, ybest


def youla_lqg(G, Qx, Ru, Qv, Rw, Q):
    '''Youla controller in LQG form'''
    # K0 = lft(J, 0)
    # Ky = lft(J, Q)
    J = youla_lqg_lftmat(G, Qx, Ru, Qv, Rw)
    return J.lft(Q)


def youla_lqg_lftmat(G, Qx, Ru, Qv, Rw):
    '''Utilitary function to Youla LQG parametrization
    Return StateSpace to be LFTed with Q'''
    #A = np.array(G.A) --> weird
    B = np.array(G.B)
    C = np.array(G.C)
    D = np.array(G.D)
    #n = A.shape[0]
    #In = np.eye(n)
    p, m = D.shape
    Im = np.eye(m)
    Ip = np.eye(p)
    # StateSpace to be LFTed with Q
    Klqg, F, L = lqg_regulator(G, Qx, Ru, Qv, Rw)
    J = control.StateSpace(Klqg.A,
        np.hstack((Klqg.B, B+L*D)), 
        np.vstack((Klqg.C, -C-D*F)), 
        np.block([[np.zeros((m,p)), Im], [Ip, Klqg.D]])) 
    return J


def lqg_regulator(G, Qx, Qu, Qw, Qv):
    '''Make LQG regulator with the following weights:
        Qxu = blkdiag(Qx*eye(n), Qu*eye(m))
        Qwv = blkdiag(Qw*eye(n), Qv*eye(p))
    For the state-feedback:
        Qx=weight on x^T*Qx*x
        Qu=weight on u^T*Qu*u
    For the observer: 
        Qw=cov(noise x)
        Qv=cov(noise y)
    Warning: returns 3 outputs: K, F, L
    '''
    A = np.array(G.A) 
    B = np.array(G.B)
    C = np.array(G.C)
    D = np.array(G.D)
    n = A.shape[0]
    In = np.eye(n)
    p, m = D.shape
    Im = np.eye(m)
    Ip = np.eye(p)
    # State feedback gain
    F = np.array(-control.lqr(A, B, Qx*In, Qu*Im)[0])
    # Estimator gain
    L = np.array(-control.lqr(A.T, C.T, Qw*In, Qv*Ip)[0]).T 
    # Assemble regulator
    Klqg = control.StateSpace(A+B*F+L*C+L*D*F, -L, F, 0)
    return Klqg, F, L


def hinfsyn_mref(G, We, Wu, Wb, Wr, CLref, Wcl, syn='Hinf'):
    '''Classic SISO Hinf synthesis (mixed-sensitivity) 
    with additional model reference for the closed-loop GS
    Warning: negative feedback (-)
    ------
    inputs:
        We, Wu, Wb, Wr: mixed-sensitivity weightings
        Wcl: model reference weighting
        CLref: reference GS to fit - warning: feedback (-)
        syn: 'Hinf' or 'H2' (reminder: gamma not respected in H2)
    ------
    outputs:
        K
        gamma
    ------
    Weightings act as follows:
    > mixsyn:
        We*S*Wr < gamma
        We*GS*Wb < gamma
        Wu*KS*Wr < gamma
        Wu*T*Wb < gamma
    > model reference:
        Wcl*T*Wr < gamma
        Wcl*(GS-CLref)*Wb < gamma
        
    '''
    if syn not in ('Hinf', 'H2'):
        raise ValueError('Only Hinf or H2 synthesis supported')

    # Zero and Identity StateSpace 
    Zo = ss_zero()
    Id = ss_one()
    
    # Augmented plant
    Wout = ss_blkdiag_list([We, Wu, Wcl, Id])
    Win = ss_blkdiag_list([Wr, Wb, Id])
    P_syn = ss_vstack(ss_hstack(Id, -Id,  Zo,  Zo), \
                      ss_hstack(Zo,  Zo,  Id,  Zo), \
                      ss_hstack(Zo,  Id,  Zo, -Id), \
                      ss_hstack(Id, -Id,  Zo,  Zo)) * \
            ss_blkdiag_list([Id, G, Id, CLref]) * \
            ss_vstack(ss_hstack(Id,  Zo,  Zo), \
                      ss_hstack(Zo,  Id,  Id), \
                      ss_hstack(Zo,  Zo,  Id), \
                      ss_hstack(Zo,  Id,  Zo))
    P_syn = Wout * P_syn * Win
    
    # Synthesis
    if syn=='Hinf':
        K, _, _, _ = control.hinfsyn(P_syn, 1, 1)
    if syn=='H2':
        K = control.h2syn(P_syn, 1, 1)

    # Compute gamma for Hinf syn
    FPK = P_syn.lft(K)
    gamma = norm(FPK)

    return K, gamma
    

def youla_laguerre_fromfile(theta, path):
    '''Reduced version of youla_laguerre(G, K0, p, theta)
    Psif = g(G, K0, p) is stored in a file at <path> and loaded
    Only the last LFT is performed here
    To make a complete function, see above -- we just need to write the SS 
    form of the Laguerre basis
    Warning: theta size should be coherent with saved Psif'''
    #N = len(theta)
    # Warning (bis): signs are switched here
    theta = theta * (-1)**(np.arange(len(theta)))
    
    # ss(theta) to be LFTed with Psif
    ss_theta = control.StateSpace([], [], [], np.array([theta]).T)
    #ss_theta = control.StateSpace(np.zeros((N,N)),
    #                              np.zeros((N,1)),
    #                              np.zeros((N,N)),
    #                              np.array([theta]).T)
    # Read matrices: K0, Psif
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Duplicate variable name*')
        rd = sio.loadmat(path)
    # Build ss(Psif)
    Psif = control.StateSpace(rd['Psi_A'], rd['Psi_B'], rd['Psi_C'], rd['Psi_D'])    
    # LFT(Psif, theta)
    Kq = Psif.lft(ss_theta)
    # Build K0
    K0 = control.StateSpace(rd['K0_A'], rd['K0_B'], rd['K0_C'], rd['K0_D'])
    # Build final controller
    K = K0 + Kq
    return K


def isstable(CL):
    '''Shortcut to assess stability of StateSpace CL (usually CL=feedback(G, K, +1))''' 
    poles = control.pole(CL) # retrieve poles
    return np.all(np.real(poles) <= 0) # assess Re(poles) <= 0


def isstablecl(G, K0, sign=+1):
    '''Shortcut to shortcut to assess stability of feedback (G, K0, sign)'''
    return isstable(control.feedback(G, K0, sign=sign))


def ss_vstack(sys1, *sysn):
    '''Equivalent of Matlab [sys1; sys2]
          --- sys1 ---> y1
     u --|  
          --- sys2 ---> y2
    A = [A1, 0; 0, A2]
    B = [B1; B2]
    C = [C1, 0; 0, C2]
    D = [D1; D2]
    Bad implementation with no pre-allocation, but working'''
    #ss_out = control.StateSpace(sys1.A, sys1.B, sys1.C, sys1.D) 
    A, B, C, D = ssdata(sys1)
    for sys in sysn:
        A = la.block_diag(A, sys.A)
        B = np.vstack((B, sys.B))
        C = la.block_diag(C, sys.C)
        D = np.vstack((D, sys.D))
    return control.StateSpace(A, B, C, D) 
    

def ss_hstack(sys1, *sysn):
    '''Equivalent of Matlab [sys1, sys2]
     u1 --- sys1 ---
                    |+---> y
     u2 --- sys2 ---
    A = [A1, 0; 0, A2]
    B = [B1, 0; 0, B2]
    C = [C1, C2]
    D = [D1, D2]
    Bad implementation with no pre-allocation, but working'''
    #ss_out = control.StateSpace(sys1.A, sys1.B, sys1.C, sys1.D) 
    A, B, C, D = ssdata(sys1)
    for sys in sysn:
        A = la.block_diag(A, sys.A)
        B = la.block_diag(B, sys.B)
        C = np.hstack((C, sys.C))
        D = np.hstack((D, sys.D))
    return control.StateSpace(A, B, C, D)


def ss_vstack_list(syslist):
    '''Same as ss_vstack but with input list'''
    A, B, C, D = control.ssdata(syslist[0])
    for sys in syslist[1:]:
        A = la.block_diag(A, sys.A)
        B = np.vstack((B, sys.B))
        C = la.block_diag(C, sys.C)
        D = np.vstack((D, sys.D))
    return control.StateSpace(A, B, C, D)


def ss_hstack_list(syslist):
    '''Same as ss_hstack but with input list'''
    A, B, C, D = control.ssdata(syslist[0])
    for sys in syslist[1:]:
        A = la.block_diag(A, sys.A)
        B = la.block_diag(B, sys.B)
        C = np.hstack((C, sys.C))
        D = np.hstack((D, sys.D))
    return control.StateSpace(A, B, C, D)


def ss_blkdiag_list(sys_list):
    '''Make block diagonal system with input list'''
    return control.append(*sys_list)


def ssdata(sys):
    '''Return StateSpace matrices as np.array instead of np.matrix'''
    A, B, C, D = control.ssdata(sys)
    # changed 03/08/2023: np.array() to np.asarray()
    return np.asarray(A), np.asarray(B), np.asarray(C), np.asarray(D) 
    

def ss_zero():
    '''Return null SISO StateSpace'''
    arr0 = np.array([])
    # or control.tf2ss(control.tf(0, 1))
    return control.StateSpace(arr0, arr0, arr0, 0) 


def ss_one():
    '''Return identity SISO StateSpace'''
    return control.tf2ss(control.tf(1, 1))


def show_ss(sys):
    '''Easy display of StateSpace in command window'''
    for mat in ssdata(sys):
        print(mat)
        print('-'*10)


def ss_inv(G):
    '''Invert StateSpace G provided that G.D is not 0'''
    gD = np.array(G.D) 
    if np.linalg.norm(gD) <= 1e-12:
        print('Warning in system inversion: system might be non invertible')
    gA = np.array(G.A)
    gB = np.array(G.B)
    gC = np.array(G.C)

    invD = np.linalg.inv(gD)
    A = gA - gB*invD*gC 
    B = gB*invD
    C = -invD*gC
    D = invD

    invG = control.StateSpace(A, B, C, D)
    return invG


def ss_transpose(G):
    '''Transpose StateSpace G=(A,B,C,D) as G.T=(A.T,C.T,B.T,D.T)'''
    A, B, C, D = ssdata(G)  
    return control.StateSpace(A.T, C.T, B.T, D.T)


def sigma_trivial(G, w):
    '''Compute singular values of system G with trivial algorithm
    by using the pulsation grid w'''
    """sigma_trivial(G,w) -> s
    G - LTI object, order n
    w - frequencies, length m
    s - (m,n) array of singular values of G(1j*w)"""
    m, p, _ = G.freqresp(w)
    sjw = (m*np.exp(1j*p)).transpose(2, 0, 1)
    sv = np.linalg.svd(sjw, compute_uv=False)
    return sv


def controller_residues(real_c=[], real_p=[], cplx_c=[], cplx_p=[]):
    '''Construct controller from residue formulation, in StateSpace form
    K = sum(real_c/(s - real_p)) + sum(imag_c / (s-imag_p) + *imag_c/(s-*imag_p))
    real_c: coefficients for real poles
    real_p: real poles, same size as real_c
    cplx_c: coefficients for complex poles
    cplx_p: complex poles (only one, automatically paired with its conj)
    Controller order is: len(real_c) + 2*len(imag_c)'''
    # Init
    K = control.StateSpace([], [], [], 0)
    # Real poles
    def ss1(c, p):
        return control.StateSpace(p, c, 1, 0)
    nreal = len(real_c)
    for ii in range(nreal):
        K += ss1(real_c[ii], real_p[ii])
    # Complex poles
    re = np.real
    im = np.imag
    def ss2(c, p):
        return control.StateSpace(np.array([[2*re(p), -np.abs(p)**2], [1, 0]]),
                                  np.array([[2*(re(p)*re(c) - im(p)*im(c)), 2*re(c)]]).T,
                                  [0, 1],
                                  [0])
    ncplx = len(cplx_c)
    for ii in range(ncplx):
        K += ss2(cplx_c[ii], cplx_p[ii])
    # End
    return K


def controller_residues_getidx(n_real, n_cplx):
    '''Return indices of real coefficients, real poles, complex coefficients and complex poles
    given n_real real poles and n_cplx complex poles
    The indices may be used to extract quantities of interest with e.g. theta[idx[n]]'''
    idx_vec = np.arange(0, 2*n_real+4*n_cplx)
    real_c_idx = idx_vec[0:n_real] 
    real_p_idx = idx_vec[n_real:2*n_real]
    cplx_c_re_idx = idx_vec[2*n_real:2*n_real+n_cplx] 
    cplx_c_im_idx = idx_vec[2*n_real+n_cplx:2*n_real+2*n_cplx]
    cplx_p_re_idx = idx_vec[2*n_real+2*n_cplx:2*n_real+3*n_cplx]
    cplx_p_im_idx = idx_vec[2*n_real+3*n_cplx:] # end=2*n_real+4*n_cplx
    return real_c_idx, real_p_idx, cplx_c_re_idx, cplx_c_im_idx, cplx_p_re_idx, cplx_p_im_idx


def controller_residues_wrapper(theta, n_real, n_cplx):
    '''From theta=[real_c, real_p, cplx_c, cplx_p] values, make controller based on residue
    formula with only theta as input
    theta == real decision variable
    Usage: make_K = lambda x: controller_residues_wrapper(x, n_real, n_cplx) 
    where n_real, n_cplx have a given value (e.g. in script)'''
    assert len(theta)==2*n_real+4*n_cplx, 'Length of parameters not suitable.' 

    real_c_idx, real_p_idx, \
    cplx_c_re_idx, cplx_c_im_idx, \
    cplx_p_re_idx, cplx_p_im_idx = controller_residues_getidx(n_real, n_cplx)

    real_c = theta[real_c_idx] 
    real_p = theta[real_p_idx]
    cplx_c = theta[cplx_c_re_idx] + 1j*theta[cplx_c_im_idx]
    cplx_p = theta[cplx_p_re_idx] + 1j*theta[cplx_p_im_idx]
    return controller_residues(real_c, real_p, cplx_c, cplx_p) 


def test_controller_residues(n_real, n_cplx):
    '''Test controller formulation through residues c1/(s-p1)+...
    n_real and n_cplx are the number of real and complex poles
    Poles are sampled randomly but stable'''
    rand = np.random.rand
    def rand_in(bnd=[-1,1], n1=1):
        return bnd[0]+(bnd[1]-bnd[0])*rand(n1,)
    
    re_p_bnd = [-10, 0] # stable poles
    im_p_bnd = [0, 10] # positive im
    re_c_bnd = [-10, 10]
    im_c_bnd = [-10, 10]

    real_c = rand_in(re_c_bnd, n_real)
    real_p = rand_in(re_p_bnd, n_real)
    cplx_c_re = rand_in(re_c_bnd, n_cplx)
    cplx_p_re = rand_in(re_p_bnd, n_cplx)
    cplx_c_im = rand_in(im_c_bnd, n_cplx)
    cplx_p_im = rand_in(im_p_bnd, n_cplx)
    
    print('Coefficients are:', real_c, cplx_c_re+1j*cplx_c_im)
    print('Poles are:', real_p, cplx_p_re+1j*cplx_p_im)

    # K1: vanilla formula
    K1 = controller_residues(real_c, real_p, cplx_c_re+1j*cplx_c_im, cplx_p_re+1j*cplx_p_im)
    # K2: 1D (for optimization)
    theta = np.array(list(real_c) + list(real_p) +
                     list(cplx_c_re) + list(cplx_c_im) +
                     list(cplx_p_re) + list(cplx_p_im))
    K2 = controller_residues_wrapper(theta, n_real, n_cplx)

    # Check eigenvalues
    compare_controllers(K1, K2)

    return K1, K2


def rncf(G):
    '''Compute right normalized coprime factorization
    G = Nr * inv(Mr)
    Support for MIMO systems not tested'''
    A, B, C, D = ssdata(G)
    n = A.shape[0]
    E = np.eye(n)
    p, m = D.shape

    if n>0:
        Q = np.zeros((n,n))
        R = np.block([[np.eye(m), D.T], [D, -np.eye(p)]])
        S = np.hstack((np.zeros((n,m)), C.T))
        #_, _, K = control.care(A, np.hstack((B, np.zeros((n,p)))), Q, R, S, E)
        K = control.care(A, np.hstack((B, np.zeros((n,p)))), Q, R, S, E)[2]
    else:
        K = np.zeros((m+p, n))

    _, s, vh = la.svd(D)
    v = vh.conj().T 
    nsv = min([p,m])
    s = np.diag(s[0:nsv]) # make diagonal matrix from nsv first elts
    Z = v @ np.diag(np.vstack((1/np.sqrt(1+s**2), np.ones((m-nsv,1))))) @ vh

    # Construct output
    F = -K[:m, :]
    Amn = A + B*F
    Bmn = B*Z
    Cmn = np.vstack((F, C+D*F))
    Dmn = np.vstack((Z, D*Z))
    FACT = control.StateSpace(Amn, Bmn, Cmn, Dmn)
    Mr = control.StateSpace(Amn, Bmn, Cmn[:m,:], Dmn[:m,:])
    Nr = control.StateSpace(Amn, Bmn, Cmn[m:m+p,:], Dmn[m:m+p,:])
    return FACT, Mr, Nr


def lncf(G):
    '''Compute left normalized coprime factorization
    using rncf
    G = inv(Ml) * Nl
    Support for MIMO systems not tested'''
    FACT = rncf(ss_transpose(G))[0] 
    FACT = ss_transpose(FACT)
    Amn, Bmn, Cmn, Dmn = ssdata(FACT)
    p, m = Dmn.shape
    Ml = control.StateSpace(Amn, Bmn[:,:p], Cmn, Dmn[:,:p]) 
    Nl = control.StateSpace(Amn, Bmn[:,p:p+m], Cmn, Dmn[:,p:p+m]) 
    return FACT, Ml, Nl


def youla_left_coprime(G, K, Q):
#def youla_left_coprime(Ul, Vl, Ml, Nl, Q):
    '''Youla controller from left coprime factorization of G and K0
    G = inv(Ml)*Nl
    K0 = inv(Vl)*Ul
    -> Ky = inv(Vt+Q*Nt)*(Ut+Q*Mt)
    There is also a formula with Ky=LFT(J, Q) but I cannot make it work
    TODO check operations on ss'''
    _, Ml, Nl = lncf(G)
    _, Vl, Ul = lncf(K)
    Ky = ss_inv(Vl + Q * Nl) * (Ul + Q * Ml) # or \
    return Ky


def youla_right_coprime(G, K, Q):
#def youla_right_coprime(Ur, Vr, Mr, Nr, Q):
    '''Youla controller from right coprime factorization of G and K0
    G = Nr*inv(Mr) 
    K0 = Ur*inv(Vr)
    -> Ky = (U+M*Q)*inv(V+N*Q)
    There is also a formula with Ky=LFT(J, Q) but I cannot make it work
    TODO check operations on ss'''
    _, Mr, Nr = rncf(G)
    _, Vr, Ur = rncf(K)
    Ky = (Ur + Mr * Q) * ss_inv(Vr + Nr * Q) # or /
    return Ky


def youla_Qab(Ka, Kb, Gstab):
    '''Return controller Qab such that Youla(G, Ka, Qab) = Kb
    Q(a-->b)
    I fear it might not be a minimum realization if Gstab contains Ka or Kb
    '''
    Qab = control.feedback(Kb - Ka, Gstab, +1)
    return Qab


def youla_Q0b(Ka, K0, G):
    '''Return controller Q0b such that Youla(G, K0, Qab) = Kb
    Difference with youla_Qab is that Ka=K0 so that Gstab is expressed as a feedback
    I am not sure of the importance of this separate function'''
    Q0b = control.feedback(Ka - K0, control.feedback(G, K0, +1), +1)
    return Q0b


def balreal(G):
    '''Return balanced realization of G'''
    return control.balred(G, orders=G.nstates)
    

def balreal_(G): 
    '''Return balanced realization of G + Hankel singular values
    Warning: two output arguments to unpack here
    Warning: fails if system is unstable
    Better: use function youla_utils.sys_hsv()'''
    return balreal(G), control.hsvd(G)


def reduceorder(G):
    '''Reduce order of ss G by balanced realization
    Warning: does not a priori conserve stability'''
    Gb = balreal(G)
    return control.minreal(Gb)


def baltransform(G):
    '''Return transformation matrix T making G balanced
    T is such that the gramians in the new space are diagonal

    Wchat = inv(T)*Wc*inv(T).T
    Wohat = T.T*Wo*T
    and the balanced system would be as follows:
    Ahat = inv(T)*A*T
    Bhat = inv(T)*B
    Chat = C*T

    Warning: works only for stable systems!!!

    Algorithms is from: Computation of system balancing transformations 
    and other applications of simultaneous diagonalization algorithms
    A. Laub; M. Heath; C. Paige; R. Ward'''
    # Compute gramians
    Wo = control.gram(G, 'o')
    Wc = control.gram(G, 'c')
    # Get cholesky decomposition of gramians
    chol = np.linalg.cholesky
    Lo = chol(Wo)
    Lc = chol(Wc)
    # SVD product of chol
    uu, ss, vvh = np.linalg.svd(Lo.T@Lc)
    SS = np.diag(ss)
    # Form transform
    inv = np.linalg.inv
    T = Lc @ vvh.T @ inv(chol(SS))
    ## Check: this should be 0
    #dWchat = inv(T)@Wc@inv(T).T - SS 
    #dWohat = T.T@Wo@T - SS
    return np.asarray(T)


def sys_hsv(sys):
    '''Compute Hankel Singular Values of system G
    G can be unstable (major difference with control toolbox)
    We use SLYCOT to compute HSV but the routine itself (why?) is not
    accessible, so we use a metaroutine that calls SLYCOT HSV (AB13AX)'''
    try:
        from slycot import ab09md
    except ImportError:
        raise Exception("can't find slycot subroutine ab09md")
    # system dimension
    n = np.size(sys.A, 0)
    m = np.size(sys.B, 1)
    p = np.size(sys.C, 0)
    # system balancing uses HSV
    _, _, _, _, _, hsv = ab09md('C', 'B', 'N', 
        n, m, p, sys.A, sys.B, sys.C,
        alpha=0.0, nr=n, tol=0.0)
    # unstable modes are assigned HSV=0.0 --> make inf as matlab convention
    hsv[hsv==0.0] = np.inf # replace 0.0 with inf
    hsv = np.sort(hsv) # increasing order
    hsv = np.flip(hsv) # put inf in first elemnts
    return hsv


def balred_rel(sys, hsv_threshold, method='truncate'):
    '''Balanced reduction of G based on Hankel Singular Values
    The control toolbox cannot compute HSV for unstable systems so
    we resort to SLYCOT directly, which makes a very complicated function
    Function is inspired from python-control toolbox'''

    if method != 'truncate' and method != 'matchdc':
        raise ValueError("supported methods are 'truncate' or 'matchdc'")
    elif method == 'truncate':
        try:
            from slycot import ab09md, ab09ad
        except ImportError:
            raise Exception(
                "can't find slycot subroutine ab09md or ab09ad")
    elif method == 'matchdc':
        try:
            from slycot import ab09nd
        except ImportError:
            raise Exception("can't find slycot subroutine ab09nd")

    time_type = 'C'             # continuous time
    alpha = 0.                  # redefine stability bnd
    job = 'B'                   # type of algorithm 
         # (B) balanced sqrt Balance & Truncate, (N) balancing-free sqrt B&T 
    equil = 'N'                 # triplet (A,B,C): scale (S) or not (N)
    tol = 0.0                   # unused because input argument missing in SLYCOT

    rsys = []                   # empty list for reduced system

    # system dimension
    n = np.size(sys.A, 0)
    m = np.size(sys.B, 1)
    p = np.size(sys.C, 0)

    # system hsv
    hsv = sys_hsv(sys)
    elim = (hsv / np.max(hsv[np.isfinite(hsv)])) < hsv_threshold
    nr = n - np.sum(elim)

    # reduce
    if method == 'truncate':
        Dr = sys.D
        # check system stability
        if np.any(np.linalg.eigvals(sys.A).real >= 0.0):
            # unstable branch
            Nr, Ar, Br, Cr, Ns, _ = ab09md(time_type, job, equil, 
                n, m, p, sys.A, sys.B, sys.C,
                alpha=alpha, nr=nr, tol=tol)
        else:
            # stable branch
            Nr, Ar, Br, Cr, _ = ab09ad(time_type, job, equil,
                n, m, p, sys.A, sys.B, sys.C, nr=nr,
                tol=tol)

    elif method == 'matchdc':
        Nr, Ar, Br, Cr, Dr, Ns, _ = ab09nd(time_type, job, equil,
            n, m, p, sys.A, sys.B, sys.C, sys.D,
            alpha=alpha, nr=nr, tol1=tol, tol2=tol)

    # output new system
    rsys.append(control.StateSpace(Ar, Br, Cr, Dr))

    return rsys[0], hsv, nr


#def matlab_lncf(A, B, C, D, eng=None): 
#    '''Left normalized coprime factorization, called with Matlab
#    [~, Ml, Nl] = lncf(sys); sys = inv(Ml) * Nl
#    NOT IMPLEMENTED YET
#    Jokes on me: rncf/lncf are from Matlab/2019a'''
#    if eng is None:
#        eng = matlab.engine.start_matlab()
#    Ml, Nl = eng.lncf_mat(matlab.double(A.tolist()), 
#        matlab.double(B.tolist()), 
#        matlab.double(C.tolist()), 
#        matlab.double(D.tolist()), nargout=2)
#    return Ml, Nl
#
#
#def matlab_rncf(A, B, C, D, eng=None):
#    '''Right normalized coprime factorization, called with Matlab
#    [~, Mr, Nr] = rncf(sys); sys = Mr * inv(Nr)
#    NOT IMPLEMENTED YET
#    Jokes on me: rncf/lncf are from Matlab/2019a'''
#    if eng is None:
#        eng = matlab.engine.start_matlab()
#    Mr, Nr = eng.rncf_mat(matlab.double(A.tolist()), 
#        matlab.double(B.tolist()), 
#        matlab.double(C.tolist()), 
#        matlab.double(D.tolist()), nargout=2)
#    return Mr, Nr


def slowfast(G, wlim):
    '''Slow-fast decomposition of SISO LTI StateSpace G
    G(s) = Gslow(s) + Gfast(s) 
    with modes in Gslow < wlim and modes in Gfast >= wlim
    Unstable modes are treated indifferently
    Bad implementation but working'''
    # residue decomposition of tf
    Gtf = control.ss2tf(G)
    r, p, k = sig.residue(Gtf.num[0][0], Gtf.den[0][0])

    # direct feedthrough considered fast
    if (k.shape==(0,)):
        k = 0

    # locate freq
    wn = np.abs(p)
    idx_slow = np.where(wn<wlim)[0]
    idx_fast = np.where(wn>=wlim)[0]
    s = control.tf('s')

    # slow
    Gslow = control.tf(0,1)
    for ii in idx_slow:
        Gslow += r[ii]/(s - p[ii])
    # ensure real
    Gslow = make_tf_real(Gslow)
    Gslow = control.tf2ss(Gslow)

    # fast
    Gfast = control.tf(k,1)
    for ii in idx_fast:
        Gfast += r[ii]/(s - p[ii])
    # ensure real
    Gfast = make_tf_real(Gfast)
    Gfast = control.tf2ss(Gfast)

    return Gslow, Gfast 


def make_tf_real(G):
    '''Ensure TF G has real num/den coefficients'''
    return control.tf(np.real(G.num[0][0]), np.real(G.den[0][0]))


def compare_controllers(K1, K2):
    '''Compare controllers of same shape with eigenvalues(A) and dcgain
    No hinfnorm function seem to exist in Python (otherwise compute hinfnorm(K1-K2))'''
    #eig1 = la.eig(K1.A)[0]
    #eig2 = la.eig(K2.A)[0]
    #deig = eig1 - eig2
    print('Comparing controllers...')
    print('\t hinfnorm diff = ', norm(K1) - norm(K2))
    #print('\t dEig norm =', la.norm(deig))
    print('\t dcgains diff =', control.dcgain(K1) - control.dcgain(K2))
    #return K1, K2


def norm(G, p=np.inf, hinf_tol=1e-6, eig_tol=1e-8):
    """
    Code adapted from HAROLD control package in Python
    https://github.com/ilayn/harold/blob/master/harold/_system_props.py
    Documentation is reproduced as is
    Function was verified in SISO, not in MIMO
    ----------
    |Computes the system p-norm. Currently, no balancing is done on the
    |system, however in the future, a scaling of some sort will be introduced.
    |Currently, only H₂ and H∞-norm are understood.
    |For H₂-norm, the standard grammian definition via controllability grammian,
    |that can be found elsewhere is used.
    |Parameters
    |----------
    |G : {State,Transfer}
    |    System for which the norm is computed
    |p : {int,np.inf}
    |    The norm type; `np.inf` for H∞- and `2` for H2-norm
    |hinf_tol: float
    |    When the progress is below this tolerance the result is accepted
    |    as converged.
    |eig_tol: float
    |    The algorithm relies on checking the eigenvalues of the Hamiltonian
    |    being on the imaginary axis or not. This value is the threshold
    |    such that the absolute real value of the eigenvalues smaller than
    |    this value will be accepted as pure imaginary eigenvalues.
    |Returns
    |-------
    |n : float
    |    Resulting p-norm
    |Notes
    |-----
    |The H∞ norm is computed via the so-called BBBS algorithm ([1]_, [2]_).
    |.. [1] N.A. Bruinsma, M. Steinbuch: Fast Computation of H∞-norm of
    |    transfer function. System and Control Letters, 14, 1990.
    |    :doi:`10.1016/0167-6911(90)90049-Z`
    |.. [2] S. Boyd and V. Balakrishnan. A regularity result for the singular
    |       values of a transfer matrix and a quadratically convergent
    |       algorithm for computing its L∞-norm. System and Control Letters,
    |       1990. :doi:`10.1016/0167-6911(90)90037-U`
    ----------
    """

    # process norm input
    if p not in (2, np.inf):
        raise ValueError('The p in p-norm is not 2 or infinity. If you'
                         ' tried the string \'inf\', use "np.inf" instead')

    T = G # supposed state space
    a, b, c, d = ssdata(T)
    T._isstable = isstable(T)

    # 2-norm
    if p == 2:
        # Handle trivial infinities
        if not np.allclose(d, np.zeros_like(d)) or (not T._isstable):
            return np.inf

        #if T.SamplingSet == 'R':
        # only continuous
        # legacy:
        # lyapunov_eq_solver(A, Y) solves:
        # solve XA + AT X + Y = 0
        # here:
        # x = lyapunov_eq_solver(a.T, b @ b.T)
        # so solves: X AT + A X + BBT = 0
        # equivalent with control toolbox:
        # control.lyap(A, Q) solves:
        # A X + X AT + Q = 0
        # Q + BBT
        # A = AT
        x = control.lyap(a, b @ b.T)
        return np.sqrt(np.trace(c @ x @ c.T))
        #else:
        #    x = lyapunov_eq_solver(a.T, b @ b.T, form='d')
        #    return np.sqrt(np.trace(c @ x @ c.T + d @ d.T))
    # ∞-norm
    elif np.isinf(p):
        if not T._isstable:
            return np.inf

        # Initial gamma0 guess
        # Get the max of the largest svd of either
        #   - feedthrough matrix
        #   - G(iw) response at the pole with smallest damping
        #   - G(iw) at w = 0

        # Formula (4.3) given in Bruinsma, Steinbuch Sys.Cont.Let. (1990)
    
        T.poles = control.pole(T)

        if any(T.poles.imag):
            J = [np.abs(x.imag/x.real/np.abs(x)) for x in T.poles]
            ind = np.argmax(J)
            low_damp_fr = np.abs(T.poles[ind])
        else:
            low_damp_fr = np.min(np.abs(T.poles))

        f, _, w = control.freqresp(sys=T, omega=[0, low_damp_fr])#, w_unit='rad/s',
                                  #output_unit='rad/s')

        T._isSISO = (T.ninputs==1 and T.noutputs==1)
        if T._isSISO:
            lb = np.max(np.abs(f))
        else:
            # Only evaluated at two frequencies, 0 and wb
            lb = np.max(la.norm(f, ord=2, axis=(0, 1)))

        # Finally
        gamma_lb = np.max([lb, la.norm(d, ord=2)])

        # Start a for loop with a definite end! Convergence is quartic!!
        for x in range(51):

            # (Step b1)
            test_gamma = gamma_lb * (1 + 2*np.sqrt(np.spacing(1.)))

            # (Step b2)
            R = d.T @ d - test_gamma**2 * np.eye(d.shape[1])
            S = d @ d.T - test_gamma**2 * np.eye(d.shape[0])
            # TODO : Implement the result of Benner for the Hamiltonian later
            solve = la.solve
            Ham = np.block([[a - b @ solve(R, d.T) @ c,
                             -test_gamma * b @ solve(R, b.T)],
                            [test_gamma * c.T @ solve(S, c),
                             -(a - b @ solve(R, d.T) @ c).T]])
            eigs_of_H = la.eigvals(Ham)

            # (Step b3)
            im_eigs = eigs_of_H[np.abs(eigs_of_H.real) <= eig_tol]
            # If none left break
            if im_eigs.size == 0:
                gamma_ub = test_gamma
                break
            else:
                # Take the ones with positive imag part
                w_i = np.sort(np.unique(np.abs(im_eigs.imag)))
                # Evaluate the cubic interpolant
                m_i = (w_i[1:] + w_i[:-1]) / 2
                f, _, w = control.freqresp(sys=T, omega=m_i)#, w_unit='rad/s',
                                         # output_unit='rad/s')
                if T._isSISO:
                    gamma_lb = np.max(np.abs(f))
                else:
                    gamma_lb = np.max(la.norm(f, ord=2, axis=(0, 1)))
                # assign gamma_ub to avoid error if loop doesnt end in 51 iter??? 
                gamma_ub = test_gamma

        return (gamma_lb + gamma_ub)/2


def condswitch(ur, yr, K, dt, w_y, w_u, w_decay):
    """Controller conditionning for switching as per Paxman phd
    Find signals and initial state compatible with offline controller
    Answers the question: if the signals were (ur, yr) then what is
    the controller state compatible with those?
    Problem: requires ur, yr which are past signals, 
    so we will need to get timeseries.csv"""
    # Discretize
    Kd = control.c2d(K, dt, 'tustin')
    A, B, C, D = control.ssdata(Kd)

    # Format input
    r = len(ur)
    Ur = ur.reshape(-1,)
    Yr = yr.reshape(-1,)
    n = Kd.nstates

    # Build propagation matrices
    invA = np.linalg.inv(A)
    Gamma_r = np.zeros((r, n))
    Gamma_r[0, :] = C@invA
    for ii in range(r-1): # 0 to r-1, update row ii+1
        Gamma_r[ii+1, :] = Gamma_r[ii, :]@invA

    Tr = np.zeros((r, r))
    Tr0 = np.zeros((r, 1)) # 1st col
    for ii in range(r):
        Tr0[ii] = C @ invA**(ii+1) @ B
    Tr0[0] += np.asarray(-D).ravel()

    # fill columns of Tr
    Tr[:, 0] = Tr0.ravel() # first col
    for jj in range(1, r): # rest of col (because Tr0[:-0]=empty)
        Tr[:, jj] = np.vstack((np.zeros((jj, 1)), Tr0[:-jj])).ravel()
    
    # Build weight matrix
    W_decay = np.diag(w_decay**np.flip(np.arange(0,r))) 
    W = la.block_diag(w_y*np.eye(r), w_u*np.eye(r))
    W *= la.block_diag(W_decay, W_decay)

    # Solve backwards-prediction problem
    Asol = W@np.block([[-Tr, Gamma_r], [np.eye(Tr.shape[0]), np.zeros(Gamma_r.shape)]]) 
    bsol = W@np.hstack((Ur, Yr))
    sol = np.linalg.lstsq(Asol, bsol, rcond=None)[0]

    # Extract solution
    xn = sol[-n:]
    yhat = sol[:r]
    uhat = Gamma_r@xn - Tr@yhat
    return xn, yhat, uhat


def export_controller(filename, K):
    '''Export frequency response of controller to file
    Be careful: mag not in dB'''
    mag, phase, w = control.bode(K, plot=False)
    A, B, C, D = ssdata(K)
    sio.savemat(filename, mdict=dict(mag=mag, phase=phase, w=w, A=A, B=B, C=C, D=D))
    print('Exported controller to file: ', filename)


def test_coprime_read():
    import utils_flowsolver as flu
    path = '/stck/wjussiau/fenics-python/ns/data/o1/coprime/'
    G = flu.read_ss(path + 'G/G.mat')
    Ml = flu.read_ss(path + 'G/Ml.mat')
    Nl = flu.read_ss(path + 'G/Nl.mat')
    Mr = flu.read_ss(path + 'G/Mr.mat')
    Nr = flu.read_ss(path + 'G/Nr.mat')
    H1 = ss_inv(Ml)*Nl
    H2 = Nr*ss_inv(Mr)
    compare_controllers(G, H1)
    compare_controllers(G, H2)
    return G, Ml, Nl, Mr, Nr


if __name__=='__main__':
    import utils_flowsolver as flu

    # Check residues formula
    print('*'*50)
    print('Testing residues formula:')
    K1, K2 = test_controller_residues(n_real=2, n_cplx=1)

    # Check MIMO (dummy system)
    print('*'*50)
    print('Testing MIMO stacking')
    A = np.array([[-0.7698, 0.08247], [0.08247, -1.561]])
    B = np.array([[0, 0.1616]]).T
    C = np.array([[1.978, 0], [0, 2.36]])
    D = np.array([[0, 0.3977]]).T
    G2 = control.StateSpace(A, B, C, D) 
    Psi = build_block_Psi(G2)

    ny = G2.C.shape[0]
    O1 = control.StateSpace([], [], [], 1) 
    Z1 = control.StateSpace([],[],[],np.zeros((1,2)))
    E1 = control.StateSpace([],[],[],np.eye(2)) 
    Psibis = ss_vstack(ss_hstack(Z1, O1), ss_hstack(E1, -G2))

    compare_controllers(Psi, Psibis)

    # Checking K00
    print('*'*50)
    print('Testing K(0)=0')
    sspath = '/scratchm/wjussiau/fenics-python/cavity/data/regulator/' 
    K0 = flu.read_ss(sspath + 'multiK/K1.mat') 
    G = flu.read_ss(sspath + 'sysid_o24_ssest_QB.mat') 
    CL = control.feedback(G, K0, +1)
    print('Feedback stable: ', isstable(CL))
    K0dc = control.dcgain(K0)
    p = 10
    theta = np.array([1.12, 3.2, -3.21])
    Q00 = basis_laguerre_K00(G, K0, p, theta) 

    K001 = youla_laguerre_K00(G, K0, p, theta, check=True)
    K00 = youla(G, K0, Q00)
    # this should be 0
    K001dc = control.dcgain(K001)
    K00dc = control.dcgain(K00)
    print('DCgain of K00 is: ', K00dc)
    print('DCgain of K001 is: ', K001dc)

    
    # Check Youla stability on MIMO
    #print('Testing Youla MIMO approach')
    #sspath = '/stck/wjussiau/fenics-python/ns/data/m1/regulator/'
    #K0 = flu.read_ss(sspath + 'K0_o8_y1=[3,0]_y2=[1,05]_S_KS_clpoles.mat')
    #G = flu.read_ss(sspath + 'sysid_o16_ny=2_y1=[3,0]_y2=[1,05].mat')
    #print('Feedback(G, K0, +1) is stable: ', isstable(control.feedback(G, K0, +1)))     
    #ndimtheta = 3
    #nmc = 10
    #for ii in range(nmc):
    #    print('\t Random Youla parameter: ', ii+1)

    #    # Stack by hand
    #    p1 = 10*np.random.rand()
    #    theta1 = np.random.randn(ndimtheta,)
    #    Q1 = basis_laguerre_ss(p=p1, theta=theta1)

    #    p2 = 10*np.random.rand()
    #    theta2 = np.random.randn(ndimtheta,)
    #    Q2 = basis_laguerre_ss(p=p2, theta=theta2)

    #    Q = ss_hstack(Q1, Q2)
    #    Ky = youla(G, K0, Q)
    #    print('\t Feedback(G, Ky, +1) is stable: ', isstablecl(G, Ky, +1)) 

    #    # Auto stack
    #    Ky2 = youla_laguerre_mimo(G, K0, p=[p1, p2], theta=np.vstack((theta1, theta2)), verbose=False)
    #    #compare_controllers(Ky, Ky2)

    if 1:
        print('Performing Hinf/H2 synthesis with reference model...')

        # %% Synthesis
        s = control.tf([1, 0], 1)
        
        G = (s-1)/(s+1)**2
        #G = (s+2)/(s+1)**2
        G = control.tf2ss(G)
        
        # Reference loop
        w = 2
        xi = 0.04
        CLr = w**2 * control.tf([1, -1], [1, w*xi, w**2])
        CLr = control.tf2ss(CLr)
        
        # Weightings
        Wr = control.tf2ss(control.tf(0.1, 1))
        We = control.tf2ss(control.tf(0.2, 1))
        Wu = control.tf2ss(control.tf(0.3, 1))
        Wb = control.tf2ss(control.tf(0.4, 1))
        Wcl = control.tf2ss(4*control.tf([1, 1], [1, 0.1]))
        
        Zss = control.tf2ss(control.tf(0, 1))
        Iss = control.tf2ss(control.tf(1, 1))
        
        # Augmented plant
        Wout = ss_blkdiag_list([We, Wu, Wcl, Iss])
        Win = ss_blkdiag_list([Wr, Wb, Iss])
        P_4blk = ss_vstack(ss_hstack(Iss, -G, -G),
                           ss_hstack(Zss, Zss, Iss),
                           ss_hstack(Zss, G-CLr, G),
                           ss_hstack(Iss, -G, -G))
        P_4blk = Wout * P_4blk * Win
        
        P_4blk = Wout * \
                ss_vstack(ss_hstack(Iss, -Iss, Zss, Zss), ss_hstack(Zss, Zss, Iss, Zss), ss_hstack(Zss, Iss, Zss, -Iss), ss_hstack(Iss, -Iss, Zss, Zss)) * \
                ss_blkdiag_list([Iss, G, Iss, CLr]) * \
                ss_vstack(ss_hstack(Iss, Zss, Zss), ss_hstack(Zss, Iss, Iss), ss_hstack(Zss, Zss, Iss), ss_hstack(Zss, Iss, Zss)) * \
                Win
        
        P_syn = P_4blk
        
        Khinf, CLhinf, gamma_hinf, _ = control.hinfsyn(P_syn, 1, 1)
        Kh2 = control.h2syn(P_syn, 1, 1)
        
        S = control.feedback(1, G*Khinf)
        KS = control.feedback(Khinf, G) 
        GS = control.feedback(G, Khinf)
        T = control.feedback(G*Khinf, 1)
        
        GS_h2 = control.feedback(G, Kh2)
        
        FPK_hand = ss_vstack(ss_hstack(We*S*Wr, -We*GS*Wb),
                             ss_hstack(Wu*KS*Wr, -Wu*T*Wb),
                             ss_hstack(Wcl*Wr*T, Wcl*Wb*(GS-CLr)))
        FPK_true = P_syn.lft(Khinf)
        
        print('Computed Hinf norm of F(P,K): {0}'.format(norm(FPK_hand)))
        print('Computed gamma idk why it doesnt work: {0}'.format(gamma_hinf))
        
        gamma_true = norm(FPK_hand)

        # Try wrapped version
        print('***'*5)
        K_wrap, gamma_wrap = hinfsyn_mref(G, We, Wu, Wb, Wr, CLr, Wcl, syn='Hinf')
        #K_wrap, gamma_wrap = hinfsyn_mref(G, We, Wu, Wb, Wr, CLr, Wcl, syn='H2')
        print('With wrapped version: norm of F(P,K): {0}'.format(gamma_wrap))





        
