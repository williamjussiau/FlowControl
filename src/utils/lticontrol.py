"""LTI control utilities: state-space I/O, controller stepping, Youla parametrization,
LQG/H-infinity synthesis, balanced reduction, coprime factorizations."""

import logging
import warnings

import control
import control.matlab as cmat
import numpy as np
import scipy.io as sio
import scipy.linalg as la
import scipy.signal as ss

logger = logging.getLogger(__name__)


# --- Controller I/O ---

def step_controller(K, x, e, dt):
    """Wrapper for stepping controller on one time step, from state (x),
    with input(e), up to time (dt) >> u=K*e
    Return controller output u and controller new state x"""
    e_rep = np.repeat(np.atleast_2d(e), repeats=2, axis=0).T
    Tsim = [0, dt]
    _, yout, xout = control.forced_response(
        K, U=e_rep, T=Tsim, X0=x, interpolate=False, return_x=True
    )
    u = np.ravel(yout)[0]
    x = xout[:, 1]
    return u, x


def read_matfile(path):
    """Read mat file without Duplicate variable warning"""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Duplicate variable name*")
        return sio.loadmat(path)


def read_regulator(path):
    """Read matrices of scipy.signal.StateSpace from provided .mat file path"""
    rd = read_matfile(path)
    return ss.StateSpace(rd["A"], rd["B"], rd["C"], rd["D"])


def read_ss(path):
    """Read matrices of control.StateSpace from provided .mat file path"""
    rd = read_matfile(path)
    return control.StateSpace(rd["A"], rd["B"], rd["C"], rd["D"])


def write_ss(sys, path):
    """Write control.StateSpace to file"""
    ssdict = {"A": sys.A, "B": sys.B, "C": sys.C, "D": sys.D}
    sio.savemat(path, ssdict)
    return 0


# --- State-space algebra ---

def ssdata(sys):
    """Return StateSpace matrices as np.array instead of np.matrix"""
    A, B, C, D = control.ssdata(sys)
    return np.asarray(A), np.asarray(B), np.asarray(C), np.asarray(D)


def ss_zero():
    """Return null SISO StateSpace"""
    arr0 = np.array([])
    return control.StateSpace(arr0, arr0, arr0, 0)


def ss_one():
    """Return identity SISO StateSpace"""
    return control.tf2ss(control.tf(1, 1))


def ss_vstack(sys1, *sysn):
    """Equivalent of Matlab [sys1; sys2]: same input, stacked outputs."""
    A, B, C, D = ssdata(sys1)
    for sys in sysn:
        A = la.block_diag(A, sys.A)
        B = np.vstack((B, sys.B))
        C = la.block_diag(C, sys.C)
        D = np.vstack((D, sys.D))
    return control.StateSpace(A, B, C, D)


def ss_hstack(sys1, *sysn):
    """Equivalent of Matlab [sys1, sys2]: stacked inputs, summed outputs."""
    A, B, C, D = ssdata(sys1)
    for sys in sysn:
        A = la.block_diag(A, sys.A)
        B = la.block_diag(B, sys.B)
        C = np.hstack((C, sys.C))
        D = np.hstack((D, sys.D))
    return control.StateSpace(A, B, C, D)


def ss_vstack_list(syslist):
    """Same as ss_vstack but with input list"""
    A, B, C, D = control.ssdata(syslist[0])
    for sys in syslist[1:]:
        A = la.block_diag(A, sys.A)
        B = np.vstack((B, sys.B))
        C = la.block_diag(C, sys.C)
        D = np.vstack((D, sys.D))
    return control.StateSpace(A, B, C, D)


def ss_hstack_list(syslist):
    """Same as ss_hstack but with input list"""
    A, B, C, D = control.ssdata(syslist[0])
    for sys in syslist[1:]:
        A = la.block_diag(A, sys.A)
        B = la.block_diag(B, sys.B)
        C = np.hstack((C, sys.C))
        D = np.hstack((D, sys.D))
    return control.StateSpace(A, B, C, D)


def ss_blkdiag_list(sys_list):
    """Make block diagonal system with input list"""
    return control.append(*sys_list)


def ss_inv(G):
    """Invert StateSpace G provided that G.D is invertible"""
    gA, gB, gC, gD = ssdata(G)
    if np.linalg.norm(gD) <= 1e-12:
        logger.warning("ss_inv: system might be non-invertible (norm(D) <= 1e-12)")
    invD = np.linalg.inv(gD)
    return control.StateSpace(
        gA - gB @ invD @ gC,
        gB @ invD,
        -invD @ gC,
        invD,
    )


def ss_transpose(G):
    """Transpose StateSpace G=(A,B,C,D) as G.T=(A.T,C.T,B.T,D.T)"""
    A, B, C, D = ssdata(G)
    return control.StateSpace(A.T, C.T, B.T, D.T)


def ssmult(G1, G2):
    """Multiplication of MIMO SS: G = G1 x G2 (OBSOLETE: use * operator)"""
    ZERO = np.zeros((G2.A.shape[0], G1.A.shape[1]))
    A = np.block([[G1.A, G1.B @ G2.C], [ZERO, G2.A]])
    B = np.vstack((G1.B @ G2.D, G2.B))
    C = np.hstack((G1.C, G1.D @ G2.C))
    D = G1.D @ G2.D
    return control.StateSpace(A, B, C, D)


def show_ss(sys):
    """Print StateSpace matrices to stdout"""
    for mat in ssdata(sys):
        print(mat)
        print("-" * 10)


# --- Stability and norms ---

def isstable(CL):
    """Return True if all poles of CL have non-positive real part"""
    return np.all(np.real(control.pole(CL)) <= 0)


def isstablecl(G, K0, sign=+1):
    """Return True if feedback(G, K0, sign) is stable"""
    return isstable(control.feedback(G, K0, sign=sign))


def sigma_trivial(G, w):
    """Compute singular values of G on frequency grid w.
    Returns array of shape (nw, n) where n = min(nout, nin)."""
    m, p, _ = G.freqresp(w)
    sjw = (m * np.exp(1j * p)).transpose(2, 0, 1)
    return np.linalg.svd(sjw, compute_uv=False)


def norm(G, p=np.inf, hinf_tol=1e-6, eig_tol=1e-8):
    """Compute H2 or H-infinity norm of G.
    Adapted from HAROLD control package (https://github.com/ilayn/harold)."""
    if p not in (2, np.inf):
        raise ValueError("p must be 2 or np.inf")

    T = G
    a, b, c, d = ssdata(T)
    T._isstable = isstable(T)

    if p == 2:
        if not np.allclose(d, np.zeros_like(d)) or not T._isstable:
            return np.inf
        x = control.lyap(a, b @ b.T)
        return np.sqrt(np.trace(c @ x @ c.T))

    # H-infinity norm
    if not T._isstable:
        return np.inf

    T.poles = control.pole(T)
    if any(T.poles.imag):
        J = [np.abs(x.imag / x.real / np.abs(x)) for x in T.poles]
        low_damp_fr = np.abs(T.poles[np.argmax(J)])
    else:
        low_damp_fr = np.min(np.abs(T.poles))

    f, _, _ = control.freqresp(sys=T, omega=[0, low_damp_fr])
    T._isSISO = T.ninputs == 1 and T.noutputs == 1
    if T._isSISO:
        lb = np.max(np.abs(f))
    else:
        lb = np.max(la.norm(f, ord=2, axis=(0, 1)))
    gamma_lb = np.max([lb, la.norm(d, ord=2)])

    for _ in range(51):
        test_gamma = gamma_lb * (1 + 2 * np.sqrt(np.spacing(1.0)))
        R = d.T @ d - test_gamma**2 * np.eye(d.shape[1])
        S = d @ d.T - test_gamma**2 * np.eye(d.shape[0])
        solve = la.solve
        Ham = np.block(
            [
                [a - b @ solve(R, d.T) @ c, -test_gamma * b @ solve(R, b.T)],
                [test_gamma * c.T @ solve(S, c), -(a - b @ solve(R, d.T) @ c).T],
            ]
        )
        eigs_of_H = la.eigvals(Ham)
        im_eigs = eigs_of_H[np.abs(eigs_of_H.real) <= eig_tol]
        if im_eigs.size == 0:
            gamma_ub = test_gamma
            break
        w_i = np.sort(np.unique(np.abs(im_eigs.imag)))
        m_i = (w_i[1:] + w_i[:-1]) / 2
        f, _, _ = control.freqresp(sys=T, omega=m_i)
        if T._isSISO:
            gamma_lb = np.max(np.abs(f))
        else:
            gamma_lb = np.max(la.norm(f, ord=2, axis=(0, 1)))
        gamma_ub = test_gamma

    return (gamma_lb + gamma_ub) / 2


# --- Youla parametrization ---

def youla(G, K0, Q):
    """Return controller K parametrized with Q using Youla formula.
    G: plant, K0: stabilizing base controller, Q: Youla parameter.
    Feedback convention: (+)."""
    Gstab = G.feedback(other=K0, sign=+1)
    Psi = build_block_Psi(Gstab)
    Kq = Psi.lft(Q)
    return K0 + Kq


def build_block_Psi(G):
    """Build block function Psi=[1, 0; 1, -G] for Youla parametrization.
    If G is SIMO: Psi=[zeros(1,ny), 1; eye(ny), -G]"""
    ny = G.noutputs
    O1 = control.StateSpace([], [], [], 1)
    Z1 = control.StateSpace([], [], [], np.zeros((1, ny)))
    E1 = control.StateSpace([], [], [], np.eye(ny))
    return ss_vstack(ss_hstack(Z1, O1), ss_hstack(E1, -G))


def youla_laguerre(G, K0, p, theta, verbose=False):
    """Compute Youla controller with Laguerre basis Q=Theta^T*Phi(s). SISO only."""
    Gstab = G.feedback(other=K0, sign=+1)
    Psi = build_block_Psi(Gstab)

    if type(theta) is int:
        N = 1
        warnings.warn("theta should be iterable, not int")
    else:
        N = len(theta)
    a = p
    a_vec = np.hstack((-a, 2 * a * (-1) ** (np.arange(2, N + 1))))
    a2 = np.triu(la.circulant(a_vec))
    b2 = np.eye(N)
    c2 = np.sqrt(2 * a) * (-1) ** (np.arange(2, N + 2))
    d2 = np.zeros((1, N))
    Qf = control.StateSpace(a2, b2, c2, d2)

    Qf1 = cmat.append(1, Qf)
    Psif = Psi * Qf1

    N = len(theta)
    theta = theta * (-1) ** (np.arange(N))
    ss_theta = control.StateSpace([], [], [], np.array([theta]).T)

    Kq = Psif.lft(ss_theta)
    K = K0 + Kq
    if verbose:
        print("\t Feedback(G, Ky, +1) is stable: ", isstablecl(G, K, +1))
    return K


def youla_laguerre_mimo(G, K0, p, theta, verbose=False):
    """Youla with SIMO plant G and MISO controller K using Laguerre basis.
    p=[p1, p2, ...], theta has Qi on each row."""
    nout = G.noutputs
    Q = basis_laguerre_ss(p=p[0], theta=theta[0, :])
    for i in range(1, nout):
        Qi = basis_laguerre_ss(p=p[i], theta=theta[i, :])
        Q = ss_hstack(Q, Qi)
    K = youla(G, K0, Q)
    if verbose:
        print("\t Feedback(G, Ky, +1) is stable: ", isstablecl(G, K, +1))
    return K


def youla_laguerre_K00(G, K0, p, theta, check=False):
    """Youla controller with K(0)=0 constraint using Laguerre basis. SISO only."""
    Q00 = basis_laguerre_K00(G, K0, p, theta)
    K = youla(G=G, K0=K0, Q=Q00)
    if check:
        print("DC gain of K (should be 0): ", control.dcgain(K))
    return K


def youla_laguerre_fromfile(theta, path):
    """Reduced Youla: Psif=g(G,K0,p) loaded from file, only LFT performed here."""
    theta = theta * (-1) ** (np.arange(len(theta)))
    ss_theta = control.StateSpace([], [], [], np.array([theta]).T)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Duplicate variable name*")
        rd = sio.loadmat(path)
    Psif = control.StateSpace(rd["Psi_A"], rd["Psi_B"], rd["Psi_C"], rd["Psi_D"])
    Kq = Psif.lft(ss_theta)
    K0 = control.StateSpace(rd["K0_A"], rd["K0_B"], rd["K0_C"], rd["K0_D"])
    return K0 + Kq


def youla_lqg(G, Qx, Ru, Qv, Rw, Q):
    """Youla controller in LQG form"""
    J = youla_lqg_lftmat(G, Qx, Ru, Qv, Rw)
    return J.lft(Q)


def youla_lqg_lftmat(G, Qx, Ru, Qv, Rw):
    """Return StateSpace J to be LFTed with Q for Youla LQG parametrization."""
    B = np.array(G.B)
    C = np.array(G.C)
    D = np.array(G.D)
    p, m = D.shape
    Im = np.eye(m)
    Ip = np.eye(p)
    Klqg, F, L = lqg_regulator(G, Qx, Ru, Qv, Rw)
    return control.StateSpace(
        Klqg.A,
        np.hstack((Klqg.B, B + L * D)),
        np.vstack((Klqg.C, -C - D * F)),
        np.block([[np.zeros((m, p)), Im], [Ip, Klqg.D]]),
    )


def youla_Qab(Ka, Kb, Gstab):
    """Return Qab such that Youla(G, Ka, Qab) = Kb."""
    return control.feedback(Kb - Ka, Gstab, +1)


def youla_Q0b(Ka, K0, G):
    """Return Q0b such that Youla(G, K0, Q0b) = Ka."""
    return control.feedback(Ka - K0, control.feedback(G, K0, +1), +1)


def youla_left_coprime(G, K, Q):
    """Youla controller from left coprime factorization of G and K0."""
    _, Ml, Nl = lncf(G)
    _, Vl, Ul = lncf(K)
    return ss_inv(Vl + Q * Nl) * (Ul + Q * Ml)


def youla_right_coprime(G, K, Q):
    """Youla controller from right coprime factorization of G and K0."""
    _, Mr, Nr = rncf(G)
    _, Vr, Ur = rncf(K)
    return (Ur + Mr * Q) * ss_inv(Vr + Nr * Q)


# --- LQG synthesis ---

def lqg_regulator(G, Qx, Qu, Qw, Qv):
    """Make LQG regulator. Returns (Klqg, F, L).
    Qx/Qu: state/input weights; Qw/Qv: process/measurement noise covariances."""
    A = np.array(G.A)
    B = np.array(G.B)
    C = np.array(G.C)
    D = np.array(G.D)
    n = A.shape[0]
    In = np.eye(n)
    p, m = D.shape
    Im = np.eye(m)
    Ip = np.eye(p)
    F = np.array(-control.lqr(A, B, Qx * In, Qu * Im)[0])
    L = np.array(-control.lqr(A.T, C.T, Qw * In, Qv * Ip)[0]).T
    Klqg = control.StateSpace(A + B * F + L * C + L * D * F, -L, F, 0)
    return Klqg, F, L


# --- H-infinity synthesis ---

def hinfsyn_mref(G, We, Wu, Wb, Wr, CLref, Wcl, syn="Hinf"):
    """Classic SISO mixed-sensitivity H-inf synthesis with model reference.
    Warning: negative feedback convention."""
    if syn not in ("Hinf", "H2"):
        raise ValueError("Only Hinf or H2 synthesis supported")

    Zo = ss_zero()
    Id = ss_one()

    Wout = ss_blkdiag_list([We, Wu, Wcl, Id])
    Win = ss_blkdiag_list([Wr, Wb, Id])
    P_syn = (
        ss_vstack(
            ss_hstack(Id, -Id, Zo, Zo),
            ss_hstack(Zo, Zo, Id, Zo),
            ss_hstack(Zo, Id, Zo, -Id),
            ss_hstack(Id, -Id, Zo, Zo),
        )
        * ss_blkdiag_list([Id, G, Id, CLref])
        * ss_vstack(
            ss_hstack(Id, Zo, Zo),
            ss_hstack(Zo, Id, Id),
            ss_hstack(Zo, Zo, Id),
            ss_hstack(Zo, Id, Zo),
        )
    )
    P_syn = Wout * P_syn * Win

    if syn == "Hinf":
        K, _, _, _ = control.hinfsyn(P_syn, 1, 1)
    else:
        K = control.h2syn(P_syn, 1, 1)

    return K, norm(P_syn.lft(K))


# --- Laguerre basis ---

def basis_laguerre_canonical(p, N):
    """Return first N transfer functions of Laguerre basis with pole p>0."""
    s = control.TransferFunction([1, 0], [1])
    Phi = np.zeros((N,), dtype=control.TransferFunction)
    L_i = 1 / (s + p)
    for i in range(N):
        Phi[i] = L_i
        L_i = L_i * (s - p) / (s + p)
    return np.sqrt(2 * p) * Phi


def basis_laguerre(p, theta):
    """Q(s) = sum_i(theta_i * phi_i(s)) with Laguerre basis."""
    if type(theta) is int:
        N = 1
        warnings.warn("theta should be iterable, not int")
    else:
        N = len(theta)
    return sum(basis_laguerre_canonical(p, N) * theta)


def basis_laguerre_canonical_ss(p, N):
    """Return first N elements of Laguerre basis with pole p>0, in state-space form."""
    a = p
    a_vec = np.hstack((-a, 2 * a * (-1) ** (np.arange(2, N + 1))))
    a2 = np.triu(la.circulant(a_vec).T)
    b2 = np.diag((-1) ** (np.arange(2, N + 2)))
    c2 = np.sqrt(2 * a) * (-1) ** (np.arange(2, N + 2))
    d2 = np.zeros((1, N))
    return control.StateSpace(a2, b2, c2, d2)


def basis_laguerre_ss(p, theta):
    """Q = sum(theta_i * phi_i(s; p)) with phi the Laguerre SS basis."""
    if type(theta) is int:
        N = 1
        warnings.warn("theta should be iterable, not int")
    else:
        N = len(theta)
    Phi = basis_laguerre_canonical_ss(p, N)
    return Phi * np.atleast_2d(np.array(theta)).T


def basis_laguerre_K00(G, K0, p, theta):
    """Compute Youla param Q00 ensuring K(0)=0. SISO only."""
    N = len(theta)
    K00 = control.dcgain(K0)
    Gstab = control.feedback(G, K0, +1)
    G00 = control.dcgain(Gstab)
    b0 = -K00 / (1 + K00 * G00)
    a0 = b0 * np.sqrt(p / 2)

    J = np.atleast_2d(np.ones((N + 1,)) * (-1) ** np.arange(0, N + 1))
    y0 = la.lstsq(J, np.array([a0]).reshape(-1))[0]
    kerJ = la.null_space(J)
    y = y0 + kerJ @ theta
    return basis_laguerre_ss(p=p, theta=y)


# --- Coprime factorizations ---

def rncf(G):
    """Right normalized coprime factorization: G = Nr * inv(Mr)."""
    A, B, C, D = ssdata(G)
    n = A.shape[0]
    p, m = D.shape

    if n > 0:
        Q = np.zeros((n, n))
        R = np.block([[np.eye(m), D.T], [D, -np.eye(p)]])
        S = np.hstack((np.zeros((n, m)), C.T))
        K = control.care(A, np.hstack((B, np.zeros((n, p)))), Q, R, S, np.eye(n))[2]
    else:
        K = np.zeros((m + p, n))

    _, s, vh = la.svd(D)
    v = vh.conj().T
    nsv = min([p, m])
    s = np.diag(s[0:nsv])
    Z = v @ np.diag(np.vstack((1 / np.sqrt(1 + s**2), np.ones((m - nsv, 1))))) @ vh

    F = -K[:m, :]
    Amn = A + B * F
    Bmn = B * Z
    Cmn = np.vstack((F, C + D * F))
    Dmn = np.vstack((Z, D * Z))
    FACT = control.StateSpace(Amn, Bmn, Cmn, Dmn)
    Mr = control.StateSpace(Amn, Bmn, Cmn[:m, :], Dmn[:m, :])
    Nr = control.StateSpace(Amn, Bmn, Cmn[m : m + p, :], Dmn[m : m + p, :])
    return FACT, Mr, Nr


def lncf(G):
    """Left normalized coprime factorization: G = inv(Ml) * Nl."""
    FACT = rncf(ss_transpose(G))[0]
    FACT = ss_transpose(FACT)
    Amn, Bmn, Cmn, Dmn = ssdata(FACT)
    p, m = Dmn.shape
    Ml = control.StateSpace(Amn, Bmn[:, :p], Cmn, Dmn[:, :p])
    Nl = control.StateSpace(Amn, Bmn[:, p : p + m], Cmn, Dmn[:, p : p + m])
    return FACT, Ml, Nl


# --- Balanced reduction ---

def balreal(G):
    """Return balanced realization of G."""
    return control.balred(G, orders=G.nstates)


def baltransform(G):
    """Return transformation matrix T making G balanced.
    Algorithm from Laub, Heath, Paige, Ward (1987)."""
    Wo = control.gram(G, "o")
    Wc = control.gram(G, "c")
    chol = np.linalg.cholesky
    Lo = chol(Wo)
    Lc = chol(Wc)
    uu, sv, vvh = np.linalg.svd(Lo.T @ Lc)
    SS = np.diag(sv)
    T = Lc @ vvh.T @ np.linalg.inv(chol(SS))
    return np.asarray(T)


def reduceorder(G):
    """Reduce order of G by balanced realization + minreal."""
    return control.minreal(balreal(G))


def sys_hsv(sys):
    """Compute Hankel Singular Values of sys (supports unstable systems)."""
    try:
        from slycot import ab09md
    except ImportError:
        raise ImportError("slycot subroutine ab09md not found")
    n = np.size(sys.A, 0)
    m = np.size(sys.B, 1)
    p = np.size(sys.C, 0)
    _, _, _, _, _, hsv = ab09md(
        "C", "B", "N", n, m, p, sys.A, sys.B, sys.C, alpha=0.0, nr=n, tol=0.0
    )
    hsv[hsv == 0.0] = np.inf
    return np.flip(np.sort(hsv))


def balred_rel(sys, hsv_threshold, method="truncate"):
    """Balanced reduction based on relative Hankel Singular Values threshold."""
    if method not in ("truncate", "matchdc"):
        raise ValueError("method must be 'truncate' or 'matchdc'")
    try:
        if method == "truncate":
            from slycot import ab09ad, ab09md
        else:
            from slycot import ab09nd
    except ImportError:
        raise ImportError("slycot not found")

    n = np.size(sys.A, 0)
    m = np.size(sys.B, 1)
    p = np.size(sys.C, 0)
    hsv = sys_hsv(sys)
    elim = (hsv / np.max(hsv[np.isfinite(hsv)])) < hsv_threshold
    nr = n - np.sum(elim)

    if method == "truncate":
        Dr = sys.D
        if np.any(np.linalg.eigvals(sys.A).real >= 0.0):
            Nr, Ar, Br, Cr, Ns, _ = ab09md(
                "C", "B", "N", n, m, p, sys.A, sys.B, sys.C,
                alpha=0.0, nr=nr, tol=0.0,
            )
        else:
            Nr, Ar, Br, Cr, _ = ab09ad(
                "C", "B", "N", n, m, p, sys.A, sys.B, sys.C, nr=nr, tol=0.0
            )
    else:
        Nr, Ar, Br, Cr, Dr, Ns, _ = ab09nd(
            "C", "B", "N", n, m, p, sys.A, sys.B, sys.C, sys.D,
            alpha=0.0, nr=nr, tol1=0.0, tol2=0.0,
        )

    return control.StateSpace(Ar, Br, Cr, Dr), hsv, nr


# --- Controller parametrization via residues ---

def controller_residues(real_c=[], real_p=[], cplx_c=[], cplx_p=[]):
    """Construct K = sum(real_c/(s-real_p)) + sum(cplx pairs) in StateSpace form."""
    K = control.StateSpace([], [], [], 0)

    def ss1(c, p):
        return control.StateSpace(p, c, 1, 0)

    for c, p in zip(real_c, real_p):
        K += ss1(c, p)

    re, im = np.real, np.imag

    def ss2(c, p):
        return control.StateSpace(
            np.array([[2 * re(p), -(np.abs(p) ** 2)], [1, 0]]),
            np.array([[2 * (re(p) * re(c) - im(p) * im(c)), 2 * re(c)]]).T,
            [0, 1],
            [0],
        )

    for c, p in zip(cplx_c, cplx_p):
        K += ss2(c, p)
    return K


def controller_residues_getidx(n_real, n_cplx):
    """Return index slices into theta vector for controller_residues_wrapper."""
    idx = np.arange(0, 2 * n_real + 4 * n_cplx)
    return (
        idx[0:n_real],
        idx[n_real : 2 * n_real],
        idx[2 * n_real : 2 * n_real + n_cplx],
        idx[2 * n_real + n_cplx : 2 * n_real + 2 * n_cplx],
        idx[2 * n_real + 2 * n_cplx : 2 * n_real + 3 * n_cplx],
        idx[2 * n_real + 3 * n_cplx :],
    )


def controller_residues_wrapper(theta, n_real, n_cplx):
    """Build controller from flat theta = [real_c, real_p, cplx_c_re, cplx_c_im, cplx_p_re, cplx_p_im]."""
    assert len(theta) == 2 * n_real + 4 * n_cplx
    rc_i, rp_i, cc_re_i, cc_im_i, cp_re_i, cp_im_i = controller_residues_getidx(n_real, n_cplx)
    return controller_residues(
        theta[rc_i], theta[rp_i],
        theta[cc_re_i] + 1j * theta[cc_im_i],
        theta[cp_re_i] + 1j * theta[cp_im_i],
    )


# --- Slow-fast decomposition ---

def slowfast(G, wlim):
    """Slow-fast decomposition: G = Gslow + Gfast, split at wlim."""
    Gtf = control.ss2tf(G)
    r, p, k = ss.residue(Gtf.num[0][0], Gtf.den[0][0])

    if k.shape == (0,):
        k = 0

    wn = np.abs(p)
    idx_slow = np.where(wn < wlim)[0]
    idx_fast = np.where(wn >= wlim)[0]
    s = control.tf("s")

    Gslow = control.tf(0, 1)
    for ii in idx_slow:
        Gslow += r[ii] / (s - p[ii])
    Gslow = control.tf2ss(make_tf_real(Gslow))

    Gfast = control.tf(k, 1)
    for ii in idx_fast:
        Gfast += r[ii] / (s - p[ii])
    Gfast = control.tf2ss(make_tf_real(Gfast))

    return Gslow, Gfast


def make_tf_real(G):
    """Ensure TF G has real num/den coefficients."""
    return control.tf(np.real(G.num[0][0]), np.real(G.den[0][0]))


# --- Controller conditioning ---

def condswitch(ur, yr, K, dt, w_y, w_u, w_decay):
    """Controller conditioning for switching (Paxman PhD).
    Finds initial state compatible with offline controller given past signals."""
    Kd = control.c2d(K, dt, "tustin")
    A, B, C, D = control.ssdata(Kd)
    r = len(ur)
    Ur = ur.reshape(-1)
    Yr = yr.reshape(-1)
    n = Kd.nstates

    invA = np.linalg.inv(A)
    Gamma_r = np.zeros((r, n))
    Gamma_r[0, :] = C @ invA
    for ii in range(r - 1):
        Gamma_r[ii + 1, :] = Gamma_r[ii, :] @ invA

    Tr = np.zeros((r, r))
    Tr0 = np.zeros((r, 1))
    for ii in range(r):
        Tr0[ii] = C @ invA ** (ii + 1) @ B
    Tr0[0] += np.asarray(-D).ravel()

    Tr[:, 0] = Tr0.ravel()
    for jj in range(1, r):
        Tr[:, jj] = np.vstack((np.zeros((jj, 1)), Tr0[:-jj])).ravel()

    W_decay = np.diag(w_decay ** np.flip(np.arange(0, r)))
    W = la.block_diag(w_y * np.eye(r), w_u * np.eye(r))
    W *= la.block_diag(W_decay, W_decay)

    Asol = W @ np.block(
        [[-Tr, Gamma_r], [np.eye(Tr.shape[0]), np.zeros(Gamma_r.shape)]]
    )
    bsol = W @ np.hstack((Ur, Yr))
    sol = np.linalg.lstsq(Asol, bsol, rcond=None)[0]

    xn = sol[-n:]
    yhat = sol[:r]
    uhat = Gamma_r @ xn - Tr @ yhat
    return xn, yhat, uhat


# --- Misc ---

def compare_controllers(K1, K2):
    """Compare two controllers by hinfnorm difference and DC gain difference."""
    print("Comparing controllers...")
    print("\t hinfnorm diff = ", norm(K1) - norm(K2))
    print("\t dcgains diff =", control.dcgain(K1) - control.dcgain(K2))


def export_controller(filename, K):
    """Export frequency response and matrices of K to .mat file."""
    mag, phase, w = control.bode(K, plot=False)
    A, B, C, D = ssdata(K)
    sio.savemat(filename, dict(mag=mag, phase=phase, w=w, A=A, B=B, C=C, D=D))
    print("Exported controller to file: ", filename)


def test_controller_residues(n_real, n_cplx):
    """Test controller_residues and controller_residues_wrapper for consistency."""
    rand = np.random.rand

    def rand_in(bnd, n1):
        return bnd[0] + (bnd[1] - bnd[0]) * rand(n1)

    real_c = rand_in([-10, 10], n_real)
    real_p = rand_in([-10, 0], n_real)
    cplx_c = rand_in([-10, 10], n_cplx) + 1j * rand_in([-10, 10], n_cplx)
    cplx_p = rand_in([-10, 0], n_cplx) + 1j * rand_in([0, 10], n_cplx)

    K1 = controller_residues(real_c, real_p, cplx_c, cplx_p)
    theta = np.array(
        list(np.real(real_c)) + list(real_p)
        + list(np.real(cplx_c)) + list(np.imag(cplx_c))
        + list(np.real(cplx_p)) + list(np.imag(cplx_p))
    )
    K2 = controller_residues_wrapper(theta, n_real, n_cplx)
    compare_controllers(K1, K2)
    return K1, K2
