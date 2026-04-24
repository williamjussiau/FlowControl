"""LTI control utilities: state-space I/O, Youla parametrization,
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
    return control.StateSpace([], [], [], 1)


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
    A, B, C, D = ssdata(syslist[0])
    for sys in syslist[1:]:
        A = la.block_diag(A, sys.A)
        B = np.vstack((B, sys.B))
        C = la.block_diag(C, sys.C)
        D = np.vstack((D, sys.D))
    return control.StateSpace(A, B, C, D)


def ss_hstack_list(syslist):
    """Same as ss_hstack but with input list"""
    A, B, C, D = ssdata(syslist[0])
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


def show_ss(sys):
    """Print StateSpace matrices to stdout"""
    for mat in ssdata(sys):
        print(mat)
        print("-" * 10)


# --- Stability and norms ---


def isstable(CL):
    """Return True if all poles of CL have strictly negative real part"""
    return np.all(np.real(control.poles(CL)) < 0)


def isstablecl(G, K0, sign=+1):
    """Return True if feedback(G, K0, sign) is stable"""
    return isstable(control.feedback(G, K0, sign=sign))


def norm(G, p=np.inf):
    """Compute H2 or H-infinity norm of G. Returns inf for unstable systems.
    Delegates to control.norm (H2) and control.linfnorm (H-inf)."""
    if p not in (2, np.inf):
        raise ValueError("p must be 2 or np.inf")
    if not isstable(G):
        return np.inf
    if p == 2:
        return control.norm(G, 2)
    return control.linfnorm(G)[0]


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
    theta = np.atleast_1d(np.asarray(theta, float))
    N = len(theta)
    Gstab = G.feedback(other=K0, sign=+1)
    Psi = build_block_Psi(Gstab)

    Qf = basis_laguerre_canonical_ss(p, N)
    Qf1 = cmat.append(1, Qf)
    Psif = Psi * Qf1

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


def youla_lqg(G, Qx, Ru, Qw, Rv, Q):
    """Youla controller in LQG form"""
    J = youla_lqg_lftmat(G, Qx, Ru, Qw, Rv)
    return J.lft(Q)


def youla_lqg_lftmat(G, Qx, Ru, Qw, Rv):
    """Return StateSpace J to be LFTed with Q for Youla LQG parametrization."""
    _, B, C, D = ssdata(G)
    p, m = D.shape
    Im = np.eye(m)
    Ip = np.eye(p)
    Klqg, F, L = lqg_regulator(G, Qx, Ru, Qw, Rv)
    return control.StateSpace(
        Klqg.A,
        np.hstack((Klqg.B, B + L @ D)),
        np.vstack((Klqg.C, -C - D @ F)),
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


def lqg_regulator(G, Qx, Ru, Qw, Rv):
    """Make LQG regulator. Returns (Klqg, F, L).
    Qx/Ru: scalar state/input cost (LQR); Qw/Rv: scalar process/measurement noise covariances.
    L uses the sign convention x_dot = (A + LC)x + ..., so L = -L_kalman."""
    A, B, C, D = ssdata(G)
    n = A.shape[0]
    p, m = D.shape
    F = np.array(-control.lqr(A, B, Qx * np.eye(n), Ru * np.eye(m))[0])
    L_kalman, _, _ = control.lqe(A, np.eye(n), C, Qw * np.eye(n), Rv * np.eye(p))
    L = -np.asarray(L_kalman)
    Klqg = control.StateSpace(A + B @ F + L @ C + L @ D @ F, -L, F, 0)
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
    theta = np.atleast_1d(np.asarray(theta, float))
    return np.dot(basis_laguerre_canonical(p, len(theta)), theta)


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
    theta = np.atleast_1d(np.asarray(theta, float))
    Phi = basis_laguerre_canonical_ss(p, len(theta))
    return Phi * np.atleast_2d(theta).T


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
    nsv = min(p, m)
    s_vals = s[:nsv]
    diag_vec = np.hstack((1 / np.sqrt(1 + s_vals**2), np.ones(m - nsv)))
    Z = v @ np.diag(diag_vec) @ vh

    F = -K[:m, :]
    Amn = A + B @ F
    Bmn = B @ Z
    Cmn = np.vstack((F, C + D @ F))
    Dmn = np.vstack((Z, D @ Z))
    FACT = control.StateSpace(Amn, Bmn, Cmn, Dmn)
    Mr = control.StateSpace(Amn, Bmn, Cmn[:m, :], Dmn[:m, :])
    Nr = control.StateSpace(Amn, Bmn, Cmn[m : m + p, :], Dmn[m : m + p, :])
    return FACT, Mr, Nr


def lncf(G):
    """Left normalized coprime factorization: G = inv(Ml) * Nl."""
    FACT = rncf(ss_transpose(G))[0]
    FACT = ss_transpose(FACT)
    Amn, Bmn, Cmn, Dmn = ssdata(FACT)
    # Dmn has shape (p_G, p_G + m_G) after transposition; first p_G cols = Ml, rest = Nl
    ncols_Ml = G.noutputs
    Ml = control.StateSpace(Amn, Bmn[:, :ncols_Ml], Cmn, Dmn[:, :ncols_Ml])
    Nl = control.StateSpace(Amn, Bmn[:, ncols_Ml:], Cmn, Dmn[:, ncols_Ml:])
    return FACT, Ml, Nl


# --- Balanced reduction ---


def balreal(G):
    """Return balanced realization of G."""
    T = baltransform(G)
    A, B, C, D = ssdata(G)
    Ti = np.linalg.inv(T)
    return control.StateSpace(Ti @ A @ T, Ti @ B, C @ T, D)


def baltransform(G):
    """Return transformation matrix T making G balanced.
    Algorithm from Laub, Heath, Paige, Ward (1987)."""
    Wo = control.gram(G, "o")
    Wc = control.gram(G, "c")
    Lo = np.linalg.cholesky(Wo)
    Lc = np.linalg.cholesky(Wc)
    _, sv, vvh = np.linalg.svd(Lo.T @ Lc)
    T = Lc @ vvh.T @ np.diag(1 / np.sqrt(sv))
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
            _, Ar, Br, Cr, _, _ = ab09md(
                "C",
                "B",
                "N",
                n,
                m,
                p,
                sys.A,
                sys.B,
                sys.C,
                alpha=0.0,
                nr=nr,
                tol=0.0,
            )
        else:
            _, Ar, Br, Cr, _ = ab09ad(
                "C", "B", "N", n, m, p, sys.A, sys.B, sys.C, nr=nr, tol=0.0
            )
    else:
        _, Ar, Br, Cr, Dr, _, _ = ab09nd(
            "C",
            "B",
            "N",
            n,
            m,
            p,
            sys.A,
            sys.B,
            sys.C,
            sys.D,
            alpha=0.0,
            nr=nr,
            tol1=0.0,
            tol2=0.0,
        )

    return control.StateSpace(Ar, Br, Cr, Dr), hsv, nr


# --- Controller parametrization via residues ---


def controller_residues(real_c=None, real_p=None, cplx_c=None, cplx_p=None):
    """Construct K = sum(real_c/(s-real_p)) + sum(cplx pairs) in StateSpace form."""
    if real_c is None:
        real_c = []
    if real_p is None:
        real_p = []
    if cplx_c is None:
        cplx_c = []
    if cplx_p is None:
        cplx_p = []
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
    if len(theta) != 2 * n_real + 4 * n_cplx:
        expected = 2 * n_real + 4 * n_cplx
        raise ValueError(
            f"theta length {len(theta)} != 2*n_real + 4*n_cplx = {expected}"
        )
    rc_i, rp_i, cc_re_i, cc_im_i, cp_re_i, cp_im_i = controller_residues_getidx(
        n_real, n_cplx
    )
    return controller_residues(
        theta[rc_i],
        theta[rp_i],
        theta[cc_re_i] + 1j * theta[cc_im_i],
        theta[cp_re_i] + 1j * theta[cp_im_i],
    )


# --- Slow-fast decomposition ---


def slowfast(G, wlim):
    """Slow-fast decomposition: G = Gslow + Gfast, split at wlim. SISO only."""
    if G.ninputs != 1 or G.noutputs != 1:
        raise ValueError("slowfast: SISO systems only")
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
    A, B, C, D = ssdata(Kd)
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
        Tr0[ii] = C @ np.linalg.matrix_power(invA, ii + 1) @ B
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


# TODO: move test_controller_residues to the test suite
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
        list(np.real(real_c))
        + list(real_p)
        + list(np.real(cplx_c))
        + list(np.imag(cplx_c))
        + list(np.real(cplx_p))
        + list(np.imag(cplx_p))
    )
    K2 = controller_residues_wrapper(theta, n_real, n_cplx)
    compare_controllers(K1, K2)
    return K1, K2
