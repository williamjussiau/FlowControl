import dolfin
from dolfin import dot, inner, div, curl, sym, grad, nabla_grad, dx
import utils_flowsolver as flu

import numpy as np
import scipy.sparse as spr
import sympy as sp

import logging

logger = logging.getLogger(__name__)


def check_mass_matrix(self, up=0, vq=0, random=True):
    """Given two vectors u, v (Functions on self.W),
    compute assemble(dot(u,v)*dx) v.s. u.local().T @ Q @ v.local()
    The result should be the same"""
    if random:
        logger.info("Creating random vectors")

        def createrandomfun():
            up = dolfin.Function(self.W)
            up.vector().set_local((np.random.randn(self.W.dim(), 1)))
            up.vector().apply("insert")
            return up

        up = createrandomfun()
        vq = createrandomfun()

    fa = dolfin.FunctionAssigner([self.V, self.P], self.W)
    u = dolfin.Function(self.V)  # velocity only
    p = dolfin.Function(self.P)
    v = dolfin.Function(self.V)
    q = dolfin.Function(self.P)
    fa.assign([u, p], up)
    fa.assign([v, q], vq)

    # True integral of velocities
    d1 = dolfin.assemble(dot(u, v) * self.dx)

    # Discretized dot product (scipy)
    Q = self.get_mass_matrix(sparse=True)
    d2 = up.vector().get_local().T @ Q @ vq.vector().get_local()
    ## Note: u.T @ Qv = (Qv).T @ u
    # d2 = (Q @ v.vector().get_local()).T @ u.vector().get_local()

    # Discretized dot product (petsc)
    QQ = self.get_mass_matrix(sparse=False)
    uu = dolfin.Vector(up.vector())
    vv = dolfin.Vector(vq.vector())
    ww = dolfin.Vector(up.vector())  # intermediate result
    QQ.mult(vv, ww)  # ww = QQ*vv
    d3 = uu.inner(ww)

    return {"integral": d1, "dot_scipy": d2, "dot_petsc": d3}


def get_block_identity(self, sparse=False):
    """Compute the block-identity associated to
    the time-continuous, space-continuous formulation:
    E*dot(x) = A*x >>> E = blk(I, I, 0), 0 being on p dofs"""
    dof_idx = flu.get_subspace_dofs(self.W)
    sz = self.W.dim()
    diagE = np.zeros(sz)
    diagE[np.hstack([dof_idx[kk] for kk in ["u", "v"]])] = 1.0
    E = spr.diags(diagE, 0)
    if sparse:
        return E
    # cast
    return flu.sparse_to_petscmat(E)


def get_mass_matrix(self, sparse=False, volume=True, uvp=False):
    """Compute the mass matrix associated to
    spatial discretization"""
    logger.info("Computing mass matrix Q...")
    up = dolfin.TrialFunction(self.W)
    vq = dolfin.TestFunction(self.W)

    M = dolfin.PETScMatrix()
    # volume integral or surface integral (unused)
    dOmega = self.dx if volume else self.ds

    mf = sum([up[i] * vq[i] for i in range(2 + uvp)]) * dOmega  # sum u, v but not p
    dolfin.assemble(mf, tensor=M)

    if sparse:
        return flu.dense_to_sparse(M)
    return M


def get_matrices_lifting(self, A, C, Q):
    """Return matrices A, B, C, Q resulting form lifting transform (Barbagallo et al. 2009)
    See get_Hw_lifting for details"""
    # Steady field with rho=1: S1
    logger.info("Computing steady actuated field...")
    self.actuator_expression.ampl = 1.0
    S1 = self.compute_steady_state_newton()
    S1v = S1.vector()

    # Q*S1 (as vector)
    sz = self.W.dim()
    QS1v = dolfin.Vector(S1v.copy())
    QS1v.set_local(
        np.zeros(
            sz,
        )
    )
    QS1v.apply("insert")
    Q.mult(S1v, QS1v)  # QS1v = Q * S1v

    # Bl = [Q*S1; -1]
    Bl = np.hstack((QS1v.get_local(), -1))  # stack -1
    Bl = np.atleast_2d(Bl).T  # as column

    # Cl = [C, 0]
    Cl = np.hstack((C, np.atleast_2d(0)))

    # Ql = diag(Q, 1)
    Qsp = flu.dense_to_sparse(Q)
    Qlsp = flu.spr.block_diag((Qsp, 1))

    # Al = diag(A, 0)
    Asp = flu.dense_to_sparse(A)
    Alsp = flu.spr.block_diag((Asp, 0))

    return Alsp, Bl, Cl, Qlsp


def get_Dxy(self):
    """Get derivation matrices Dx, Dy in V space
    such that Dx*u = u.dx(0), Dy*u = u.dx(1)"""
    u = dolfin.TrialFunction(self.V)
    ut = dolfin.TestFunction(self.V)
    Dx = dolfin.assemble(inner(u.dx(0), ut) * self.dx)
    Dy = dolfin.assemble(inner(u.dx(1), ut) * self.dx)
    return Dx, Dy


def compute_vorticity(self, u=None):
    """Compute vorticity field of given velocity field u"""
    if u is None:
        u = self.u_
    # should probably project on space of order n-1 --> self.P
    vorticity = flu.projectm(curl(u), V=self.V.sub(0).collapse())
    return vorticity


def compute_divergence(self, u=None):
    """Compute divergence field of given velocity field u"""
    if u is None:
        u = self.u_
    divergence = flu.projectm(div(u), self.P)
    return divergence


def stress_tensor(nu, u, p):
    """Compute stress tensor (for lift & drag)"""
    return 2.0 * nu * (sym(grad(u))) - p * dolfin.Identity(p.geometric_dimension())


def compute_steady_state_newton(fs, max_iter=25, initial_guess=None):
    """Compute steady state with built-in nonlinear solver (Newton method)
    initial_guess is a (u,p)_0"""
    fs.make_form_mixed_steady(initial_guess=initial_guess)
    # if initial_guess is None:
    #    print('- Newton solver without initial guess')
    up_ = fs.up_
    # u_, p_ = self.u_, self.p_
    # Solver param
    nl_solver_param = {
        "newton_solver": {
            "linear_solver": "mumps",
            "preconditioner": "default",
            "maximum_iterations": max_iter,
            "report": bool(fs.verbose),
        }
    }
    dolfin.solve(fs.F0 == 0, up_, fs.bc["bcu"], solver_parameters=nl_solver_param)
    # Return
    return up_


def compute_steady_state_picard(fs, max_iter=10, tol=1e-14):
    """Compute steady state with fixed-point iteration
    Should have a larger convergence radius than Newton method
    if initialization is bad in Newton method (and it is)
    TODO: residual not 0 if u_ctrl not 0 (see bc probably)"""
    fs.make_form_mixed_steady()
    iRe = dolfin.Constant(1 / fs.Re)

    # for residual computation
    bcu_inlet0 = dolfin.DirichletBC(
        fs.W.sub(0),
        dolfin.Constant((0, 0)),
        fs.boundaries.loc["inlet"].subdomain,
    )
    bcu0 = fs.bc["bcu"] + [bcu_inlet0]

    # define forms
    up0 = dolfin.Function(fs.W)
    up1 = dolfin.Function(fs.W)

    u, p = dolfin.TrialFunctions(fs.W)
    v, q = dolfin.TestFunctions(fs.W)

    class initial_condition(dolfin.UserExpression):
        def eval(self, value, x):
            value[0] = 1.0
            value[1] = 0.0
            value[2] = 0.0

        def value_shape(self):
            return (3,)

    up0.interpolate(initial_condition())
    u0 = dolfin.as_vector((up0[0], up0[1]))

    ap = (
        dot(dot(u0, nabla_grad(u)), v) * dx
        + iRe * inner(nabla_grad(u), nabla_grad(v)) * dx
        - p * div(v) * dx
        - q * div(u) * dx
    )  # steady dolfin.lhs
    Lp = (
        dolfin.Constant(0) * inner(u0, v) * dx + dolfin.Constant(0) * q * dx
    )  # zero dolfin.rhs
    bp = dolfin.assemble(Lp)

    solverp = dolfin.LUSolver("mumps")
    ndof = fs.W.dim()

    for i in range(max_iter):
        Ap = dolfin.assemble(ap)
        [bc.apply(Ap, bp) for bc in fs.bc["bcu"]]

        solverp.solve(Ap, up1.vector(), bp)

        up0.assign(up1)
        u, p = up1.split()

        # show_max(u, 'u')
        res = dolfin.assemble(dolfin.action(ap, up1))
        [bc.apply(res) for bc in bcu0]
        res_norm = dolfin.norm(res) / dolfin.sqrt(ndof)
        if fs.verbose:
            logger.info(
                "Picard iteration: {0}/{1}, residual: {2}".format(
                    i + 1, max_iter, res_norm
                )
            )
        if res_norm < tol:
            if fs.verbose:
                logger.info("Residual norm lower than tolerance {0}".format(tol))
            break

    return up1


def get_div0_u(fs, xloc, yloc, size):
    """Create velocity field with zero divergence"""
    # V = self.V
    P = fs.P

    # Define courant function
    xm, ym = sp.symbols("x[0], x[1]")
    rr = (xm - xloc) ** 2 + (ym - yloc) ** 2
    fpsi = 0.25 * sp.exp(-1 / 2 * rr / size**2)
    # Piecewise does not work too well
    # fpsi = sp.Piecewise(   (sp.exp(-1/2 * rr / sigm**2),
    # rr <= nsig**2 * sigm**2), (0, True) )
    dfx = fpsi.diff(xm, 1)
    dfy = fpsi.diff(ym, 1)

    # Take derivatives
    # psi = dolfin.Expression(sp.ccode(fpsi), element=V.ufl_element())
    dfx_expr = dolfin.Expression(sp.ccode(dfx), element=P.ufl_element())
    dfy_expr = dolfin.Expression(sp.ccode(dfy), element=P.ufl_element())

    # Check
    # psiproj = flu.projectm(psi, P)
    # flu.write_xdmf('psi.xdmf', psiproj, 'psi')
    # flu.write_xdmf('psi_dx.xdmf', flu.projectm(dfx_expr, P), 'psidx')
    # flu.write_xdmf('psi_dy.xdmf', flu.projectm(dfy_expr, P), 'psidy')

    # Make velocity field
    upsi = flu.projectm(dolfin.as_vector([dfy_expr, -dfx_expr]), fs.V)
    return upsi


def get_div0_u_random(fs, sigma=0.1, seed=0):
    """Create random velocity field with zero divergence"""
    # CG2 scalar
    P2 = fs.V.sub(0).collapse()

    # Make scalar potential field in CG2 (scalar)
    a0 = dolfin.Function(P2)
    np.random.seed(seed)
    a0.vector()[:] += sigma * np.random.randn(a0.vector()[:].shape[0])

    # Take curl, then by definition div(u0)=div(curl(a0))=0
    Ve = dolfin.VectorElement("CG", fs.mesh.ufl_cell(), 1)
    V1 = dolfin.FunctionSpace(fs.mesh, Ve)

    u0 = flu.projectm(curl(a0), V1)

    ##divu0 = flu.projectm(div(u0), self.P)
    return u0


class localized_perturbation_u(dolfin.UserExpression):
    """Perturbation localized in disk
    Use: u = dolfin.interpolate(localized_perturbation_u(), self.V)
    or something like that"""

    def eval(self, value, x):
        if (x[0] - -2.5) ** 2 + (x[1] - 0.1) ** 2 <= 1:
            value[0] = 0.05
            value[1] = 0.05
        else:
            value[0] = 0
            value[1] = 0

    def value_shape(self):
        return (2,)


def compute_steady_state(fs, method="newton", u_ctrl=0.0, **kwargs):
    """Compute flow steady state with given steady control"""

    # Save old control value, just in case
    actuation_ampl_old = fs.actuator_expression.ampl
    # Set control value to prescribed u_ctrl
    fs.actuator_expression.ampl = u_ctrl

    # If start is zero (i.e. not restart): compute
    # Note : could add a flag 'compute_steady_state' to compute or read...
    if fs.Tstart == 0:  # and compute_steady_state
        # Solve
        if method == "newton":
            up0 = compute_steady_state_newton(fs, **kwargs)
        else:
            up0 = compute_steady_state_picard(fs, **kwargs)

        # assign up0, u0, p0 and write
        fa_W2VP = dolfin.FunctionAssigner([fs.V, fs.P], fs.W)
        u0 = dolfin.Function(fs.V)
        p0 = dolfin.Function(fs.P)
        fa_W2VP.assign([u0, p0], up0)

        # Save steady state
        if fs.save_every:
            flu.write_xdmf(
                fs.paths["u0"],
                u0,
                "u",
                time_step=0.0,
                append=False,
                write_mesh=True,
            )
            flu.write_xdmf(
                fs.paths["p0"],
                p0,
                "p",
                time_step=0.0,
                append=False,
                write_mesh=True,
            )
        if fs.verbose:
            logger.info("Stored base flow in: %s", fs.savedir0)

        fs.y_meas_steady = fs.make_measurement(mixed_field=up0)

    # If start is not zero: read steady state (should exist - should check though...)
    else:
        u0, p0, up0 = load_steady_state(fs, assign=True)

    # Compute lift & drag
    cl, cd = fs.compute_force_coefficients(u0, p0)
    # cl, cd = 0, 1
    if fs.verbose:
        logger.info("Lift coefficient is: cl = %f", cl)
        logger.info("Drag coefficient is: cd = %f", cd)

    # Set old actuator amplitude
    fs.actuator_expression.ampl = actuation_ampl_old

    # assign steady state
    fs.up0 = up0
    fs.u0 = u0
    fs.p0 = p0
    # assign steady cl, cd
    fs.cl0 = cl
    fs.cd0 = cd
    # assign steady energy
    fs.Eb = (
        1 / 2 * dolfin.norm(u0, norm_type="L2", mesh=fs.mesh) ** 2
    )  # same as <up, Q@up>


def load_steady_state(fs, assign=True):  # TODO move to utils???
    u0 = dolfin.Function(fs.V)
    p0 = dolfin.Function(fs.P)
    flu.read_xdmf(fs.paths["u0"], u0, "u")
    flu.read_xdmf(fs.paths["p0"], p0, "p")

    # Assign u0, p0 >>> up0
    fa_VP2W = dolfin.FunctionAssigner(fs.W, [fs.V, fs.P])
    up0 = dolfin.Function(fs.W)
    fa_VP2W.assign(up0, [u0, p0])

    if assign:
        fs.u0 = u0  # full field (u+upert)
        fs.p0 = p0
        fs.up0 = up0
        fs.y_meas_steady = fs.make_measurement(mixed_field=up0)

        # assign steady energy
        fs.Eb = (
            1 / 2 * dolfin.norm(u0, norm_type="L2", mesh=fs.mesh) ** 2
        )  # same as <up, Q@up>
    return u0, p0, up0


def print_progress(fs, runtime):
    """Single line to print progress"""
    logger.info(
        "--- iter: %5d/%5d --- time: %3.3f/%3.2f --- elapsed %5.5f ---"
        % (fs.iter, fs.num_steps, fs.t, fs.Tf + fs.Tstart, runtime)
    )


def make_y_dataframe_column_name(sensor_nr):
    """Return column names of different measurements y_meas_i"""
    return ["y_meas_" + str(i + 1) for i in range(sensor_nr)]


def assign_measurement_to_dataframe(df, y_meas, index, sensor_nr):
    """Assign measurement (array y_meas) to DataFrame at index
    Essentially convert array (y_meas) to separate columns (y_meas_i)"""
    y_meas_str = make_y_dataframe_column_name(sensor_nr)
    for i_meas, name_meas in enumerate(y_meas_str):
        df.loc[index, name_meas] = y_meas[i_meas]
