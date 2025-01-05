"""
Incompressible Navier-Stokes equations

  u' + u . nabla(u)) - div(sigma(u, p)) = f
                                 div(u) = 0
Equations were made non-dimensional
----------------------------------------------------------------------
"""

# from __future__ import print_function

import FlowSolver
import dolfin
from dolfin import dot, inner, nabla_grad, div, dx
import numpy as np
import time
import pandas as pd
import FlowSolverParameters

import logging

from pathlib import Path

import utils_flowsolver as flu
import utils_extract as flu2


# LOG
dolfin.set_log_level(dolfin.LogLevel.INFO)  # DEBUG TRACE PROGRESS INFO
logger = logging.getLogger(__name__)
FORMAT = "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)


class CylinderFlowSolver(FlowSolver.FlowSolver):
    """Base class for calculating flow
    Is instantiated with several structures (dicts) containing parameters
    See method .step and main for time-stepping (possibly actuated)
    Contain methods for frequency-response computation"""

    def __init__(self, **kwargs):  # redundant def here
        super().__init__(**kwargs)

    # Abstract methods
    def make_boundaries(self):
        """Define boundaries (inlet, outlet, walls, cylinder, actuator)"""
        MESH_TOL = dolfin.DOLFIN_EPS
        # Define as compiled subdomains
        ## Inlet
        inlet = dolfin.CompiledSubDomain(
            "on_boundary && \
                near(x[0], xinfa, MESH_TOL)",
            xinfa=self.params_mesh.xinfa,
            MESH_TOL=MESH_TOL,
        )
        ## Outlet
        outlet = dolfin.CompiledSubDomain(
            "on_boundary && \
                near(x[0], xinf, MESH_TOL)",
            xinf=self.params_mesh.xinf,
            MESH_TOL=MESH_TOL,
        )
        ## Walls
        walls = dolfin.CompiledSubDomain(
            "on_boundary && \
                (near(x[1], -yinf, MESH_TOL) ||   \
                 near(x[1], yinf, MESH_TOL))",
            yinf=self.params_mesh.yinf,
            MESH_TOL=MESH_TOL,
        )

        ## Cylinder
        delta = self.params_flow.actuator_angular_size * dolfin.pi / 180
        theta_tol = 1 * dolfin.pi / 180

        # Compiled subdomains (str)
        # = increased speed but decreased readability

        def between_cpp(x, xmin, xmax):
            return f"{x}>={xmin} && {x}<={xmax}"

        close_to_cylinder_cpp = (
            between_cpp("x[0]", "-d/2", "d/2")
            + " && "
            + between_cpp("x[1]", "-d/2", "d/2")
        )
        theta_cpp = "atan2(x[1], x[0])"
        cone_up_cpp = between_cpp(
            theta_cpp + "-pi/2", "-delta/2 - theta_tol", "delta/2 + theta_tol"
        )
        cone_lo_cpp = between_cpp(
            theta_cpp + "+pi/2", "-delta/2 - theta_tol", "delta/2 + theta_tol"
        )
        cone_ri_cpp = between_cpp(
            theta_cpp, "-pi/2+delta/2 - theta_tol", "pi/2-delta/2 + theta_tol"
        )
        cone_le_cpp = (
            "("
            + between_cpp(theta_cpp, "-pi", "-pi/2-delta/2 + theta_tol")
            + ")"
            + " || "
            + "("
            + between_cpp(theta_cpp, "pi/2+delta/2 - theta_tol", "pi")
            + ")"
        )

        cylinder = dolfin.CompiledSubDomain(
            "on_boundary"
            + " && "
            + close_to_cylinder_cpp
            + " && "
            + "("
            + cone_le_cpp
            + " ||  "
            + cone_ri_cpp
            + ")",
            d=self.params_flow.d,
            delta=delta,
            theta_tol=theta_tol,
        )
        actuator_up = dolfin.CompiledSubDomain(
            "on_boundary" + " && " + close_to_cylinder_cpp + " && " + cone_up_cpp,
            d=self.params_flow.d,
            delta=delta,
            theta_tol=theta_tol,
        )
        actuator_lo = dolfin.CompiledSubDomain(
            "on_boundary" + " && " + close_to_cylinder_cpp + " && " + cone_lo_cpp,
            d=self.params_flow.d,
            delta=delta,
            theta_tol=theta_tol,
        )

        # assign boundaries as pd.DataFrame
        boundaries_names = [
            "inlet",
            "outlet",
            "walls",
            "cylinder",
            "actuator_up",
            "actuator_lo",
        ]
        boundaries_df = pd.DataFrame(
            index=boundaries_names,
            data={
                "subdomain": [inlet, outlet, walls, cylinder, actuator_up, actuator_lo]
            },
        )
        self.actuator_angular_size_rad = delta
        return boundaries_df

    def make_bcs(self):
        """Define boundary conditions"""
        # create zeroBC for perturbation formulation
        bcu_inlet = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((0, 0)),
            self.boundaries.loc["inlet"].subdomain,
        )
        bcu_walls = dolfin.DirichletBC(
            self.W.sub(0).sub(1),
            dolfin.Constant(0),
            self.boundaries.loc["walls"].subdomain,
        )
        bcu_cylinder = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((0, 0)),
            self.boundaries.loc["cylinder"].subdomain,
        )
        bcu_actuation_up = dolfin.DirichletBC(
            self.W.sub(0),
            self.actuator_expression,
            self.boundaries.loc["actuator_up"].subdomain,
        )
        bcu_actuation_lo = dolfin.DirichletBC(
            self.W.sub(0),
            self.actuator_expression,
            self.boundaries.loc["actuator_lo"].subdomain,
        )
        bcu = [bcu_inlet, bcu_walls, bcu_cylinder, bcu_actuation_up, bcu_actuation_lo]

        return {"bcu": bcu, "bcp": []}  # log perturbation bcs

    def make_measurement(self, field=None, mixed_field=None):
        """Perform measurement"""
        ns = self.params_flow.sensor_nr
        y_meas = np.zeros((ns,))

        for isensor in range(ns):
            xs_i = self.params_flow.sensor_location[isensor]
            ts_i = self.params_flow.sensor_type[isensor]

            # no mixed field (u,v,p) is given
            if field is not None:
                idx_dim = 0 if ts_i == "u" else 1
                y_meas_i = flu.MpiUtils.peval(field, xs_i)[idx_dim]
            else:
                if mixed_field is None:
                    # depending on sensor type, eval attribute field
                    if ts_i == "u":
                        y_meas_i = flu.MpiUtils.peval(self.u_, xs_i)[0]
                    else:
                        if ts_i == "v":
                            y_meas_i = flu.MpiUtils.peval(self.u_, xs_i)[1]
                        else:  # sensor_type=='p':
                            y_meas_i = flu.MpiUtils.peval(self.p_, xs_i)
                else:
                    # some mixed field in W = (u, v, p) is given
                    # eval and take index corresponding to sensor
                    sensor_types = dict(u=0, v=1, p=2)
                    y_meas_i = flu.MpiUtils.peval(mixed_field, xs_i)[sensor_types[ts_i]]

            y_meas[isensor] = y_meas_i
        return y_meas

    def make_actuator(self):
        """Define actuator on boundary
        Could be defined as volume actuator some day"""
        # TODO
        # return actuator type (vol, bc)
        # + MIMO -> list
        L = self.params_flow.d / 2 * np.tan(self.actuator_angular_size_rad / 2)
        nsig = 2  # self.nsig_actuator
        actuator_bc = dolfin.Expression(
            ["0", "(x[0]>=L || x[0] <=-L) ? 0 : ampl*-1*(x[0]+L)*(x[0]-L) / (L*L)"],
            element=self.V.ufl_element(),
            ampl=1,
            L=L,
            den=(L / nsig) ** 2,
            nsig=nsig,
        )

        return actuator_bc

    # Steady state
    def compute_steady_state(self, method="newton", u_ctrl=0.0, **kwargs):
        """Overriding is useless, should do an additional method"""
        super().compute_steady_state(method, u_ctrl, **kwargs)
        # assign steady cl, cd
        cl, cd = self.compute_force_coefficients(self.U0, self.P0)

        self.cl0 = cl
        self.cd0 = cd
        if self.verbose:
            logger.info(f"Lift coefficient is: cl = {cl}")
            logger.info(f"Drag coefficient is: cd = {cd}")

    # Matrix computations
    def get_A(
        self, perturbations=True, shift=0.0, timeit=True, UP0=None
    ):  # TODO idk, merge with make_mixed_form?
        """Get state-space dynamic matrix A linearized around some field UP0"""
        logger.info("Computing jacobian A...")

        if timeit:
            t0 = time.time()

        Jac = dolfin.PETScMatrix()
        v, q = dolfin.TestFunctions(self.W)
        iRe = dolfin.Constant(1 / self.params_flow.Re)
        shift = dolfin.Constant(shift)

        if UP0 is None:
            UP_ = self.UP0  # base flow
        else:
            UP_ = UP0
        U_, p_ = UP_.split()

        if perturbations:  # perturbation equations linearized
            up = dolfin.TrialFunction(self.W)
            u, p = dolfin.split(up)
            dF0 = (
                -dot(dot(U_, nabla_grad(u)), v) * dx
                - dot(dot(u, nabla_grad(U_)), v) * dx
                - iRe * inner(nabla_grad(u), nabla_grad(v)) * dx
                + p * div(v) * dx
                + div(u) * q * dx
                - shift * dot(u, v) * dx
            )  # sum u, v but not p
            # create zeroBC for perturbation formulation
            bcu_inlet = dolfin.DirichletBC(
                self.W.sub(0),
                dolfin.Constant((0, 0)),
                self.boundaries.loc["inlet"].subdomain,
            )
            bcu_walls = dolfin.DirichletBC(
                self.W.sub(0).sub(1),
                dolfin.Constant(0),
                self.boundaries.loc["walls"].subdomain,
            )
            bcu_cylinder = dolfin.DirichletBC(
                self.W.sub(0),
                dolfin.Constant((0, 0)),
                self.boundaries.loc["cylinder"].subdomain,
            )
            bcu_actuation_up = dolfin.DirichletBC(
                self.W.sub(0),
                self.actuator_expression,
                self.boundaries.loc["actuator_up"].subdomain,
            )
            bcu_actuation_lo = dolfin.DirichletBC(
                self.W.sub(0),
                self.actuator_expression,
                self.boundaries.loc["actuator_lo"].subdomain,
            )
            bcu = [
                bcu_inlet,
                bcu_walls,
                bcu_cylinder,
                bcu_actuation_up,
                bcu_actuation_lo,
            ]
            self.actuator_expression.ampl = 0.0
            bcs = bcu
        else:
            F0 = (
                -dot(dot(U_, nabla_grad(U_)), v) * dx
                - iRe * inner(nabla_grad(U_), nabla_grad(v)) * dx
                + p_ * div(v) * dx
                + q * div(U_) * dx
                - shift * dot(U_, v) * dx
            )
            # prepare derivation
            du = dolfin.TrialFunction(self.W)
            dF0 = dolfin.derivative(F0, UP_, du=du)
            # import pdb
            # pdb.set_trace()
            ## shift
            # dF0 = dF0 - shift*dot(U_,v)*dx
            # bcs
            self.actuator_expression.ampl = 0.0
            bcs = self.bc["bcu"]

        dolfin.assemble(dF0, tensor=Jac)
        [bc.apply(Jac) for bc in bcs]

        if timeit:
            logger.info(f"Elapsed time: {time.time() - t0}")

        return Jac

    def get_B(self, export=False, timeit=True):  # TODO keep here
        """Get actuation matrix B"""
        logger.info("Computing actuation matrix B...")

        if timeit:
            t0 = time.time()

        # for an exponential actuator -> just evaluate actuator_exp on every coordinate, kinda?
        # for a boundary actuator -> evaluate actuator on boundary
        actuator_ampl_old = self.actuator_expression.ampl
        self.actuator_expression.ampl = 1.0

        # Method 1
        # restriction of actuation of boundary
        class RestrictFunction(dolfin.UserExpression):
            def __init__(self, boundary, fun, **kwargs):
                self.boundary = boundary
                self.fun = fun
                super(RestrictFunction, self).__init__(**kwargs)

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

        Bi = []
        for actuator_name in ["actuator_up", "actuator_lo"]:
            actuator_restricted = RestrictFunction(
                boundary=self.boundaries.loc[actuator_name].subdomain,
                fun=self.actuator_expression,
            )
            actuator_restricted = dolfin.interpolate(actuator_restricted, self.W)
            # actuator_restricted = flu.projectm(actuator_restricted, self.W)
            Bi.append(actuator_restricted)

        # this is supposedly B
        B_all_actuator = flu.projectm(sum(Bi), self.W)
        # get vector
        B = B_all_actuator.vector().get_local()
        # remove very small values (should be 0 but are not)
        B = flu.dense_to_sparse(
            B, eliminate_zeros=True, eliminate_under=1e-14
        ).toarray()
        B = B.T  # vertical B

        if export:
            # fa = dolfin.FunctionAssigner([self.V, self.P], self.W)
            vv = dolfin.Function(self.V)
            # pp = dolfin.Function(self.P)
            ww = dolfin.Function(self.W)
            ww.assign(B_all_actuator)
            # fa.assign([vv, pp], ww)
            vv, pp = ww.split()
            flu.write_xdmf("B.xdmf", vv, "B")

        self.actuator_expression.ampl = actuator_ampl_old

        if timeit:
            logger.info(f"Elapsed time: {time.time() - t0}")

        return B

    def get_C(self, timeit=True, check=False):  # TODO keep here
        """Get measurement matrix C"""
        logger.info("Computing measurement matrix C...")

        if timeit:
            t0 = time.time()

        fspace = self.W
        uvp = dolfin.Function(fspace)
        uvp_vec = uvp.vector()
        dofmap = fspace.dofmap()

        ndof = fspace.dim()
        ns = self.params_flow.sensor_nr
        C = np.zeros((ns, ndof))

        idof_old = 0
        # xs = self.sensor_location
        # Iteratively put each DOF at 1
        # And evaluate measurement on said DOF
        for idof in dofmap.dofs():
            uvp_vec[idof] = 1
            if idof_old > 0:
                uvp_vec[idof_old] = 0
            idof_old = idof
            C[:, idof] = self.make_measurement(mixed_field=uvp)
            # mixed_field permits p sensor

        # check:
        if check:
            for i in range(ns):
                sensor_types = dict(u=0, v=1, p=2)
                logger.debug(
                    f"True probe: {self.up0(self.sensor_location[i])[sensor_types[self.sensor_type[0]]]}"
                )
                logger.debug(
                    f"\t with fun: {self.make_measurement(mixed_field=self.up0)}"
                )
                logger.debug(f"\t with C@x: {C[i] @ self.up0.vector().get_local()}")

        if timeit:
            logger.info(f"Elapsed time: {time.time() - t0}")

        return C

    # Additional, case-specific func
    def compute_force_coefficients(self, u, p):  # keep this one in here
        """Compute lift & drag coefficients"""
        nu = self.params_flow.uinf * self.params_flow.d / self.params_flow.Re

        sigma = flu2.stress_tensor(nu, u, p)
        Fo = -dot(sigma, self.n)

        # integration surfaces names
        surfaces_names = ["cylinder", "actuator_up", "actuator_lo"]
        # integration surfaces indices
        surfaces_idx = [self.boundaries.loc[nm].idx for nm in surfaces_names]

        # define drag & lift expressions
        # sum symbolic forces
        drag_sym = sum(
            [Fo[0] * self.ds(int(sfi)) for sfi in surfaces_idx]
        )  # (forced int)
        lift_sym = sum(
            [Fo[1] * self.ds(int(sfi)) for sfi in surfaces_idx]
        )  # (forced int)
        # integrate sum of symbolic forces
        lift = dolfin.assemble(lift_sym)
        drag = dolfin.assemble(drag_sym)

        # define force coefficients by normalizing
        cd = drag / (1 / 2 * self.params_flow.uinf**2 * self.params_flow.d)
        cl = lift / (1 / 2 * self.params_flow.uinf**2 * self.params_flow.d)
        return cl, cd


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
if __name__ == "__main__":
    t000 = time.time()
    cwd = Path(__file__).parent

    logger.info("Trying to instantiate FlowSolver...")

    params_flow = FlowSolverParameters.ParamFlow(Re=100)
    params_flow.uinf = 1.0
    params_flow.d = 1.0
    params_flow.sensor_location = np.array([[3, 0], [3.1, 1], [3.1, -1]])
    params_flow.sensor_type = ["v", "v", "v"]
    params_flow.actuator_angular_size = 10
    params_flow.actuator_type = [FlowSolverParameters.ACTUATOR_TYPE.BC]

    params_time = FlowSolverParameters.ParamTime(
        num_steps=10, dt=0.005, Tstart=0.0, Trestartfrom=0.0, restart_order=2
    )

    params_save = FlowSolverParameters.ParamSave(
        save_every=5, save_every_old=5, path_out=cwd / "data_output"
    )
    params_save.compute_norms = True

    params_solver = FlowSolverParameters.ParamSolver(
        throw_error=True, is_eq_nonlinear=True, ic_add_perturbation=1
    )

    params_mesh = FlowSolverParameters.ParamMesh(
        meshpath=cwd / "data_input" / "o1.xdmf"
    )
    params_mesh.xinf = 20
    params_mesh.xinfa = -10
    params_mesh.yinf = 10

    fs = CylinderFlowSolver(
        params_flow=params_flow,
        params_time=params_time,
        params_save=params_save,
        params_solver=params_solver,
        params_mesh=params_mesh,
        verbose=2,
    )

    logger.info("__init__(): successful!")

    logger.info("Compute steady state...")
    u_ctrl_steady = 0.0
    fs.compute_steady_state(method="picard", max_iter=3, tol=1e-7, u_ctrl=u_ctrl_steady)
    fs.compute_steady_state(
        method="newton", max_iter=25, u_ctrl=u_ctrl_steady, initial_guess=fs.UP0
    )
    fs.load_steady_state()

    logger.info("Init time-stepping")
    # fs.initialize_time_stepping(IC=dolfin.Function(fs.W))
    fs.initialize_time_stepping(IC=None)

    logger.info("Step several times")
    G = flu.read_ss(cwd / "data_input" / "sysid_o16_d=3_ssest.mat")
    Kss = flu.read_ss(cwd / "data_input" / "Kopt_reduced13.mat")

    x_ctrl = np.zeros((Kss.A.shape[0],))

    u_ctrl = 0
    for i in range(fs.params_time.num_steps):
        y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas)
        y_meas_err = -y_meas[0]
        u_ctrl, x_ctrl = flu.step_controller(Kss, x_ctrl, y_meas_err, fs.params_time.dt)
        fs.step(u_ctrl=u_ctrl)

    flu.summarize_timings(fs, t000)
    fs.write_timeseries()

    ###########################################
    params_time_restart = params_time
    params_time_restart.Tstart = 0.05
    params_time_restart.dt_old = 0.005

    fs_restart = CylinderFlowSolver(
        params_flow=params_flow,
        params_time=params_time_restart,
        params_save=params_save,
        params_solver=params_solver,
        params_mesh=params_mesh,
        verbose=3,
    )

    fs_restart.load_steady_state()
    fs_restart.initialize_time_stepping(Tstart=fs.params_time.Tstart)

    for i in range(fs_restart.params_time.num_steps):
        y_meas = flu.MpiUtils.mpi_broadcast(fs_restart.y_meas)
        y_meas_err = -y_meas[0]
        u_ctrl, x_ctrl = flu.step_controller(
            Kss, x_ctrl, y_meas_err, fs_restart.params_time.dt
        )
        fs_restart.step(u_ctrl=u_ctrl)

    fs_restart.write_timeseries()

    logger.info("Checking utilitary functions")
    fs.get_A()

    logger.info("Testing max(u) and mean(u)...")
    u_max_ref = 1.6345453902677856
    u_mean_ref = -0.0009997385060749036
    u_max = flu.apply_fun(fs.u_, np.max)
    u_mean = flu.apply_fun(fs.u_, np.mean)

    logger.info(f"umax: {u_max} // {u_max_ref}")
    logger.info(f"umean: {u_mean} // {u_mean_ref}")

    assert np.isclose(u_max, u_max_ref)
    assert np.isclose(u_mean, u_mean_ref)

    logger.info(fs_restart.timeseries)

    logger.info("End with success")


## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
