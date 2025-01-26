"""
Incompressible Navier-Stokes equations

  u' + u . nabla(u)) - div(sigma(u, p)) = f
                                 div(u) = 0
Equations were made non-dimensional
----------------------------------------------------------------------
"""

import flowsolver
import dolfin
from dolfin import dot, inner, nabla_grad, div, dx
import numpy as np
import time
import pandas as pd
import flowsolverparameters
# from controller import Controller

import logging

from pathlib import Path

import utils_flowsolver as flu
# import utils_extract as flu2


# LOG
dolfin.set_log_level(dolfin.LogLevel.INFO)  # DEBUG TRACE PROGRESS INFO
logger = logging.getLogger(__name__)
FORMAT = "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)


class CavityFlowSolver(flowsolver.FlowSolver):
    """Base class for calculating flow
    Is instantiated with several structures (dicts) containing parameters
    See method .step and main for time-stepping (possibly actuated)
    Contain methods for frequency-response computation"""

    # TODO
    # picard initial guess
    # form: - dot(f, v) * dx
    # problem with make_sensor and sensor_location -> sensor_size? around some location
    # or make_sensor then make_measurement uses sensor?
    # or class Sensor -> size x, size y, can be 0 for point sensor... or directly a type -> intg or point
    # and class Actuator? with an Expression, a type and stuff

    # def __init__(
    #     self,
    # ):
    #     pass

    # @abstractmethod
    # def _make_boundaries(self) -> pd.DataFrame:
    def _make_boundaries(self):
        """Define boundaries (inlet, outlet, walls, and so on)
          Geometry and boundaries are the following:
                           sf
          ------------------------------------------
          |                                        |
        in|                                        |out
          |                                        |
          -----x0nsl---      -----x0nsr-------------
             sf     ns|      | ns           sf
                      |      |
                      |      |
                      --------
                         ns
        """
        MESH_TOL = dolfin.DOLFIN_EPS
        L = self.params_flow.d
        D = self.params_flow.d
        xinfa = self.params_mesh.xinfa
        xinf = self.params_mesh.xinf
        yinf = self.params_mesh.yinf
        x0ns_left = self.params_mesh.x0ns_left
        x0ns_right = self.params_mesh.x0ns_right

        ## Inlet
        inlet = dolfin.CompiledSubDomain(
            "on_boundary && \
                near(x[0], xinfa, MESH_TOL)",
            xinfa=xinfa,
            MESH_TOL=MESH_TOL,
        )

        ## Outlet
        outlet = dolfin.CompiledSubDomain(
            "on_boundary && \
                near(x[0], xinf, MESH_TOL)",
            xinf=xinf,
            MESH_TOL=MESH_TOL,
        )

        ## Upper wall
        upper_wall = dolfin.CompiledSubDomain(
            "on_boundary && \
                     near(x[1], yinf, MESH_TOL)",
            yinf=yinf,
            MESH_TOL=MESH_TOL,
        )

        ## Open cavity
        # cavity left
        class bnd_cavity_left(dolfin.SubDomain):
            """Left wall of cavity"""

            def inside(self, x, on_boundary):
                return (
                    on_boundary
                    and dolfin.between(x[1], (-D, 0))
                    and dolfin.near(x[0], 0)
                )

        cavity_left = bnd_cavity_left()

        # cavity bottom
        class bnd_cavity_botm(dolfin.SubDomain):
            """Bottom wall of cavity"""

            def inside(self, x, on_boundary):
                return (
                    on_boundary
                    and dolfin.between(x[0], (0, L))
                    and dolfin.near(x[1], -D)
                )

        cavity_botm = bnd_cavity_botm()

        # cavity right
        class bnd_cavity_right(dolfin.SubDomain):
            """Right wall of cavity"""

            def inside(self, x, on_boundary):
                return (
                    on_boundary
                    and dolfin.between(x[1], (-D, 0))
                    and dolfin.near(x[0], L)
                )

        cavity_right = bnd_cavity_right()

        # Lower wall
        # left
        # stress free
        class bnd_lower_wall_left_sf(dolfin.SubDomain):
            """Lower wall left, stress free"""

            def inside(self, x, on_boundary):
                return (
                    on_boundary
                    and x[0] >= xinfa
                    and x[0] <= x0ns_left + 10 * MESH_TOL
                    and dolfin.near(x[1], 0)
                )
                # add MESH_TOL to force all cells to belong to a dolfin.SubDomain

        lower_wall_left_sf = bnd_lower_wall_left_sf()

        # no slip
        class bnd_lower_wall_left_ns(dolfin.SubDomain):
            """Lower wall left, no stress"""

            def inside(self, x, on_boundary):
                return (
                    on_boundary
                    and x[0] >= x0ns_left - 10 * MESH_TOL
                    and x[0] <= 0
                    and dolfin.near(x[1], 0)
                )
                # add MESH_TOL to force all cells to belong to a dolfin.SubDomain

        lower_wall_left_ns = bnd_lower_wall_left_ns()

        # right
        # no slip
        class bnd_lower_wall_right_ns(dolfin.SubDomain):
            """Lower wall right, no slip"""

            def inside(self, x, on_boundary):
                return (
                    on_boundary
                    and dolfin.between(x[0], (L, x0ns_right))
                    and dolfin.near(x[1], 0)
                )

        lower_wall_right_ns = bnd_lower_wall_right_ns()

        # stress free
        class bnd_lower_wall_right_sf(dolfin.SubDomain):
            """Lower wall right, stress free"""

            def inside(self, x, on_boundary):
                return (
                    on_boundary
                    and dolfin.between(x[0], (x0ns_right, xinf))
                    and dolfin.near(x[1], 0)
                )

        lower_wall_right_sf = bnd_lower_wall_right_sf()

        # Concatenate all boundaries
        subdmlist = [
            inlet,
            outlet,
            upper_wall,
            cavity_left,
            cavity_botm,
            cavity_right,
            lower_wall_left_sf,
            lower_wall_left_ns,
            lower_wall_right_ns,
            lower_wall_right_sf,
        ]

        boundaries_list = [
            "inlet",
            "outlet",
            "upper_wall",
            "cavity_left",
            "cavity_botm",
            "cavity_right",
            "lower_wall_left_sf",
            "lower_wall_left_ns",
            "lower_wall_right_ns",
            "lower_wall_right_sf",
        ]
        boundaries_df = pd.DataFrame(
            index=boundaries_list, data={"subdomain": subdmlist}
        )

        return boundaries_df

    def make_sensor(self):
        """Define sensor-related quantities (surface of integration, dolfin.SubDomain...)"""
        # define sensor surface
        xs0 = 1.0
        xs1 = 1.1
        MESH_TOL = dolfin.DOLFIN_EPS
        sensor_subdm = dolfin.CompiledSubDomain(
            "on_boundary && near(x[1], 0, MESH_TOL) && x[0]>=xs0 && x[0]<=xs1",
            MESH_TOL=MESH_TOL,
            xs0=xs0,
            xs1=xs1,
        )
        # define function to index cells
        sensor_mark = dolfin.MeshFunction(
            "size_t", self.mesh, self.mesh.topology().dim() - 1
        )
        # define sensor as index 100 and mark
        SENSOR_IDX = 100
        sensor_subdm.mark(sensor_mark, SENSOR_IDX)
        # define surface element ds on sensor
        ds_sensor = dolfin.Measure("ds", domain=self.mesh, subdomain_data=sensor_mark)
        self.ds_sensor = ds_sensor
        # self.sensor_dolfin.SubDomain = sensor_subdm

        # Append sensor to boundaries but not to markers... might it be dangerous?
        df_sensor = pd.DataFrame(
            data=dict(subdomain=sensor_subdm, idx=SENSOR_IDX), index=["sensor"]
        )
        # self.boundaries = self.boundaries.append(df_sensor)
        self.boundaries = pd.concat((self.boundaries, df_sensor))
        self.sensor_ok = True  # TODO rm

    # @abstractmethod
    def _make_actuator(self):
        """Define actuator: on boundary (with ternary operator cond?true:false) or volumic..."""
        # make actuator with amplitude 1
        actuator_expr = dolfin.Expression(
            [
                "0",
                "ampl*eta*exp(-0.5*((x[0]-x10)*(x[0]-x10)+(x[1]-x20)*(x[1]-x20))/(sig*sig))",
            ],
            element=self.V.ufl_element(),
            ampl=1,
            eta=1,
            sig=0.0849,
            x10=-0.1,
            x20=0.02,
        )

        BtB = dolfin.norm(actuator_expr, mesh=self.mesh)  # coeff for checking things
        actuator_expr.eta = 1 / BtB  # 1/BtB #1/int2d (not the same)
        return actuator_expr

    # @abstractmethod
    def _make_bcs(self):
        """Define boundary conditions"""
        # inlet : u=uinf, v=0
        bcu_inlet = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((0, 0)),
            self.boundaries.loc["inlet"].subdomain,
        )
        # upper wall : dy(u)=0 # TODO
        bcu_upper_wall = dolfin.DirichletBC(
            self.W.sub(0).sub(1),
            dolfin.Constant(0),
            self.boundaries.loc["upper_wall"].subdomain,
        )
        # lower wall left sf : v=0 + dy(u)=0 # TODO
        bcu_lower_wall_left_sf = dolfin.DirichletBC(
            self.W.sub(0).sub(1),
            dolfin.Constant(0),
            self.boundaries.loc["lower_wall_left_sf"].subdomain,
        )
        # lower wall left ns : u=0; v=0
        bcu_lower_wall_left_ns = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((0, 0)),
            self.boundaries.loc["lower_wall_left_ns"].subdomain,
        )
        # lower wall right ns : u=0; v=0
        bcu_lower_wall_right_ns = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((0, 0)),
            self.boundaries.loc["lower_wall_right_ns"].subdomain,
        )
        # lower wall right sf : v=0 + dy(u)=0 # TODO
        bcu_lower_wall_right_sf = dolfin.DirichletBC(
            self.W.sub(0).sub(1),
            dolfin.Constant(0),
            self.boundaries.loc["lower_wall_right_sf"].subdomain,
        )
        # cavity : no slip, u=0; v=0
        bcu_cavity_left = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((0, 0)),
            self.boundaries.loc["cavity_left"].subdomain,
        )
        bcu_cavity_botm = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((0, 0)),
            self.boundaries.loc["cavity_botm"].subdomain,
        )
        bcu_cavity_right = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((0, 0)),
            self.boundaries.loc["cavity_right"].subdomain,
        )

        bcu = [
            bcu_inlet,
            bcu_upper_wall,
            bcu_lower_wall_left_sf,
            bcu_lower_wall_left_ns,
            bcu_lower_wall_right_ns,
            bcu_lower_wall_right_sf,
            bcu_cavity_left,
            bcu_cavity_botm,
            bcu_cavity_right,
        ]

        # pressure on outlet -> TODO not free like cylinder?
        bcp_outlet = dolfin.DirichletBC(
            self.W.sub(1),
            dolfin.Constant(0),
            self.boundaries.loc["outlet"].subdomain,
        )

        bcp = [bcp_outlet]

        return {"bcu": bcu, "bcp": bcp}

    # @abstractmethod
    # def make_measurement(self) -> np.ndarray:
    def make_measurement(self, field=None, mixed_field=None):
        """Perform measurement and assign"""

        if not hasattr(self, "sensor_ok"):  # TODO quickfix for sensor
            self.make_sensor()

        ns = self.params_control.sensor_number
        y_meas = np.zeros((ns,))

        ds_sensor = self.ds_sensor
        SENSOR_IDX = int(self.boundaries.loc["sensor"].idx)

        for isensor in range(ns):
            if field is not None:
                ff = field
            else:
                if mixed_field is not None:
                    ff = mixed_field
                else:
                    ff = self.u_
            y_meas_i = dolfin.assemble(ff.dx(1)[0] * ds_sensor(int(SENSOR_IDX)))
            y_meas[isensor] = y_meas_i

        return y_meas

    def get_A(self, perturbations=True, shift=0.0, timeit=True, up_0=None):
        """Get state-space dynamic matrix A around some state up_0"""
        if timeit:
            print("Computing jacobian A...")
            t0 = time.time()

        Jac = dolfin.PETScMatrix()
        v, q = dolfin.TestFunctions(self.W)
        iRe = dolfin.Constant(1 / self.Re)
        shift = dolfin.Constant(shift)
        self.actuator_expression.ampl = 0.0

        if up_0 is None:
            up_ = self.up0  # base flow
        else:
            up_ = up_0
        u_, p_ = up_.dolfin.split()

        if perturbations:  # perturbation equations lidolfin.nearized
            up = dolfin.TrialFunction(self.W)
            u, p = dolfin.split(up)
            # u0 = self.u0
            dF0 = (
                -dot(dot(u_, nabla_grad(u)), v) * dx
                - dot(dot(u, nabla_grad(u_)), v) * dx
                - iRe * inner(nabla_grad(u), nabla_grad(v)) * dx
                + p * div(v) * dx
                + div(u) * q * dx
                - shift * dot(u, v) * dx
            )  # sum u, v but not p
            bcu = self.bc_p["bcu"]
            bcs = bcu
        else:  # full ns + derivative
            up_ = self.up0
            u_, p_ = dolfin.split(up_)
            F0 = (
                -dot(dot(u_, nabla_grad(u_)), v) * dx
                - iRe * inner(nabla_grad(u_), nabla_grad(v)) * dx
                + p_ * div(v) * dx
                + q * div(u_) * dx
                - shift * dot(u_, v) * dx
            )
            # prepare derivation
            du = dolfin.TrialFunction(self.W)
            dF0 = dolfin.derivative(F0, up_, du=du)
            ## shift
            # dF0 = dF0 - shift*dot(u_,v)*dx
            # bcs
            bcs = self.bc["bcu"]

        dolfin.assemble(dF0, tensor=Jac)
        [bc.apply(Jac) for bc in bcs]

        if timeit:
            print("Elapsed time: ", time.time() - t0)

        return Jac

    def get_B(self, export=False, timeit=True):
        """Get actuation matrix B"""
        print("Computing actuation matrix B...")

        if timeit:
            t0 = time.time()

        # for an exponential actuator -> just evaluate actuator_exp on every coordinate, kinda?
        # for a boundary actuator -> evaluate actuator on boundary
        actuator_ampl_old = self.actuator_expression.ampl
        self.actuator_expression.ampl = 1.0

        # Method 2
        # Projet actuator expression on W
        class ExpandFunctionSpace(dolfin.UserExpression):
            """Expand function from space [V1, V2] to [V1, V2, P]"""

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
        actuator_extended = dolfin.interpolate(actuator_extended, self.W)
        B_proj = flu.projectm(actuator_extended, self.W)
        B = B_proj.vector().get_local()

        # remove very small values (should be 0 but are not)
        B = flu.dense_to_sparse(
            B, eliminate_zeros=True, eliminate_under=1e-14
        ).toarray()
        B = B.T  # vertical B

        if export:
            fa = dolfin.FunctionAssigner([self.V, self.P], self.W)
            vv = dolfin.Function(self.V)
            pp = dolfin.Function(self.P)
            ww = dolfin.Function(self.W)
            ww.assign(B_proj)
            fa.assign([vv, pp], ww)
            flu.write_xdmf("B.xdmf", vv, "B")

        self.actuator_expression.ampl = actuator_ampl_old

        if timeit:
            print("Elapsed time: ", time.time() - t0)

        return B

    def get_C(self, timeit=True, check=False, verbose=False):
        """Get measurement matrix C"""
        # Solution to make it faster:
        # localize the region of dofs where C is going to be nonzero
        # and only account for dofs in this region
        print("Computing measurement matrix C...")

        if timeit:
            t0 = time.time()

        # Initialize
        fspace = self.W  # function space
        uvp = dolfin.Function(fspace)  # function to store C
        uvp_vec = uvp.vector()  # as vector
        ndof = fspace.dim()  # size of C
        ns = self.params_control.sensor_number
        C = np.zeros((ns, ndof))

        dofmap = fspace.dofmap()  # indices of dofs
        dofmap_x = fspace.tabulate_dof_coordinates()  # coordinates of dofs

        # Box that encapsulates all dofs on sensor
        margin = 0.05
        xmin = 1 - margin
        xmax = 1.1 + margin
        ymin = 0 - margin
        ymax = 0 + margin
        xymin = np.array([xmin, ymin]).reshape(1, -1)
        xymax = np.array([xmax, ymax]).reshape(1, -1)
        # keep dofs with coordinates inside box
        dof_in_box = (
            np.greater_equal(dofmap_x, xymin) * np.less_equal(dofmap_x, xymax)
        ).all(axis=1)
        # retrieve said dof index
        dof_in_box_idx = np.array(dofmap.dofs())[dof_in_box]

        # Iteratively put each DOF at 1
        # And evaluate measurement on said DOF
        idof_old = 0
        ii = 0  # counter of the number of dofs evaluated
        for idof in dof_in_box_idx:
            ii += 1
            if verbose and not ii % 1000:
                print("get_C::eval iter {0} - dof n°{1}/{2}".format(ii, idof, ndof))
            # set field 1 at said dof
            uvp_vec[idof] = 1
            uvp_vec[idof_old] = 0
            idof_old = idof
            # retrieve coordinates
            # dof_x = dofmap_x[idof] # not needed for measurement
            # evaluate measurement
            C[:, idof] = self.make_measurement(mixed_field=uvp)

        # check:
        if check:
            for i in range(ns):
                # sensor_types = dict(u=0, v=1, p=2)
                # print('True probe: ', self.up0(self.sensor_location[i])[sensor_types[self.sensor_type[0]]])
                # true probe would be make_measurement(...)
                print("\t with fun:", self.make_measurement(mixed_field=self.up0))
                print("\t with C@x: ", C[i] @ self.up0.vector().get_local())

        if timeit:
            print("Elapsed time: ", time.time() - t0)

        return C


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

    params_flow = flowsolverparameters.ParamFlow(Re=7500)
    params_flow.uinf = 1.0
    params_flow.d = 1.0

    # params_time = flowsolverparameters.ParamTime(num_steps=10, dt=0.0004, Tstart=0.0)
    params_time = flowsolverparameters.ParamTime(num_steps=100, dt=0.001, Tstart=0.0)

    params_save = flowsolverparameters.ParamSave(
        save_every=5, path_out=cwd / "data_output"
    )

    params_solver = flowsolverparameters.ParamSolver(
        throw_error=True, is_eq_nonlinear=True, ic_add_perturbation=1.0, shift=0.0
    )

    params_mesh = flowsolverparameters.ParamMesh(
        meshpath=cwd / "data_input" / "cavity_coarse.xdmf"
    )
    params_mesh.xinf = 2.5
    params_mesh.xinfa = -1.2
    params_mesh.yinf = 0.5
    params_mesh.x0ns_left = -0.4  # sf left from x0ns, ns right from x0ns
    params_mesh.x0ns_right = 1.75  # ns left from x0ns, sf right from x0ns

    params_restart = flowsolverparameters.ParamRestart()

    params_control = flowsolverparameters.ParamControl(
        sensor_location=np.array([[3, 0], [3.1, 1], [3.1, -1]]),
        sensor_type=[flowsolverparameters.SENSOR_TYPE.V] * 3,
        sensor_number=3,
        actuator_type=[flowsolverparameters.ACTUATOR_TYPE.BC],
        actuator_location=np.array([[3, 0]]),
        actuator_number=2,
        actuator_parameters=dict(angular_size_deg=10),
    )

    fs = CavityFlowSolver(
        params_flow=params_flow,
        params_time=params_time,
        params_save=params_save,
        params_solver=params_solver,
        params_mesh=params_mesh,
        params_restart=params_restart,
        params_control=params_control,
        verbose=5,
    )

    logger.info("__init__(): successful!")

    logger.info("Exporting subdomains...")
    flu.export_subdomains(
        fs.mesh, fs.boundaries.subdomain, cwd / "data_output" / "subdomains.xdmf"
    )

    logger.info("Compute steady state...")
    uctrl0 = 0.0
    fs.compute_steady_state(method="picard", max_iter=10, tol=1e-7, u_ctrl=uctrl0)
    fs.compute_steady_state(
        method="newton", max_iter=10, u_ctrl=uctrl0, initial_guess=fs.fields.UP0
    )

    logger.info("Init time-stepping")
    fs.initialize_time_stepping(ic=None)  # or ic=dolfin.Function(fs.W)

    logger.info("Step several times")
    for _ in range(fs.params_time.num_steps):
        y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas)
        u_ctrl = 0
        fs.step(u_ctrl=u_ctrl)

    flu.summarize_timings(fs, t000)
    fs.write_timeseries()

## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
