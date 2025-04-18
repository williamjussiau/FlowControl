"""
Incompressible Navier-Stokes equations

  u' + u . nabla(u)) - div(sigma(u, p)) = f
                                 div(u) = 0
Equations were made non-dimensional
----------------------------------------------------------------------
"""

import logging
import time
from pathlib import Path

import dolfin
import flowsolver
import flowsolverparameters
import numpy as np
import pandas
import utils_flowsolver as flu
from actuator import ActuatorForceGaussianV
from flowfield import BoundaryConditions
from sensor import SENSOR_TYPE, SensorHorizontalWallShear, SensorPoint

# from controller import Controller


# import utils_extract as flu2


# LOG
dolfin.set_log_level(dolfin.LogLevel.INFO)  # DEBUG TRACE PROGRESS INFO
logger = logging.getLogger(__name__)
FORMAT = "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)


class CavityFlowSolver(flowsolver.FlowSolver):
    """Flow over an open cavity. Proposed Re=7500."""

    # @abstractmethod
    # def _make_boundaries(self) -> pd.DataFrame:
    def _make_boundaries(self):
        #                    sf
        #   ------------------------------------------
        #   |                                        |
        # in|                                        |out
        #   |                                        |
        #   -----x0nsl---      -----x0nsr-------------
        #      sf     ns|      | ns           sf
        #               |      |
        #               |      |
        #               --------
        #                  ns
        near_cpp = flu.near_cpp
        between_cpp = flu.between_cpp
        and_cpp = flu.and_cpp()
        on_boundary_cpp = flu.on_boundary_cpp()

        MESH_TOL = dolfin.DOLFIN_EPS
        L = self.params_flow.user_data["L"]
        D = self.params_flow.user_data["D"]
        xinfa = self.params_mesh.user_data["xinfa"]
        xinf = self.params_mesh.user_data["xinf"]
        yinf = self.params_mesh.user_data["yinf"]
        x0ns_left = self.params_mesh.user_data["x0ns_left"]
        x0ns_right = self.params_mesh.user_data["x0ns_right"]

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

        # Open cavity
        if 0:  # not-compiled syntax (kept for information)
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

            # # stress free
            class bnd_lower_wall_right_sf(dolfin.SubDomain):
                """Lower wall right, stress free"""

                def inside(self, x, on_boundary):
                    return (
                        on_boundary
                        and dolfin.between(x[0], (x0ns_right, xinf))
                        and dolfin.near(x[1], 0)
                    )

            lower_wall_right_sf = bnd_lower_wall_right_sf()

        else:  # compiled
            cavity_left = dolfin.CompiledSubDomain(
                on_boundary_cpp
                + and_cpp
                + near_cpp("x[0]", "0", "MESH_TOL")
                + and_cpp
                + between_cpp("x[1]", "-D", "0"),
                MESH_TOL=MESH_TOL,
                D=D,
            )

            cavity_botm = dolfin.CompiledSubDomain(
                on_boundary_cpp
                + and_cpp
                + near_cpp("x[1]", "-D", "MESH_TOL")
                + and_cpp
                + between_cpp("x[0]", "0", "L"),
                L=L,
                D=D,
                MESH_TOL=MESH_TOL,
            )

            cavity_right = dolfin.CompiledSubDomain(
                on_boundary_cpp
                + and_cpp
                + "near(x[0], L, MESH_TOL)"
                + and_cpp
                + between_cpp("x[1]", -D, 0),
                L=L,
                D=D,
                MESH_TOL=MESH_TOL,
            )

            lower_wall_left_sf = dolfin.CompiledSubDomain(
                on_boundary_cpp
                + and_cpp
                + "x[0] >= xinfa"
                + and_cpp
                + "x[0] <= x0ns_left + 10*MESH_TOL"
                + and_cpp
                + near_cpp("x[1]", "0"),
                xinfa=xinfa,
                x0ns_left=x0ns_left,
                MESH_TOL=MESH_TOL,
            )

            lower_wall_left_ns = dolfin.CompiledSubDomain(
                on_boundary_cpp
                + and_cpp
                + "x[0] >= x0ns_left - 10*MESH_TOL"
                + and_cpp
                + "x[0] <= 0"
                + and_cpp
                + near_cpp("x[1]", "0"),
                x0ns_left=x0ns_left,
                MESH_TOL=MESH_TOL,
            )

            lower_wall_right_ns = dolfin.CompiledSubDomain(
                on_boundary_cpp
                + and_cpp
                + near_cpp("x[1]", 0)
                + and_cpp
                + between_cpp("x[0]", "L", "x0ns_right"),
                x0ns_right=x0ns_right,
                L=L,
                MESH_TOL=MESH_TOL,
            )

            lower_wall_right_sf = dolfin.CompiledSubDomain(
                on_boundary_cpp
                + and_cpp
                + near_cpp("x[1]", 0)
                + and_cpp
                + between_cpp("x[0]", "x0ns_right", "xinf"),
                x0ns_right=x0ns_right,
                xinf=xinf,
                MESH_TOL=MESH_TOL,
            )

        # Concatenate all boundaries
        subdomains_list = [
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

        boundaries_names = [
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

        boundaries_df = pandas.DataFrame(
            index=boundaries_names, data={"subdomain": subdomains_list}
        )

        return boundaries_df

    # @abstractmethod
    def _make_bcs(self):
        # inlet : u=uinf, v=0
        bcu_inlet = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((0, 0)),
            self.get_subdomain("inlet"),
        )
        # upper wall : dy(u)=0 # TODO
        bcu_upper_wall = dolfin.DirichletBC(
            self.W.sub(0).sub(1),
            dolfin.Constant(0),
            self.get_subdomain("upper_wall"),
        )
        # lower wall left sf : v=0 + dy(u)=0 # TODO
        bcu_lower_wall_left_sf = dolfin.DirichletBC(
            self.W.sub(0).sub(1),
            dolfin.Constant(0),
            self.get_subdomain("lower_wall_left_sf"),
        )
        # lower wall left ns : u=0; v=0
        bcu_lower_wall_left_ns = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((0, 0)),
            self.get_subdomain("lower_wall_left_ns"),
        )
        # lower wall right ns : u=0; v=0
        bcu_lower_wall_right_ns = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((0, 0)),
            self.get_subdomain("lower_wall_right_ns"),
        )
        # lower wall right sf : v=0 + dy(u)=0 # TODO
        bcu_lower_wall_right_sf = dolfin.DirichletBC(
            self.W.sub(0).sub(1),
            dolfin.Constant(0),
            self.get_subdomain("lower_wall_right_sf"),
        )
        # cavity : no slip, u=0; v=0
        bcu_cavity_left = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((0, 0)),
            self.get_subdomain("cavity_left"),
        )
        bcu_cavity_botm = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((0, 0)),
            self.get_subdomain("cavity_botm"),
        )
        bcu_cavity_right = dolfin.DirichletBC(
            self.W.sub(0),
            dolfin.Constant((0, 0)),
            self.get_subdomain("cavity_right"),
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

        # pressure on outlet -> p=0 or p free (standard outflow)
        # bcp_outlet = dolfin.DirichletBC(
        #     self.W.sub(1),
        #     dolfin.Constant(0),
        #     self.get_subdomain("outlet"),
        # )
        # bcp = [bcp_outlet]
        bcp = []

        return BoundaryConditions(bcu=bcu, bcp=bcp)

    def _steady_state_default_initial_guess(self) -> dolfin.UserExpression:
        class default_initial_guess(dolfin.UserExpression):
            def eval(self, value, x):
                value[0] = 1.0
                value[1] = 0.0
                value[2] = 0.0
                if x[1] <= 0:  # inside cavity
                    value[0] = 0.0

            def value_shape(self):
                return (3,)

        return default_initial_guess()


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

    params_flow = flowsolverparameters.ParamFlow(Re=7500, uinf=1.0)
    params_flow.user_data["L"] = 1.0
    params_flow.user_data["D"] = 1.0

    params_time = flowsolverparameters.ParamTime(num_steps=10, dt=0.0004, Tstart=0.0)

    params_save = flowsolverparameters.ParamSave(
        save_every=5, path_out=cwd / "data_output"
    )

    params_solver = flowsolverparameters.ParamSolver(throw_error=True, is_eq_nonlinear=True, shift=0.0)

    params_mesh = flowsolverparameters.ParamMesh(
        meshpath=cwd / "data_input" / "cavity_coarse.xdmf"
    )
    params_mesh.user_data["xinf"] = 2.5
    params_mesh.user_data["xinfa"] = -1.2
    params_mesh.user_data["yinf"] = 0.5
    params_mesh.user_data["x0ns_left"] = -0.4  # sf left from x0ns, ns right from x0ns
    params_mesh.user_data["x0ns_right"] = 1.75  # ns left from x0ns, sf right from x0ns

    params_restart = flowsolverparameters.ParamRestart()

    actuator_force = ActuatorForceGaussianV(
        sigma=0.0849, position=np.array([-0.1, 0.02])
    )
    sensor_feedback = SensorHorizontalWallShear(
        sensor_index=100,
        x_sensor_left=1.0,
        x_sensor_right=1.1,
        y_sensor=0.0,
        sensor_type=SENSOR_TYPE.OTHER,
    )
    sensor_perf_1 = SensorPoint(
        sensor_type=SENSOR_TYPE.U, position=np.array([0.1, 0.1])
    )
    params_control = flowsolverparameters.ParamControl(
        sensor_list=[sensor_feedback, sensor_perf_1],
        actuator_list=[actuator_force],
    )

    params_ic = flowsolverparameters.ParamIC(
        xloc=2.0, yloc=0.0, radius=0.5, amplitude=1.0
    )

    fs = CavityFlowSolver(
        params_flow=params_flow,
        params_time=params_time,
        params_save=params_save,
        params_solver=params_solver,
        params_mesh=params_mesh,
        params_restart=params_restart,
        params_control=params_control,
        params_ic=params_ic,
        verbose=5,
    )

    logger.info("__init__(): successful!")

    logger.info("Exporting subdomains...")
    flu.export_subdomains(
        fs.mesh, fs.boundaries.subdomain, cwd / "data_output" / "subdomains.xdmf"
    )

    logger.info("Compute steady state...")
    uctrl0 = [0.0]
    fs.compute_steady_state(method="picard", max_iter=10, tol=1e-7, u_ctrl=uctrl0)
    fs.compute_steady_state(
        method="newton", max_iter=10, u_ctrl=uctrl0, initial_guess=fs.fields.UP0
    )

    logger.info("Init time-stepping")
    fs.initialize_time_stepping(ic=None)  # or ic=dolfin.Function(fs.W)

    logger.info("Step several times")
    for _ in range(fs.params_time.num_steps):
        y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas)
        u_ctrl = [0.3 + 0.1 * y_meas[0]]
        fs.step(u_ctrl=u_ctrl)

    flu.summarize_timings(fs, t000)
    fs.write_timeseries()

    logger.info(fs.timeseries)


## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
