"""
Incompressible Navier-Stokes equations

  u' + u . nabla(u)) - div(sigma(u, p)) = f
                                 div(u) = 0
Equations were made non-dimensional
----------------------------------------------------------------------
"""

import flowsolver
import dolfin
import numpy as np
import time
import pandas as pd
import flowsolverparameters
from sensor import SensorPoint, SENSOR_TYPE
from actuator import ActuatorBCParabolicV
from controller import Controller

import logging

from pathlib import Path

import utils_flowsolver as flu
import utils_extract as flu2


# LOG
dolfin.set_log_level(dolfin.LogLevel.INFO)  # DEBUG TRACE PROGRESS INFO
logger = logging.getLogger(__name__)
FORMAT = "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)


class CylinderFlowSolver(flowsolver.FlowSolver):
    """Flow past a cylinder. Proposed Re=100."""

    # Abstract methods
    def _make_boundaries(self):
        MESH_TOL = dolfin.DOLFIN_EPS
        ## Inlet
        inlet = dolfin.CompiledSubDomain(
            "on_boundary && \
                near(x[0], xinfa, MESH_TOL)",
            xinfa=self.params_mesh.user_data["xinfa"],
            MESH_TOL=MESH_TOL,
        )
        ## Outlet
        outlet = dolfin.CompiledSubDomain(
            "on_boundary && \
                near(x[0], xinf, MESH_TOL)",
            xinf=self.params_mesh.user_data["xinf"],
            MESH_TOL=MESH_TOL,
        )
        ## Walls
        walls = dolfin.CompiledSubDomain(
            "on_boundary && \
                (near(x[1], -yinf, MESH_TOL) ||   \
                 near(x[1], yinf, MESH_TOL))",
            yinf=self.params_mesh.user_data["yinf"],
            MESH_TOL=MESH_TOL,
        )

        ## Cylinder
        # Compiled subdomains (str)
        # = increased speed but decreased readability

        def between_cpp(x: str, xmin: str, xmax: str, tol: str = "0.0"):
            return f"{x}>={xmin}-{tol} && {x}<={xmax}+{tol}"

        and_cpp = " && "
        or_cpp = " || "
        on_boundary_cpp = "on_boundary"

        radius = self.params_flow.user_data["D"] / 2
        ldelta = radius * np.sin(
            self.params_control.actuator_list[0].angular_size_deg / 2 * dolfin.pi / 180
        )

        # close_to_cylinder_cpp = between_cpp("x[0]*x[0] + x[1]*x[1]", "0", "2*radius*radius")
        close_to_cylinder_cpp = (
            between_cpp("x[0]", "-radius", "radius")
            + and_cpp
            + between_cpp("x[1]", "-radius", "radius")
        )
        cylinder_boundary_cpp = on_boundary_cpp + and_cpp + close_to_cylinder_cpp

        cone_up_cpp = (
            between_cpp("x[0]", "-ldelta", "ldelta", tol="0.01")
            + and_cpp
            + between_cpp("x[1]", "0", "radius")
        )
        cone_lo_cpp = (
            between_cpp("x[0]", "-ldelta", "ldelta", tol="0.01")
            + and_cpp
            + between_cpp("x[1]", "-radius", "0")
        )

        cone_le_cpp = between_cpp("x[0]", "-radius", "-ldelta")
        cone_ri_cpp = between_cpp("x[0]", "ldelta", "radius")

        cylinder = dolfin.CompiledSubDomain(
            cylinder_boundary_cpp
            + and_cpp
            + "("
            + cone_le_cpp
            + or_cpp
            + cone_ri_cpp
            + ")",
            radius=radius,
            ldelta=ldelta,
        )
        actuator_up = dolfin.CompiledSubDomain(
            cylinder_boundary_cpp + and_cpp + cone_up_cpp, radius=radius, ldelta=ldelta
        )
        actuator_lo = dolfin.CompiledSubDomain(
            cylinder_boundary_cpp + and_cpp + cone_lo_cpp, radius=radius, ldelta=ldelta
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

        return boundaries_df

    def _make_bcs(self):
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
            self.params_control.actuator_list[0].expression,
            self.boundaries.loc["actuator_up"].subdomain,
        )
        bcu_actuation_lo = dolfin.DirichletBC(
            self.W.sub(0),
            self.params_control.actuator_list[1].expression,
            self.boundaries.loc["actuator_lo"].subdomain,
        )
        bcu = [bcu_inlet, bcu_walls, bcu_cylinder, bcu_actuation_up, bcu_actuation_lo]

        return {"bcu": bcu, "bcp": []}  # log perturbation bcs

    # Steady state
    def compute_steady_state(self, u_ctrl, method="newton", **kwargs):
        super().compute_steady_state(method=method, u_ctrl=u_ctrl, **kwargs)
        # assign steady cl, cd
        cl, cd = self.compute_force_coefficients(self.fields.U0, self.fields.P0)

        self.cl0 = cl
        self.cd0 = cd
        if self.verbose:
            logger.info(f"Lift coefficient is: cl = {cl}")
            logger.info(f"Drag coefficient is: cd = {cd}")

    # Additional, case-specific func
    def compute_force_coefficients(
        self, u: dolfin.Function, p: dolfin.Function
    ) -> tuple[float, float]:  # keep this one in here
        """Compute lift & drag coefficients acting on the cylinder."""
        D = self.params_flow.user_data["D"]
        nu = self.params_flow.uinf * D / self.params_flow.Re

        sigma = flu2.stress_tensor(nu, u, p)
        facet_normals = dolfin.FacetNormal(self.mesh)
        Fo = -dolfin.dot(sigma, facet_normals)

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
        cd = drag / (1 / 2 * self.params_flow.uinf**2 * D)
        cl = lift / (1 / 2 * self.params_flow.uinf**2 * D)
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

    params_flow = flowsolverparameters.ParamFlow(Re=100, uinf=1.0)
    params_flow.user_data["D"] = 1.0

    params_time = flowsolverparameters.ParamTime(num_steps=10, dt=0.005, Tstart=0.0)

    params_save = flowsolverparameters.ParamSave(
        save_every=5, path_out=cwd / "data_output"
    )

    params_solver = flowsolverparameters.ParamSolver(
        throw_error=True, is_eq_nonlinear=True, shift=0.0
    )

    params_mesh = flowsolverparameters.ParamMesh(
        meshpath=cwd / "data_input" / "o1.xdmf"
    )
    params_mesh.user_data["xinf"] = 20
    params_mesh.user_data["xinfa"] = -10
    params_mesh.user_data["yinf"] = 10

    params_restart = flowsolverparameters.ParamRestart()

    # duplicate actuators (1 top, 1 bottom) but assign same control input to each
    actuator_bc_1 = ActuatorBCParabolicV(angular_size_deg=10)
    actuator_bc_2 = ActuatorBCParabolicV(angular_size_deg=10)
    sensor_feedback = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3, 0]))
    sensor_perf_1 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3.1, 1]))
    sensor_perf_2 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3.1, -1]))
    params_control = flowsolverparameters.ParamControl(
        sensor_list=[sensor_feedback, sensor_perf_1, sensor_perf_2],
        actuator_list=[actuator_bc_1, actuator_bc_2],
    )

    params_ic = flowsolverparameters.ParamIC(
        xloc=2.0, yloc=0.0, radius=0.5, amplitude=1.0
    )

    fs = CylinderFlowSolver(
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
    uctrl0 = [0.0, 0.0]
    fs.compute_steady_state(method="picard", max_iter=3, tol=1e-7, u_ctrl=uctrl0)

    fs.compute_steady_state(
        method="newton", max_iter=25, u_ctrl=uctrl0, initial_guess=fs.fields.UP0
    )

    logger.info("Init time-stepping")
    fs.initialize_time_stepping(ic=None)  # or ic=dolfin.Function(fs.W)

    logger.info("Step several times")
    Kss = Controller.from_file(file=cwd / "data_input" / "Kopt_reduced13.mat", x0=0)

    for _ in range(fs.params_time.num_steps):
        y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas)
        u_ctrl = Kss.step(y=-y_meas[0], dt=fs.params_time.dt)
        fs.step(u_ctrl=[u_ctrl[0], u_ctrl[0]])
        # or
        # fs.step(u_ctrl=np.repeat(u_ctrl, repeats=2, axis=0))

    flu.summarize_timings(fs, t000)
    logger.info(fs.timeseries)
    fs.write_timeseries()

    ################################################################################################
    ################################################################################################
    params_time_restart = flowsolverparameters.ParamTime(
        num_steps=10, dt=0.005, Tstart=0.05
    )
    params_restart = flowsolverparameters.ParamRestart(
        save_every_old=5,
        restart_order=2,
        dt_old=0.005,
        Trestartfrom=0.0,
    )

    fs_restart = CylinderFlowSolver(
        params_flow=params_flow,
        params_time=params_time_restart,
        params_save=params_save,
        params_solver=params_solver,
        params_mesh=params_mesh,
        params_restart=params_restart,
        params_control=params_control,
        params_ic=params_ic,
        verbose=5,
    )

    fs_restart.load_steady_state()
    fs_restart.initialize_time_stepping(Tstart=fs_restart.params_time.Tstart)

    for _ in range(fs_restart.params_time.num_steps):
        y_meas = flu.MpiUtils.mpi_broadcast(fs_restart.y_meas)
        u_ctrl = Kss.step(y=-y_meas[0], dt=fs_restart.params_time.dt)
        fs_restart.step(u_ctrl=np.repeat(u_ctrl, repeats=2, axis=0))

    fs_restart.write_timeseries()

    logger.info(fs_restart.timeseries)

    logger.info("Testing max(u) and mean(u)...")
    u_max_ref = 2.2855984664058986
    u_mean_ref = 0.3377669778983669
    u_max = flu.apply_fun(fs_restart.fields.Usave, np.max)
    u_mean = flu.apply_fun(fs_restart.fields.Usave, np.mean)

    logger.info(f"umax: {u_max} (found) // (ref) {u_max_ref}")
    logger.info(f"umean: {u_mean} (found) // (ref) {u_mean_ref}")

    assert np.isclose(u_max, u_max_ref)
    assert np.isclose(u_mean, u_mean_ref)

    logger.info(
        "Last line should be: 10  0.100  0.000000  0.131695  0.009738  0.009810  0.122620  0.222280"
    )

    logger.info("End with success")


## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
# if __name__ == "__main__":
#     main()
