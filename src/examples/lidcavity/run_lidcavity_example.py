import logging
import time
from pathlib import Path

import dolfin

import flowcontrol.flowsolverparameters as flowsolverparameters
import utils.utils_flowsolver as flu
from examples.lidcavity.lidcavityflowsolver import LidCavityFlowSolver
from flowcontrol.actuator import ActuatorBCParabolicV


def main():
    # LOG
    dolfin.set_log_level(dolfin.LogLevel.INFO)  # DEBUG TRACE PROGRESS INFO
    logger = logging.getLogger(__name__)
    FORMAT = "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]: %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)

    t000 = time.time()
    cwd = Path(__file__).parent

    logger.info("Trying to instantiate FlowSolver...")

    params_flow = flowsolverparameters.ParamFlow(Re=4000, uinf=0.1)
    params_flow.user_data["D"] = 1.0

    params_time = flowsolverparameters.ParamTime(num_steps=10, dt=0.005, Tstart=0.0)

    params_save = flowsolverparameters.ParamSave(
        save_every=5, path_out=cwd / "data_output"
    )

    params_solver = flowsolverparameters.ParamSolver(
        throw_error=True, is_eq_nonlinear=True, shift=0.0
    )

    params_mesh = flowsolverparameters.ParamMesh(
        meshpath=cwd / "data_input" / "mesh64.xdmf"
    )
    # mesh is in upper-right quadrant
    params_mesh.user_data["yup"] = 1
    params_mesh.user_data["ylo"] = 0
    params_mesh.user_data["xri"] = 1
    params_mesh.user_data["xle"] = 0

    params_restart = flowsolverparameters.ParamRestart()

    actuator_bc_up = ActuatorBCParabolicV(angular_size_deg=10)
    params_control = flowsolverparameters.ParamControl(
        sensor_list=[],
        actuator_list=[actuator_bc_up],
    )

    params_ic = flowsolverparameters.ParamIC(
        xloc=2.0, yloc=0.0, radius=0.5, amplitude=1.0
    )

    fs = LidCavityFlowSolver(
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
    fs.compute_steady_state(method="picard", max_iter=25, tol=1e-7, u_ctrl=uctrl0)

    fs.compute_steady_state(
        method="newton", max_iter=25, u_ctrl=uctrl0, initial_guess=fs.fields.UP0
    )

    logger.info("Init time-stepping")
    fs.initialize_time_stepping(ic=None)  # or ic=dolfin.Function(fs.W)

    logger.info("Step several times")

    for _ in range(fs.params_time.num_steps):
        y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas)  # should be empty
        u_ctrl = [0.01]

        fs.step(u_ctrl=[u_ctrl[0]])

    flu.summarize_timings(fs, t000)
    logger.info(fs.timeseries)
    fs.write_timeseries()

    logger.info("End with success")


def make_mesh(nx, ny, export=True, filename="mesh.xdmf"):
    """Make simple unit square mesh with dolfin native function
    The mesh can be exported to a xdmf file"""
    mesh = dolfin.UnitSquareMesh(nx, ny)  # options: left, right, crossed

    if export:
        meshpath = Path.cwd() / "src" / "examples" / "lidcavity" / "data_input"
        with dolfin.XDMFFile(str(meshpath / filename)) as meshfile:
            print(f"--- Exported mesh file in {meshpath} ---")
            meshfile.write(mesh)

    return mesh


if __name__ == "__main__":
    main()
