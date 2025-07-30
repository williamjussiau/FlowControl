"""
----------------------------------------------------------------------
Flow over an open cavity
Nondimensional incompressible Navier-Stokes equations
Suggested Re=7500
----------------------------------------------------------------------
This file demonstrates the following possibilites:
    - Initialize CavityFlowSolver object
    - Compute steady-state
    - Perform open-loop time simulation
----------------------------------------------------------------------
"""

import logging
import time
from pathlib import Path

import dolfin
import numpy as np

import flowcontrol.flowsolverparameters as flowsolverparameters
import utils.utils_flowsolver as flu
from examples.cavity.cavityflowsolver import CavityFlowSolver
from flowcontrol.actuator import ActuatorForceGaussianV, ActuatorBCParabolicV
from flowcontrol.sensor import SENSOR_TYPE, SensorHorizontalWallShear, SensorPoint
from examples.cavity.compute_steady_state import Re

def run_lidcavity_with_ic(Re, xloc, yloc, radius, amplitude, save_dir, num_steps=100):
    # LOG
    dolfin.set_log_level(dolfin.LogLevel.INFO)  # DEBUG TRACE PROGRESS INFO
    logger = logging.getLogger(__name__)
    FORMAT = "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]: %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.INFO)

    t000 = time.time()
    cwd = Path(__file__).parent

    logger.info("Trying to instantiate FlowSolver...")

    params_flow = flowsolverparameters.ParamFlow(Re=Re, uinf=1.0)
    params_flow.user_data["L"] = 1.0
    params_flow.user_data["D"] = 1.0

    params_time = flowsolverparameters.ParamTime(num_steps=num_steps, dt=0.0004, Tstart=0.0)


    # params_save = flowsolverparameters.ParamSave(
    #     save_every=5, path_out=save_dir
    # )
    params_save = flowsolverparameters.ParamSave(
        save_every=200, path_out=save_dir
    )

    params_solver = flowsolverparameters.ParamSolver(
        throw_error=True, is_eq_nonlinear=True, shift=0.0
    )

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
    # actuator_bc = ActuatorBCParabolicV(
    #     width=7/20, position_x=np.array([-7/40])
    # )
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

    # params_ic = flowsolverparameters.ParamIC(
    #     xloc=2.0, yloc=0.0, radius=0.5, amplitude=1.0
    # )
    params_ic = flowsolverparameters.ParamIC(
        xloc=xloc, yloc=yloc, radius=radius, amplitude=amplitude
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
        fs.mesh, fs.boundaries.subdomain, save_dir / "subdomains.xdmf"
    )

    logger.info("Load steady state...")
    fs.load_steady_state(
        path_u_p=[
            cwd / "data_output" / "steady" / f"U0.xdmf",
            cwd / "data_output" / "steady" / f"P0.xdmf",
        ]
    )

    logger.info("Init time-stepping")
    fs.initialize_time_stepping(ic=None)  # or ic=dolfin.Function(fs.W)

    logger.info("Step several times")
    for _ in range(fs.params_time.num_steps):
        y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas)
        # u_ctrl = [0.3 + 0.1 * y_meas[0]]
        u_ctrl = [0.0 * y_meas[0]]
        # u_ctrl = [np.sin(fs.t) + 0 * y_meas[0]]
        fs.step(u_ctrl=u_ctrl)

    flu.summarize_timings(fs, t000)
    fs.write_timeseries()

    logger.info(fs.timeseries)

    return


def main():

    base_dir = Path("/Users/jaking/Desktop/PhD/open_cavity")
    parent_dir = base_dir / f"Re{Re}_test_2"
    parent_dir.mkdir(parents=True, exist_ok=True)

    # x_vals = np.linspace(0.2, 0.8, 3)
    # y_vals = np.linspace(0.2, 0.8, 3)
    # x_vals = np.linspace(0.2, 0.2, 1)
    # y_vals = np.linspace(0.2, 0.2, 1)
    x_vals = [2.0]
    y_vals = [0.0]
    # x_vals = [-2.0]
    # y_vals = [0.0]
    radius = 0.5
    amplitude = 1.0
    num_steps = 20000
    count = 1
    for xloc in x_vals:
        for yloc in y_vals:
            save_dir = parent_dir / f"run{count}"
            save_dir.mkdir(parents=True, exist_ok=True)

            print(f"Running simulation {count} with xloc={xloc:.3f}, yloc={yloc:.3f}, radius={radius:.3f}, amplitude={amplitude:.3f}")
            run_lidcavity_with_ic(Re, xloc, yloc, radius, amplitude, save_dir, num_steps)
            print(f"Finished simulation {count}, results saved in {save_dir}")
            count += 1


if __name__ == "__main__":
    main()
