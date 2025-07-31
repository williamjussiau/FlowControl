import logging
import time
from pathlib import Path

import dolfin
import numpy as np

import flowcontrol.flowsolverparameters as flowsolverparameters
import utils.utils_flowsolver as flu
from examples.cylinder.cylinderflowsolver import CylinderFlowSolver
from flowcontrol.actuator import ActuatorBCParabolicV
from flowcontrol.controller import Controller
from flowcontrol.sensor import SENSOR_TYPE, SensorPoint
from examples.cylinder.compute_steady_state import Re

def run_lidcavity_with_ic(Re, xloc, yloc, radius, amplitude, save_dir, num_steps=100):
    # LOG
    dolfin.set_log_level(dolfin.LogLevel.INFO)  # DEBUG TRACE PROGRESS INFO
    logger = logging.getLogger(__name__)
    FORMAT = "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]: %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.INFO)

    t000 = time.time()
    cwd = Path(__file__).parent

    logger.info("Trying to instantiate FlowSolver...")

    params_flow = flowsolverparameters.ParamFlow(Re=100, uinf=1.0)
    params_flow.user_data["D"] = 1.0

    params_time = flowsolverparameters.ParamTime(num_steps=num_steps, dt=0.005, Tstart=0.0)

    params_save = flowsolverparameters.ParamSave(
        save_every=5, path_out=save_dir
    )

    params_solver = flowsolverparameters.ParamSolver(
        throw_error=True, is_eq_nonlinear=True, shift=0.0
    )

    params_mesh = flowsolverparameters.ParamMesh(
        meshpath=cwd / "data_input" / "O1.xdmf"
    )
    params_mesh.user_data["xinf"] = 20
    params_mesh.user_data["xinfa"] = -10
    params_mesh.user_data["yinf"] = 10

    params_restart = flowsolverparameters.ParamRestart()

    # duplicate actuators (1 top, 1 bottom) but assign same control input to each
    angular_size_deg = 10
    actuator_bc_1 = ActuatorBCParabolicV(
        width=ActuatorBCParabolicV.angular_size_deg_to_width(
            angular_size_deg, params_flow.user_data["D"] / 2
        ),
        position_x=0.0,
    )
    actuator_bc_2 = ActuatorBCParabolicV(
        width=ActuatorBCParabolicV.angular_size_deg_to_width(
            angular_size_deg, params_flow.user_data["D"] / 2
        ),
        position_x=0.0,
    )
    sensor_feedback = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3, 0]))
    sensor_perf_1 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3.1, 1]))
    sensor_perf_2 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3.1, -1]))
    params_control = flowsolverparameters.ParamControl(
        sensor_list=[sensor_feedback, sensor_perf_1, sensor_perf_2],
        actuator_list=[actuator_bc_1, actuator_bc_2],
    )

    params_ic = flowsolverparameters.ParamIC(
        xloc=xloc, yloc=yloc, radius=radius, amplitude=amplitude
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

    return


def main():

    base_dir = Path("/Users/jaking/Desktop/PhD/cylinder")
    parent_dir = base_dir / f"Re{Re}_test"
    parent_dir.mkdir(parents=True, exist_ok=True)

    # x_vals = np.linspace(0.2, 0.8, 3)
    # y_vals = np.linspace(0.2, 0.8, 3)
    # x_vals = np.linspace(0.2, 0.2, 1)
    # y_vals = np.linspace(0.2, 0.2, 1)
    x_vals = [2.0]
    y_vals = [0.0]
    radius = 0.5
    amplitude = 1.0
    num_steps = 1000
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
