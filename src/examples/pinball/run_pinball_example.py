import logging
import time
from pathlib import Path

import dolfin
import numpy as np

import flowcontrol.flowsolverparameters as flowsolverparameters
import utils.utils_flowsolver as flu
from examples.pinball.pinballflowsolver import PinballFlowSolver
from flowcontrol.actuator import ActuatorBCParabolicV, ActuatorBCRotation
from flowcontrol.controller import Controller
from flowcontrol.sensor import SENSOR_TYPE, SensorPoint


def main():
    # LOG
    dolfin.set_log_level(dolfin.LogLevel.INFO)  # DEBUG TRACE PROGRESS INFO
    logger = logging.getLogger(__name__)
    t000 = time.time()
    cwd = Path(__file__).parent

    logger.info("Trying to instantiate FlowSolver...")

    # All parameters
    params_flow = flowsolverparameters.ParamFlow(Re=50, uinf=1.0)
    params_flow.user_data["D"] = 1.0

    params_time = flowsolverparameters.ParamTime(num_steps=200, dt=0.005, Tstart=0.0)

    params_save = flowsolverparameters.ParamSave(
        save_every=2, path_out=cwd / "data_output"
    )

    params_solver = flowsolverparameters.ParamSolver(
        throw_error=True, is_eq_nonlinear=True, shift=0.0
    )

    params_mesh = flowsolverparameters.ParamMesh(
        meshpath=cwd / "data_input" / "mesh_middle_gmsh.xdmf"
    )
    params_mesh.user_data["xinf"] = 20
    params_mesh.user_data["xinfa"] = -6
    params_mesh.user_data["yinf"] = 6

    params_restart = flowsolverparameters.ParamRestart()

    # Actuators
    mode_actuation = "rot"  # "suc" or "rot"
    params_flow.user_data["mode_actuation"] = mode_actuation

    if mode_actuation == "suc":
        angular_size_deg = 10
        actuator_charm_bc = ActuatorBCParabolicV(
            width=ActuatorBCParabolicV.angular_size_deg_to_width(
                angular_size_deg, params_flow.user_data["D"] / 2
            ),
            position_x=-1.5 * np.cos(np.pi / 6),
        )
        actuator_top_bc = ActuatorBCParabolicV(
            width=ActuatorBCParabolicV.angular_size_deg_to_width(
                angular_size_deg, params_flow.user_data["D"] / 2
            ),
            position_x=0.0,
        )
        actuator_bottom_bc = ActuatorBCParabolicV(
            width=ActuatorBCParabolicV.angular_size_deg_to_width(
                angular_size_deg, params_flow.user_data["D"] / 2
            ),
            position_x=0.0,
        )
    elif mode_actuation == "rot":
        actuator_charm_bc = ActuatorBCRotation(
            position_x=-1.5 * np.cos(np.pi / 6),
            position_y=-1.5 * np.sin(np.pi / 6),
        )
        actuator_top_bc = ActuatorBCRotation(
            position_x=0.0,
            position_y=+0.75,
        )
        actuator_bottom_bc = ActuatorBCRotation(
            position_x=0.0,
            position_y=-0.75,
        )

    else:
        raise ValueError(f"Unknown actuation mode : {mode_actuation}")

    logger.info(f"Actuation mode : {mode_actuation.upper()}")
    # Sensors
    sensor_feedback = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([8, 0]))
    sensor_perf_1 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([10, 0]))
    sensor_perf_2 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([12, 0]))
    params_control = flowsolverparameters.ParamControl(
        sensor_list=[sensor_feedback, sensor_perf_1, sensor_perf_2],
        actuator_list=[actuator_charm_bc, actuator_top_bc, actuator_bottom_bc],
    )

    # IC
    params_ic = flowsolverparameters.ParamIC(
        xloc=2.0, yloc=0.0, radius=0.5, amplitude=1.0
    )

    fs = PinballFlowSolver(
        params_flow=params_flow,
        params_time=params_time,
        params_save=params_save,
        params_solver=params_solver,
        params_mesh=params_mesh,
        params_restart=params_restart,
        params_control=params_control,
        params_ic=params_ic,
        verbose=10,
    )

    logger.info("__init__(): successful!")

    logger.info("Exporting subdomains...")
    flu.export_subdomains(
        fs.mesh, fs.boundaries.subdomain, cwd / "data_output" / "subdomains.xdmf"
    )

    logger.info("Compute steady state...")
    uctrl0 = [0.0, 0.0, 0.0]

    fs.compute_steady_state(method="picard", max_iter=10, tol=1e-7, u_ctrl=uctrl0)
    fs.compute_steady_state(
        method="newton", max_iter=10, u_ctrl=uctrl0, initial_guess=fs.fields.UP0
    )
    # fs.load_steady_state()

    logger.info("Init time-stepping")
    fs.initialize_time_stepping(ic=None)  # or ic=dolfin.Function(fs.W)

    logger.info("Step several times")
    # Kss = Controller.from_file(file=cwd / "data_input" / "Kdx8dy0p0.mat", x0=0)
    tlen = 0.10  # characteristic length of gaussian bump
    tpeak = [0.25, 0.5, 0.75]  # peaking time
    u0peak = [1.0, -1.5, 2.0]  # peaking amplitude

    # fs.get_B(export='true',)
    def gaussian_bump(t, tpeak):
        return np.exp(-1 / 2 * (t - tpeak) ** 2 / tlen**2)

    for _ in range(fs.params_time.num_steps):
        # y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas)
        # u_ctrl = Kss.step(y=+y_meas[0], dt=fs.params_time.dt)
        u_ctrlc = u0peak[0] * gaussian_bump(fs.t, tpeak[0])
        u_ctrlt = u0peak[1] * gaussian_bump(fs.t, tpeak[1])
        u_ctrlb = u0peak[2] * gaussian_bump(fs.t, tpeak[2])
        fs.step(u_ctrl=[u_ctrlc, u_ctrlt, u_ctrlb])

    flu.summarize_timings(fs, t000)
    logger.info(fs.timeseries)
    fs.write_timeseries()


if __name__ == "__main__":
    main()
