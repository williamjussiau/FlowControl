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

# LOG
dolfin.set_log_level(dolfin.LogLevel.INFO)  # DEBUG TRACE PROGRESS INFO
logger = logging.getLogger(__name__)
FORMAT = "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG)


t000 = time.time()
cwd = Path(__file__).parent

logger.info("Trying to instantiate FlowSolver...")

params_flow = flowsolverparameters.ParamFlow(Re=100, uinf=1.0)
params_flow.user_data["D"] = 1.0

params_time = flowsolverparameters.ParamTime(num_steps=10, dt=0.005, Tstart=0.0)

params_save = flowsolverparameters.ParamSave(save_every=5, path_out=cwd / "data_output")

params_solver = flowsolverparameters.ParamSolver(
    throw_error=True, is_eq_nonlinear=True, shift=0.0
)

params_mesh = flowsolverparameters.ParamMesh(meshpath=cwd / "data_input" / "O1.xdmf")
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

params_ic = flowsolverparameters.ParamIC(xloc=2.0, yloc=0.0, radius=0.5, amplitude=1.0)

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
## Demonstrate restart
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
    "Last line should be: 10  0.100  0.000000  0.131695  0.009738  0.009810  0.122620  xxxxxxxx"
)

logger.info("End with success")
