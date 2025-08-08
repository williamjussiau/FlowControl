"""
----------------------------------------------------------------------
Flow past a cylinder
Nondimensional incompressible Navier-Stokes equations
Supercritical Hopf bifurcation at Re_c=46
Suggested Re=100
----------------------------------------------------------------------
This file demonstrates the following possibilites:
    - Initialize CylinderFlowSolver object
    - Compute steady-state
    - Load controller from file
    - Perform closed-loop time simulation
    - Restart simulation
----------------------------------------------------------------------
"""

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
from dolfin import div, grad, project

Re = 100

def main():
    # LOG
    dolfin.set_log_level(dolfin.LogLevel.INFO)  # DEBUG TRACE PROGRESS INFO
    logger = logging.getLogger(__name__)
    FORMAT = "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]: %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)

    t000 = time.time()
    cwd = Path(__file__).parent

    logger.info("Trying to instantiate FlowSolver...")

    params_flow = flowsolverparameters.ParamFlow(Re=Re, uinf=1.0)
    params_flow.user_data["D"] = 1.0

    params_time = flowsolverparameters.ParamTime(num_steps=10, dt=0.005, Tstart=0.0)

    params_save = flowsolverparameters.ParamSave(
        save_every=5, path_out=cwd / "data_output"
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
    uctrl0 = [1.0, 1.0]
    fs.compute_steady_state(method="picard", max_iter=3, tol=1e-7, u_ctrl=uctrl0)
    fs.compute_steady_state(
        method="newton", max_iter=25, u_ctrl=uctrl0, initial_guess=fs.fields.UP0
    )
    # Expected:
    # Newton iteration 4: r (abs) = 6.901e-14 (tol = 1.000e-10) r (rel) = 1.109e-11 (tol = 1.000e-09)

    point = dolfin.Point(3.0, 0.0)

    # Split mixed function into velocity and pressure
    u, p = fs.fields.UP0.split()

    # Evaluate at the point
    u_val = u(point)
    p_val = p(point)

    print(f"Velocity at (3, 0): {u_val}")
    print(f"Pressure at (3, 0): {p_val}")

    mesh = u.function_space().mesh()
    family = u.function_space().ufl_element().family()
    degree = u.function_space().ufl_element().degree()

    # Project gradient (if you want)
    from dolfin import TensorFunctionSpace
    W = TensorFunctionSpace(mesh, family, degree)
    grad_u_proj = project(grad(u), W)
    grad_u_val = grad_u_proj(point)
    print(f"Gradient of u at (3, 0): {grad_u_val}")

    # Project Laplacian
    from dolfin import VectorFunctionSpace

    V_vec = VectorFunctionSpace(mesh, family, degree)
    laplace_u_proj = project(div(grad(u)), V_vec)
    laplace_u_val = laplace_u_proj(point)
    print(f"Laplacian of u at (3, 0): {laplace_u_val}")


if __name__ == "__main__":
    main()
