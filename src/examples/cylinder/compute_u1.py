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
from dolfin import div, grad, project, dot, inner, nabla_grad, dx

Re = 100

def solve_linearized_steady_state(fs, U0_base, u_ctrl, max_iter=50, tol=1e-8):
    """
    Solve linearized Navier-Stokes around U0_base under constant control u_ctrl
    """
    logger = logging.getLogger(__name__)
    
    # Create boundary conditions with control input
    fs.set_actuators_u_ctrl(u_ctrl)
    BC = fs._make_BCs()
    
    # Get function spaces
    W = fs.W
    
    # Define trial and test functions for perturbation
    up = dolfin.TrialFunction(W)
    vq = dolfin.TestFunction(W)
    u, p = dolfin.split(up)
    v, q = dolfin.split(vq)
    
    # Extract base flow velocity
    U0, P0 = U0_base.split()
    U0_vec = dolfin.as_vector((U0[0], U0[1]))
    
    # Reynolds number
    invRe = dolfin.Constant(1.0 / fs.params_flow.Re)
    
    # Linearized weak form around U0_base
    # (U0 · ∇)u + (u · ∇)U0 + ∇p - (1/Re)∇²u = 0
    # ∇ · u = 0
    a_linearized = (
        dot(dot(U0_vec, nabla_grad(u)), v) * dx +  # Convection by base flow
        dot(dot(u, nabla_grad(U0_vec)), v) * dx +  # Linearized convection
        invRe * inner(nabla_grad(u), nabla_grad(v)) * dx +  # Viscous term
        p * div(v) * dx +  # Pressure gradient
        div(u) * q * dx   # Incompressibility
    )
    
    # Zero right-hand side (steady state) - only test functions allowed
    L_linearized = dolfin.Constant(0.0) * v[0] * dx + dolfin.Constant(0.0) * q * dx
    
    # Assemble system
    A = dolfin.assemble(a_linearized)
    b = dolfin.assemble(L_linearized)
    
    # Apply boundary conditions (these include the control input)
    # Access the actual boundary condition lists from the BoundaryConditions object
    for bc in BC.bcu:  # velocity boundary conditions
        bc.apply(A, b)
    for bc in BC.bcp:  # pressure boundary conditions (if any)
        bc.apply(A, b)
    
    # Solve
    up_solution = dolfin.Function(W)
    solver = dolfin.LUSolver("mumps")
    solver.solve(A, up_solution.vector(), b)
    
    # Verify residual
    residual = dolfin.assemble(dolfin.action(a_linearized, up_solution))
    for bc in BC.bcu:
        bc.apply(residual)
    for bc in BC.bcp:
        bc.apply(residual)
    res_norm = dolfin.norm(residual)
    
    logger.info(f"Linearized steady state solved. Residual norm: {res_norm:.2e}")
    
    return up_solution

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

    logger.info("Compute uncontrolled steady state (base flow)...")
    uctrl0 = [0, 0]
    fs.compute_steady_state(method="picard", max_iter=3, tol=1e-7, u_ctrl=uctrl0)
    fs.compute_steady_state(
        method="newton", max_iter=25, u_ctrl=uctrl0, initial_guess=fs.fields.UP0
    )
    
    # Save base flow
    U0_base = fs.fields.UP0.copy(deepcopy=True)
    U0, P0 = U0_base.split()
    
    # Save uncontrolled steady state
    u0_file = dolfin.XDMFFile(str(cwd / "data_output" / "U0_uncontrolled.xdmf"))
    p0_file = dolfin.XDMFFile(str(cwd / "data_output" / "P0_uncontrolled.xdmf"))
    u0_file.write(U0, 0.0)
    p0_file.write(P0, 0.0)
    
    logger.info("Computing linearized steady state under control...")
    u_ctrl_linearized = [1.0, 1.0]  # Control input
    
    # Solve linearized system
    UP_linearized = solve_linearized_steady_state(fs, U0_base, u_ctrl_linearized)
    
    # Extract velocity and pressure
    U_lin, P_lin = UP_linearized.split()
    
    # Save linearized steady state
    u_lin_file = dolfin.XDMFFile(str(cwd / "data_output" / "U_linearized_controlled.xdmf"))
    p_lin_file = dolfin.XDMFFile(str(cwd / "data_output" / "P_linearized_controlled.xdmf"))
    u_lin_file.write(U_lin, 0.0)
    p_lin_file.write(P_lin, 0.0)
    
    # Also compute full nonlinear controlled steady state for comparison
    logger.info("Computing full nonlinear controlled steady state for comparison...")
    fs.compute_steady_state(method="picard", max_iter=10, tol=1e-7, u_ctrl=u_ctrl_linearized)
    fs.compute_steady_state(method="newton", max_iter=25, u_ctrl=u_ctrl_linearized, initial_guess=fs.fields.UP0)
    
    UP_nonlinear = fs.fields.UP0.copy(deepcopy=True)
    U_nonlin, P_nonlin = UP_nonlinear.split()
    
    # Save nonlinear controlled steady state
    u_nonlin_file = dolfin.XDMFFile(str(cwd / "data_output" / "U_nonlinear_controlled.xdmf"))
    p_nonlin_file = dolfin.XDMFFile(str(cwd / "data_output" / "P_nonlinear_controlled.xdmf"))
    u_nonlin_file.write(U_nonlin, 0.0)
    p_nonlin_file.write(P_nonlin, 0.0)
    
    # Compute difference between linearized and nonlinear solutions
    UP_diff = UP_linearized.copy(deepcopy=True)
    UP_diff.vector()[:] = UP_linearized.vector()[:] - UP_nonlinear.vector()[:]
    
    diff_file = dolfin.XDMFFile(str(cwd / "data_output" / "difference_lin_vs_nonlin.xdmf"))
    diff_file.write(UP_diff, 0.0)
    
    # Print some statistics
    diff_norm = dolfin.norm(UP_diff)
    lin_norm = dolfin.norm(UP_linearized)
    nonlin_norm = dolfin.norm(UP_nonlinear)
    
    logger.info(f"Linearized solution norm: {lin_norm:.6f}")
    logger.info(f"Nonlinear solution norm: {nonlin_norm:.6f}")
    logger.info(f"Difference norm: {diff_norm:.6f}")
    logger.info(f"Relative difference: {diff_norm/max(lin_norm, nonlin_norm):.6f}")
    
    logger.info("Analysis completed successfully.")

if __name__ == "__main__":
    main()