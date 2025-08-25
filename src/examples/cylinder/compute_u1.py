"""
----------------------------------------------------------------------
Flow past a cylinder
Nondimensional incompressible Navier-Stokes equations
Supercritical Hopf bifurcation at Re_c=46
Suggested Re=100
----------------------------------------------------------------------
This file demonstrates the following possibilites:
    - Solve the linearized steady-state Navier-Stokes equations around a base flow
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
    
    # Get function spaces
    W = fs.W
    
    # Define trial and test functions
    up = dolfin.TrialFunction(W)
    vq = dolfin.TestFunction(W)
    u, p = dolfin.split(up)
    v, q = dolfin.split(vq)
    
    # Extract base flow
    U0, P0 = U0_base.split()
    U0_vec = dolfin.as_vector((U0[0], U0[1]))
    
    # Reynolds number
    invRe = dolfin.Constant(1.0 / fs.params_flow.Re)
    
    # Linearized operator
    a_linearized = (
        dot(dot(U0_vec, nabla_grad(u)), v) * dx  # Convection by base flow
        + dot(dot(u, nabla_grad(U0_vec)), v) * dx  # Linearized convection
        + invRe * inner(nabla_grad(u), nabla_grad(v)) * dx  # Viscous term
        - p * div(v) * dx  # Pressure term
        - div(u) * q * dx  # Incompressibility
    )
    
    # RHS = 0 for homogeneous linearized problem
    L_linearized = dolfin.Constant(0.0) * dot(v, dolfin.as_vector([1, 1])) * dx
    
    # Set control input for actuators
    fs.set_actuators_u_ctrl(u_ctrl)
    
    # Debug: check control values
    logger.info(f"Control input: {u_ctrl}")
    
    # Boundary conditions for perturbation field
    bcu_inlet_pert = dolfin.DirichletBC(
        fs.W.sub(0),
        dolfin.Constant((0, 0)),
        fs.get_subdomain("inlet"),
    )
    
    bcu_walls_pert = dolfin.DirichletBC(
        fs.W.sub(0).sub(1),
        dolfin.Constant(0),
        fs.get_subdomain("walls"),
    )
    
    bcu_cylinder_pert = dolfin.DirichletBC(
        fs.W.sub(0),
        dolfin.Constant((0, 0)),
        fs.get_subdomain("cylinder"),
    )
    
    bcu_actuation_up_pert = dolfin.DirichletBC(
        fs.W.sub(0),
        fs.params_control.actuator_list[0].expression,
        fs.get_subdomain("actuator_up"),
    )
    bcu_actuation_lo_pert = dolfin.DirichletBC(
        fs.W.sub(0),
        fs.params_control.actuator_list[1].expression,
        fs.get_subdomain("actuator_lo"),
    )

    bcu = [bcu_inlet_pert, bcu_walls_pert, bcu_cylinder_pert,
           bcu_actuation_up_pert, bcu_actuation_lo_pert]
    
    # Assemble system
    A = dolfin.assemble(a_linearized)
    b = dolfin.assemble(L_linearized)
    
    # Debug: check RHS norm before BC application
    rhs_norm_before = b.norm("l2")
    logger.info(f"RHS norm before BC application: {rhs_norm_before:.2e}")
    
    # Apply boundary conditions
    for bc in bcu:
        bc.apply(A, b)
    
    # Debug: check RHS norm after BC application
    rhs_norm_after = b.norm("l2")
    logger.info(f"RHS norm after BC application: {rhs_norm_after:.2e}")
    
    # Solve with simpler solver settings
    up_perturbation = dolfin.Function(W)
    solver = dolfin.LUSolver("mumps")
    solver.solve(A, up_perturbation.vector(), b)
    
    # Compute residual more carefully
    residual_vec = A * up_perturbation.vector() - b
    res_norm = residual_vec.norm("l2")
    
    # Also check the solution norm
    sol_norm = up_perturbation.vector().norm("l2")
    logger.info(f"Solution norm: {sol_norm:.2e}")
    logger.info(f"Residual norm: {res_norm:.2e}")
    logger.info(f"Relative residual: {res_norm/max(sol_norm, 1e-12):.2e}")
    
    # Check if residual is acceptable
    if res_norm > 1e-6:
        logger.warning(f"Large residual norm {res_norm:.2e} - solution may not be accurate")
    
    return up_perturbation

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
    
    # Save uncontrolled steady state using flu.write_xdmf
    # flu.write_xdmf(
    #     cwd / "data_output" / "U0_uncontrolled.xdmf",
    #     U0,
    #     "U0",
    #     time_step=0.0,
    #     append=False,
    #     write_mesh=True,
    # )
    
    # flu.write_xdmf(
    #     cwd / "data_output" / "P0_uncontrolled.xdmf",
    #     P0,
    #     "P0",
    #     time_step=0.0,
    #     append=False,
    #     write_mesh=True,
    # )
    
    logger.info("Computing linearized steady state under control...")
    u_ctrl = [1.0, 1.0]  # Control input
    
    # Solve linearized system
    UP_lin = solve_linearized_steady_state(fs, U0_base, u_ctrl)
    
    # Extract velocity and pressure
    U_lin, P_lin = UP_lin.split()
    
    # Save linearized steady state using flu.write_xdmf
    flu.write_xdmf(
        cwd / "data_output" / "U_linearized_controlled.xdmf",
        U_lin,
        "U_lin",
        time_step=0.0,
        append=False,
        write_mesh=True,
    )
    
    flu.write_xdmf(
        cwd / "data_output" / "P_linearized_controlled.xdmf",
        P_lin,
        "P_lin",
        time_step=0.0,
        append=False,
        write_mesh=True,
    )

    logger.info("Analysis completed successfully.")

    # Extract only velocity DOFs
    velocity_dofs = U_lin.function_space().dofmap().dofs()
    U_lin_field_data = U_lin.vector().get_local()[velocity_dofs]

    # Extract only pressure DOFs
    pressure_dofs = P_lin.function_space().dofmap().dofs()
    P_lin_field_data = P_lin.vector().get_local()[pressure_dofs]
    UP_lin_field_data = UP_lin.vector().get_local()

    np.save(cwd / "data_output" / "U_lin_field_data_unit_control.npy", U_lin_field_data)
    np.save(cwd / "data_output" / "P_lin_field_data_unit_control.npy", P_lin_field_data)
    np.save(cwd / "data_output" / "UP_lin_field_data_unit_control.npy", UP_lin_field_data)


if __name__ == "__main__":
    main()