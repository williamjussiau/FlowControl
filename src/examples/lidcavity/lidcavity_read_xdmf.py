"""
----------------------------------------------------------------------
Lid-driven cavity
Nondimensional incompressible Navier-Stokes equations
Supercritical Hopf bifurcation near Re_c=7700
----------------------------------------------------------------------
This file demonstrates the following possibilites:
    - Read XDMF files and store them into numpy arrays
----------------------------------------------------------------------
"""

from pathlib import Path

import dolfin
import numpy as np

import flowcontrol.flowsolverparameters as flowsolverparameters
import utils.utils_flowsolver as flu
from examples.lidcavity.lidcavityflowsolver import LidCavityFlowSolver
from flowcontrol.actuator import ActuatorBCParabolicV
from flowcontrol.sensor import SENSOR_TYPE, SensorPoint


def main():
    ## Initialize LidCavityFlowSolver to have access to dolfin.Function and dolfin.FunctionSpace
    cwd = Path(__file__).parent

    params_flow = flowsolverparameters.ParamFlow(Re=8000, uinf=1)
    params_flow.user_data["D"] = 1.0

    params_time = flowsolverparameters.ParamTime(num_steps=100, dt=0.005, Tstart=0.0)

    params_save = flowsolverparameters.ParamSave(
        save_every=20, path_out=cwd / "data_output"
    )

    params_solver = flowsolverparameters.ParamSolver(
        throw_error=True, is_eq_nonlinear=True, shift=0.0
    )

    params_mesh = flowsolverparameters.ParamMesh(
        meshpath=cwd / "data_input" / "mesh128.xdmf"
    )
    # mesh is in upper-right quadrant
    params_mesh.user_data["yup"] = 1
    params_mesh.user_data["ylo"] = 0
    params_mesh.user_data["xri"] = 1
    params_mesh.user_data["xle"] = 0

    params_restart = flowsolverparameters.ParamRestart()

    actuator_bc_up = ActuatorBCParabolicV(angular_size_deg=10)
    sensor_1 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([0.05, 0.5]))
    params_control = flowsolverparameters.ParamControl(
        sensor_list=[sensor_1],
        actuator_list=[actuator_bc_up],
    )

    params_ic = flowsolverparameters.ParamIC(
        xloc=0.1, yloc=0.1, radius=0.1, amplitude=0.1
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
        verbose=10,
    )

    ##########################################################
    ##########################################################
    ##########################################################
    # XDMF files
    xdmf_files = [
        "U_restart0,0.xdmf",
        "Uprev_restart0,0.xdmf",
    ]

    # Allocate dolfin.Function
    U_field = dolfin.Function(fs.V)
    P_field = dolfin.Function(fs.P)
    UP_field = dolfin.Function(fs.W)

    # Nr of DOFs to allocate numpy arrays
    ndof_u = fs.V.dim()
    ndof_p = fs.P.dim()
    ndof = fs.W.dim()
    nsnapshots_max = 100

    # Data will be stored as lists of 2D arrays
    # each array has shape [ndof, nsnapshots]
    # because nr of snapshots is initially unknown
    # and potentially different for each file
    # if not: preallocate 3D array
    U_field_alldata = []
    P_field_alldata = []
    UP_field_alldata = []

    U_field_data = np.empty((ndof_u,))
    P_field_data = np.empty((ndof_p,))
    UP_field_data = np.empty((ndof,))

    for jfile, file_name in enumerate(xdmf_files):
        for icounter in range(nsnapshots_max):
            # try:
            print(f"Reading {jfile} file: {file_name}, counter={icounter}")

            flu.read_xdmf(file_name, U_field, "U", counter=icounter)
            U_field_data = np.asarray(
                U_field.vector().get_local()
            )  # .get_local() == [:]
            print(max(U_field_data))
            U_field_alldata.append(U_field_data)
            print(len(U_field_alldata))
            # except RuntimeError:
            #    print("EOF -- Reached End Of File")
            #    break
        # finally:
        #     print("coucou")


if __name__ == "__main__":
    main()
