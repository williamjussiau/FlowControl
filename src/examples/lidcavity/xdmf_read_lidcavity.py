"""
----------------------------------------------------------------------
Lid-driven cavity
Nondimensional incompressible Navier-Stokes equations
Supercritical Hopf bifurcation near Re_c=7700
----------------------------------------------------------------------
This file demonstrates how to read XDMF files and store
fields into numpy arrays
----------------------------------------------------------------------
"""
Re = 8000  # Reynolds number

from pathlib import Path

import dolfin
import numpy as np

import flowcontrol.flowsolverparameters as flowsolverparameters
import utils.utils_flowsolver as flu
from examples.lidcavity.lidcavityflowsolver import LidCavityFlowSolver
from flowcontrol.actuator import ActuatorBCParabolicV
from flowcontrol.sensor import SENSOR_TYPE, SensorPoint


def main():
    ##########################################################
    ## Initialize LidCavityFlowSolver to have access to dolfin.Function and dolfin.FunctionSpace
    ##########################################################
    cwd = Path(__file__).parent

    params_flow = flowsolverparameters.ParamFlow(Re=Re, uinf=1)
    params_flow.user_data["D"] = 1.0

    params_time = flowsolverparameters.ParamTime(num_steps=100, dt=0.005, Tstart=0.0)

    params_save = flowsolverparameters.ParamSave(
        save_every=20, path_out=cwd / "data_output"
    )

    params_solver = flowsolverparameters.ParamSolver(
        throw_error=True, is_eq_nonlinear=True, shift=0.0
    )

    params_mesh = flowsolverparameters.ParamMesh(
        meshpath=cwd / "data_input" / "lidcavity_5.xdmf"
    )
    # mesh is in upper-right quadrant
    params_mesh.user_data["yup"] = 1
    params_mesh.user_data["ylo"] = 0
    params_mesh.user_data["xri"] = 1
    params_mesh.user_data["xle"] = 0

    params_restart = flowsolverparameters.ParamRestart()

    angular_size_deg = 10
    actuator_bc_up = ActuatorBCParabolicV(
        width=ActuatorBCParabolicV.angular_size_deg_to_width(
            angular_size_deg=angular_size_deg,
            cylinder_radius=params_flow.user_data["D"] / 2,
        ),
        position_x=0.0,
    )
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
    # Foreword
    ##########################################################
    # I recommend reading: https://github.com/williamjussiau/FlowControl/wiki/Numerical-details

    #  - when velocity and pressure fields are computed,
    #       they are stored in different variables
    #  - it is possible to merge them using FLowSolver.merge(u, p)
    #       into a mixed field living in W = V x P
    #  - be careful to indexing of degrees of freedom, it is nontrivial

    # What is computed by FlowSolver?
    #  - steady-state U0, P0
    #  - perturbation fields u, p

    # Which field is stored by FlowSolver?
    #  - steady-state U0, P0
    #  - full field U=U0+u, P=P0+p
    #  --> it is what is being read here

    # When is it saved?
    #  - fields are saved every save_every steps, corresponding
    #      to the time t=save_every*dt
    #  - the previous fields are also saved for restarting purposes,
    #      they correspond to time t-dt

    # How to read?
    #  - cf. code below
    #  - be careful to field names: U/P for current, U0/P0 for
    #      steady-state, and U_n/P_n for previous

    # How is it being read?
    #  - reading a field stores content in a dolfin.Function f
    #  - we can access the contents with f.vector().get_local() or f.vector()[:]
    #  - usually, everything is passed by reference, so do not forget to
    #       copy arrays when storing them

    ##########################################################
    # XDMF files
    ##########################################################
    # xdmf_files = [
    #     "src/examples/lidcavity/data_output/@_restart0,0.xdmf",
    #     "src/examples/lidcavity/data_output/@_restart0,0.xdmf",
    # ]

    xdmf_files = [
    cwd / "data_output" / "@_restart0,0.xdmf",
    cwd / "data_output" / "@_restart0,0.xdmf",
    ]

    def make_file_name_for_field(field_name, original_file_name):
        "Make file name for field U or P, using provided original"
        "file name with placeholder @"
        return str(original_file_name).replace("@", field_name)

    # Data will be stored as 3D arrays
    # of size [ndof*, nsnapshots, nfiles]
    nsnapshots = 3
    ndof_u = fs.V.dim()
    ndof_p = fs.P.dim()
    ndof = fs.W.dim()
    nfiles = len(xdmf_files)

    # Allocate arrays for 1 trajectory (= 1 file)
    U_field_data = np.empty((ndof_u, nsnapshots))
    P_field_data = np.empty((ndof_p, nsnapshots))
    UP_field_data = np.empty((ndof, nsnapshots))

    # Allocate arrays for all data (stack files on axis 2)
    U_field_alldata = np.empty((ndof_u, nsnapshots, nfiles))
    P_field_alldata = np.empty((ndof_p, nsnapshots, nfiles))
    UP_field_alldata = np.empty((ndof, nsnapshots, nfiles))

    # Allocate empty dolfin.Function
    U_field = dolfin.Function(fs.V)
    P_field = dolfin.Function(fs.P)
    UP_field = dolfin.Function(fs.W)

    for jfile in range(nfiles):
        print(f"* Reading file nr={jfile}, name={xdmf_files[jfile]}")

        file_name_U = make_file_name_for_field("U", xdmf_files[jfile])
        file_name_P = make_file_name_for_field("P", xdmf_files[jfile])

        for icounter in range(nsnapshots):
            print(f"\t counter={icounter}")

            try:
                flu.read_xdmf(file_name_U, U_field, "U", counter=icounter)
                flu.read_xdmf(file_name_P, P_field, "P", counter=icounter)
                UP_field = fs.merge(U_field, P_field)
            except RuntimeError:
                print("\t *** EOF -- Reached End Of File")
                break

            U_field_data[:, icounter] = np.copy(U_field.vector().get_local())
            P_field_data[:, icounter] = np.copy(P_field.vector().get_local())
            UP_field_data[:, icounter] = np.copy(UP_field.vector().get_local())

        U_field_alldata[:, :, jfile] = np.copy(U_field_data)
        P_field_alldata[:, :, jfile] = np.copy(P_field_data)
        UP_field_alldata[:, :, jfile] = np.copy(UP_field_data)

        print(f"\t -> Reached snapshot = {icounter} - Fetching next file")

    print("Finished reading trajectores")

    ##########################################################
    # Steady-state
    ##########################################################
    U0 = dolfin.Function(fs.V)
    P0 = dolfin.Function(fs.P)

    # file_name_U0 = "src/examples/lidcavity/data_output/steady/U0.xdmf"
    # file_name_P0 = "src/examples/lidcavity/data_output/steady/P0.xdmf"

    file_name_U0 = cwd / "data_output" / "steady" / "U0.xdmf"
    file_name_P0 = cwd / "data_output" / "steady" / "P0.xdmf"

    print(f"* Reading steady-states at {file_name_U0}, {file_name_P0}")
    # flu.read_xdmf(file_name_U0, U_field, "U0")
    # flu.read_xdmf(file_name_P0, P_field, "P0")
    flu.read_xdmf(file_name_U0, U0, "U0")
    flu.read_xdmf(file_name_P0, P0, "P0")
    UP0 = fs.merge(U0, P0)

    U0_field_data = U0.vector().get_local()
    P0_field_data = P0.vector().get_local()
    UP0_field_data = UP0.vector().get_local()

    print("Finished reading steady-states")

    save_dir = Path("/Users/james/Desktop/PhD/lid_driven_cavity")
    save_dir.mkdir(parents=True, exist_ok=True)  # Create the folder if it doesn't exist

    np.save(save_dir / "U_field_alldata.npy", U_field_alldata)
    np.save(save_dir / "P_field_alldata.npy", P_field_alldata)
    np.save(save_dir / "UP_field_alldata.npy", UP_field_alldata)

    np.save(save_dir / "U0_field_data.npy", U0_field_data)
    np.save(save_dir / "P0_field_data.npy", P0_field_data)
    np.save(save_dir / "UP0_field_data.npy", UP0_field_data)


if __name__ == "__main__":
    main()
