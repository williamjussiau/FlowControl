from examples.lidcavity.compute_steady_state_increasing_Re import Re_final as Re
from pathlib import Path
import numpy as np

def run_lidcavity_with_ic(Re, xloc, yloc, radius, amplitude, save_dir):
    import logging
    import time
    from pathlib import Path

    import dolfin
    import numpy as np

    import flowcontrol.flowsolverparameters as flowsolverparameters
    import utils.utils_flowsolver as flu
    from examples.lidcavity.lidcavityflowsolver import LidCavityFlowSolver
    from flowcontrol.actuator import ActuatorBCUniformU
    from flowcontrol.sensor import SENSOR_TYPE, SensorPoint

    t000 = time.time()
    cwd = Path(__file__).parent

    dolfin.set_log_level(dolfin.LogLevel.INFO)
    logger = logging.getLogger(__name__)
    FORMAT = "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]: %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)

    params_flow = flowsolverparameters.ParamFlow(Re=Re, uinf=1)
    params_flow.user_data["D"] = 1.0

    params_time = flowsolverparameters.ParamTime(num_steps=10000, dt=0.005, Tstart=0.0)

    params_save = flowsolverparameters.ParamSave(
        save_every=20, path_out=save_dir
    )

    params_solver = flowsolverparameters.ParamSolver(
        throw_error=True, is_eq_nonlinear=True, shift=0.0
    )

    params_mesh = flowsolverparameters.ParamMesh(
        meshpath=cwd / "data_input" / "lidcavity_5.xdmf"
    )
    params_mesh.user_data["yup"] = 1
    params_mesh.user_data["ylo"] = 0
    params_mesh.user_data["xri"] = 1
    params_mesh.user_data["xle"] = 0

    params_restart = flowsolverparameters.ParamRestart()

    actuator_bc_up = ActuatorBCUniformU()
    sensor_1 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([0.05, 0.5]))
    sensor_2 = SensorPoint(sensor_type=SENSOR_TYPE.U, position=np.array([0.5, 0.95]))
    sensor_3 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([0.5, 0.95]))
    params_control = flowsolverparameters.ParamControl(
        sensor_list=[sensor_1, sensor_2, sensor_3],
        actuator_list=[actuator_bc_up],
    )

    params_ic = flowsolverparameters.ParamIC(
        xloc=xloc, yloc=yloc, radius=radius, amplitude=amplitude
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

    logger.info("__init__(): successful!")

    logger.info("Exporting subdomains...")
    flu.export_subdomains(
        fs.mesh, fs.boundaries.subdomain, save_dir / "subdomains.xdmf"
    )

    logger.info("Load steady state...")
    fs.load_steady_state(
        path_u_p=[
            cwd / "data_output" / "steady" / f"U0_Re={Re}.xdmf",
            cwd / "data_output" / "steady" / f"P0_Re={Re}.xdmf",
        ]
    )

    logger.info("Init time-stepping")
    fs.initialize_time_stepping(ic=None)

    logger.info("Step several times")
    for _ in range(fs.params_time.num_steps):
        y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas)
        # This is still for unforced simulations
        # TODO: Add forcing to the workflow
        u_ctrl = [0 * y_meas[0]]
        fs.step(u_ctrl=[u_ctrl[0]])

    xdmf_files = [
    cwd / "data_output" / "@_restart0,0.xdmf",
    # cwd / "data_output" / "@_restart0,0.xdmf",
    ]

    def make_file_name_for_field(field_name, original_file_name):
        "Make file name for field U or P, using provided original"
        "file name with placeholder @"
        return str(original_file_name).replace("@", field_name)

    # Data will be stored as 3D arrays
    # of size [ndof*, nsnapshots, nfiles]
    nsnapshots = fs.params_time.num_steps // fs.params_save.save_every
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
    flu.read_xdmf(file_name_U0, U_field, "U0")
    flu.read_xdmf(file_name_P0, P_field, "P0")
    UP0 = fs.merge(U0, P0)

    U0_field_data = U0.vector().get_local()
    P0_field_data = P0.vector().get_local()
    UP0_field_data = UP0.vector().get_local()

    print("Finished reading steady-states")

    np.save(save_dir / "U_field_alldata.npy", U_field_alldata)
    np.save(save_dir / "P_field_alldata.npy", P_field_alldata)
    np.save(save_dir / "UP_field_alldata.npy", UP_field_alldata)

    np.save(save_dir / "U0_field_data.npy", U0_field_data)
    np.save(save_dir / "P0_field_data.npy", P0_field_data)
    np.save(save_dir / "UP0_field_data.npy", UP0_field_data)

if __name__ == "__main__":
    # Adapt to wherever you want to save the results
    base_dir = Path("/Users/james/Desktop/PhD/lid_driven_cavity")
    parent_dir = base_dir / f"Re{Re}"
    parent_dir.mkdir(parents=True, exist_ok=True)

    x_vals = np.linspace(0.2, 0.8, 3)
    y_vals = np.linspace(0.2, 0.8, 3)
    radius = 0.1
    amplitude = 0.1
    count = 1
    for xloc in x_vals:
        for yloc in y_vals:
            save_dir = parent_dir / f"run{count}"
            save_dir.mkdir(parents=True, exist_ok=True)

            print(f"Running simulation {count} with xloc={xloc:.3f}, yloc={yloc:.3f}, radius={radius:.3f}, amplitude={amplitude:.3f}")
            run_lidcavity_with_ic(Re, xloc, yloc, radius, amplitude, save_dir)
            print(f"Finished simulation {count}, results saved in {save_dir}")
            count += 1