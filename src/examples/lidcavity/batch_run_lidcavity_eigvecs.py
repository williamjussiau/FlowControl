from examples.lidcavity.compute_steady_state_increasing_Re import Re_final as Re
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import scipy.io

def cleanup_redundant_files(save_dir):
    """Delete redundant Uprev files to save disk space"""
    import os
    from pathlib import Path
    
    patterns_to_delete = [
        "Uprev_restart0,0.h5",
        "Uprev_restart0,0.xdmf", 
        "Pprev_restart0,0.h5",
        "Pprev_restart0,0.xdmf"
    ]
    
    deleted_files = []
    for pattern in patterns_to_delete:
        file_path = save_dir / pattern
        if file_path.exists():
            try:
                os.remove(file_path)
                deleted_files.append(pattern)
                print(f"✓ Deleted redundant file: {pattern}")
            except Exception as e:
                print(f"⚠ Failed to delete {pattern}: {e}")
    
    if deleted_files:
        print(f"Cleaned up {len(deleted_files)} redundant files")
    else:
        print("No redundant files found to clean up")

def load_eigendata(mat_file_path, num_eigenvectors=1):
    """Load eigendata from .mat file and return the last num_eigenvectors (fastest growing modes)
    
    Args:
        mat_file_path: Path to .mat file with structured 'eig_data' containing 'lambda' and 'vec' fields
        num_eigenvectors: Number of eigenvectors to extract (default 1, the ones with largest real value)
    
    Returns:
        eigenvalues: Real parts of the last num_eigenvectors eigenvalues 
        eigenvectors: Real parts of corresponding eigenvectors (ndof x num_eigenvectors)
    """
    import scipy.io
    import numpy as np
    
    print(f"Loading eigendata from {mat_file_path}")
    
    # Load .mat file
    mat_data = scipy.io.loadmat(mat_file_path)
    
    # Extract the structured data
    eig_data = mat_data['eig_data']
    
    # Access the lambda and vec fields from the structured array
    eigenvalues_complex_raw = eig_data['lambda'][0]  # Array of arrays
    eigenvectors_complex = eig_data['vec'][0]        # List/array of eigenvectors
    
    # Flatten the eigenvalues to a simple 1D array
    eigenvalues_complex = np.array([val.flatten()[0] for val in eigenvalues_complex_raw])
    
    print(f"Loaded {len(eigenvalues_complex)} eigenvalues")
    print(f"Number of eigenvectors: {len(eigenvectors_complex)}")
    print(f"First eigenvector shape: {eigenvectors_complex[0].shape}")
    
    # Take the last num_eigenvectors (fastest growing modes since sorted by descending real part)
    eigenvalues_selected = eigenvalues_complex[:num_eigenvectors]
    eigenvectors_selected_list = eigenvectors_complex[:num_eigenvectors]
    
    # Stack the individual eigenvectors into a matrix (ndof x num_eigenvectors)
    eigenvectors_matrix = np.hstack([vec.flatten()[:, np.newaxis] for vec in eigenvectors_selected_list])
    
    print(f"Selected eigenvalues: {eigenvalues_selected}")
    print(f"Selected eigenvectors shape: {eigenvectors_matrix.shape}")
    
    return eigenvalues_selected, eigenvectors_matrix

def run_lidcavity_with_eigenvector_ic(Re, phase_angle, eigenvector_amplitude, forcing_frequency, forcing_amplitude, save_dir, num_steps=100):
    """Run lid cavity with eigenvector-based initial condition"""
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

    # Copy all parameter setup from your existing function
    params_flow = flowsolverparameters.ParamFlow(Re=Re, uinf=1)
    params_flow.user_data["D"] = 1.0

    params_time = flowsolverparameters.ParamTime(num_steps=num_steps, dt=0.005, Tstart=0.0)
    params_save = flowsolverparameters.ParamSave(save_every=20, path_out=save_dir)
    params_solver = flowsolverparameters.ParamSolver(throw_error=True, is_eq_nonlinear=True, shift=0.0)
    
    params_mesh = flowsolverparameters.ParamMesh(meshpath=cwd / "data_input" / "lidcavity_3.xdmf")
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
        sensor_list=[sensor_1, sensor_2, sensor_3], actuator_list=[actuator_bc_up]
    )

    # Use empty IC params to prevent automatic perturbation generation
    params_ic = flowsolverparameters.ParamIC(xloc=0.0, yloc=0.0, radius=0.0, amplitude=0.0)

    fs = LidCavityFlowSolver(
        params_flow=params_flow, params_time=params_time, params_save=params_save,
        params_solver=params_solver, params_mesh=params_mesh, params_restart=params_restart,
        params_control=params_control, params_ic=params_ic, verbose=10
    )

    logger.info("__init__(): successful!")

    logger.info("Exporting subdomains...")
    flu.export_subdomains(fs.mesh, fs.boundaries.subdomain, save_dir / "subdomains.xdmf")

    logger.info("Load steady state...")
    fs.load_steady_state(
        path_u_p=[
            cwd / "data_output" / "steady" / f"U0_Re={Re}.xdmf",
            cwd / "data_output" / "steady" / f"P0_Re={Re}.xdmf",
        ]
    )

    # Load eigendata and create eigenvector IC
    logger.info("Loading eigendata...")
    eigenvalue, eigenvector = load_eigendata(cwd / "data_output" / "eig_data.mat", num_eigenvectors=1)
    logger.info("Eigendata loading successful")

    eigenvector_real_part = np.real(eigenvector)
    eigenvector_imag_part = np.imag(eigenvector)

    perturbation_vector = (np.cos(phase_angle) * eigenvector_real_part + 
                          np.sin(phase_angle) * eigenvector_imag_part)

    # Create eigenvector IC
    logger.info("Creating eigenvector-based initial condition...")
        
    if len(eigenvector) == fs.W.dim():
        # Mixed space eigenvector
        logger.info("Processing mixed-space eigenvector (real part)")
        UP_pert_vec = eigenvector_amplitude * perturbation_vector
        
    else:
        raise ValueError(f"Eigenvector size {len(eigenvector)} doesn't match W ({fs.W.dim()})")

    logger.info("Creating steady state + perturbation IC...")
    UP_ic = dolfin.Function(fs.W)
    UP_ic.vector().set_local(UP_pert_vec)

    # Log perturbation info
    pert_norm = np.linalg.norm(UP_pert_vec)
    logger.info(f"Perturbation norm: {pert_norm:.6f}")

    logger.info("Init time-stepping with eigenvector IC")
    fs.initialize_time_stepping(ic=UP_ic)

    logger.info("Step several times")
    for _ in range(fs.params_time.num_steps):
        y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas)
        # This is still for unforced simulations
        # TODO: Add forcing to the workflow
        if fs.t < 100:
            u_ctrl = [0 * y_meas[0]]
        else:
            u_ctrl = [forcing_amplitude * np.sin(forcing_frequency * fs.t) + 0 * y_meas[0]]
        
        fs.step(u_ctrl=[u_ctrl[0]])

    xdmf_files = [
    save_dir / "@_restart0,0.xdmf",
    # cwd / "data_output" / "@_restart0,0.xdmf",
    ]

    logger.info("Cleanung up redundant prev files")
    cleanup_redundant_files(save_dir)
    logger.info("Finished cleaning up")

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
    flu.read_xdmf(file_name_U0, U0, "U0")
    flu.read_xdmf(file_name_P0, P0, "P0")
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

    ##########################################################
    # Extract mesh coordinates corresponding to DOF ordering
    ##########################################################
    print("Extracting mesh coordinates...")
    
    # Get DOF coordinates for velocity space (V)
    V_coords = fs.V.tabulate_dof_coordinates()
    
    # Get DOF coordinates for pressure space (P)
    P_coords = fs.P.tabulate_dof_coordinates()
    
    # Get DOF coordinates for mixed space (W)
    W_coords = fs.W.tabulate_dof_coordinates()
    
    print(f"V DOF coordinates shape: {V_coords.shape}")
    print(f"P DOF coordinates shape: {P_coords.shape}")
    print(f"W DOF coordinates shape: {W_coords.shape}")

    # Save mesh coordinates
    np.save(save_dir / "V_dof_coordinates.npy", V_coords)
    np.save(save_dir / "P_dof_coordinates.npy", P_coords)
    np.save(save_dir / "W_dof_coordinates.npy", W_coords)

    ##########################################################
    # DOF Indices and Mixed Space Mapping
    ##########################################################
    print("Computing DOF indices and mixed space mappings...")

    # Get velocity and pressure DOF indices in the mixed space
    vel_dofs_in_mixed = np.array(fs.W.sub(0).dofmap().dofs())
    pres_dofs_in_mixed = np.array(fs.W.sub(1).dofmap().dofs())

    # Get component DOFs for both spaces
    V_u_dofs = fs.V.sub(0).dofmap().dofs()
    V_v_dofs = fs.V.sub(1).dofmap().dofs()
    W_u_dofs = fs.W.sub(0).sub(0).dofmap().dofs()
    W_v_dofs = fs.W.sub(0).sub(1).dofmap().dofs()

    # Get coordinates for mapping
    # V_coords = fs.V.tabulate_dof_coordinates()
    # W_coords = fs.W.tabulate_dof_coordinates()

    # Build coordinate-based mapping between V and W velocity spaces
    V_u_coords = V_coords[V_u_dofs]
    V_v_coords = V_coords[V_v_dofs]
    W_u_coords = W_coords[W_u_dofs]
    W_v_coords = W_coords[W_v_dofs]

    # Create KDTrees for coordinate matching
    tree_u = cKDTree(W_u_coords)
    tree_v = cKDTree(W_v_coords)

    # Find coordinate mappings
    _, u_mapping = tree_u.query(V_u_coords, k=1)
    _, v_mapping = tree_v.query(V_v_coords, k=1)

    # Create full mapping from V to W velocity DOFs
    V_to_W_vel_mapping = np.zeros(len(U_field.vector().get_local()), dtype=int)

    # Map u-components
    for i, v_dof in enumerate(V_u_dofs):
        V_to_W_vel_mapping[v_dof] = W_u_dofs[u_mapping[i]]

    # Map v-components  
    for i, v_dof in enumerate(V_v_dofs):
        V_to_W_vel_mapping[v_dof] = W_v_dofs[v_mapping[i]]

    # Test the mapping
    UP_test_vec = UP_field.vector().get_local()
    U_extracted_mapped = UP_test_vec[V_to_W_vel_mapping]
    U_direct = U_field.vector().get_local()

    mapping_test = np.allclose(U_extracted_mapped, U_direct, atol=1e-12)
    print(f"Velocity extraction test passed: {mapping_test}")
    print(f"Max difference: {np.max(np.abs(U_extracted_mapped - U_direct))}")

    if mapping_test:
        print("✓ Velocity extraction from mixed space works correctly!")
        
        # Save all DOF mappings and indices
        np.save(save_dir / "V_to_W_vel_mapping.npy", V_to_W_vel_mapping)
        np.save(save_dir / "vel_dofs_in_mixed.npy", vel_dofs_in_mixed)
        np.save(save_dir / "pres_dofs_in_mixed.npy", pres_dofs_in_mixed)
        # np.save(save_dir / "W_u_dofs.npy", W_u_dofs)
        # np.save(save_dir / "W_v_dofs.npy", W_v_dofs)
        # np.save(save_dir / "V_u_dofs.npy", V_u_dofs)
        # np.save(save_dir / "V_v_dofs.npy", V_v_dofs)
        
        # Create utility functions file
        utility_code = '''# Utility functions for velocity extraction/insertion
    import numpy as np

    def extract_velocity_from_mixed(mixed_vector, V_to_W_mapping):
        """Extract velocity components from mixed vector using correct mapping"""
        return mixed_vector[V_to_W_mapping]

    def insert_velocity_into_mixed(mixed_vector, velocity_vector, V_to_W_mapping):
        """Insert velocity components into mixed vector using correct mapping"""
        mixed_copy = mixed_vector.copy()
        mixed_copy[V_to_W_mapping] = velocity_vector
        return mixed_copy

    # Example usage:
    # V_to_W_mapping = np.load("V_to_W_vel_mapping.npy")
    # mixed_data = np.load("UP_field_alldata.npy")
    # velocity_data = extract_velocity_from_mixed(mixed_data[:, 0, 0], V_to_W_mapping)
    '''
        
        with open(save_dir / "velocity_extraction_utils.py", "w") as f:
            f.write(utility_code)
        
        print("✓ DOF mappings and utility functions saved!")
        
    else:
        raise RuntimeError("Velocity extraction mapping failed!")

    print(f"Velocity DOFs in mixed space: {len(vel_dofs_in_mixed)}")
    print(f"Pressure DOFs in mixed space: {len(pres_dofs_in_mixed)}")

# def run_lidcavity_with_ic(Re, xloc, yloc, radius, amplitude, save_dir, num_steps=100):
#     import logging
#     import time
#     from pathlib import Path

#     import dolfin
#     import numpy as np

#     import flowcontrol.flowsolverparameters as flowsolverparameters
#     import utils.utils_flowsolver as flu
#     from examples.lidcavity.lidcavityflowsolver import LidCavityFlowSolver
#     from flowcontrol.actuator import ActuatorBCUniformU
#     from flowcontrol.sensor import SENSOR_TYPE, SensorPoint

#     t000 = time.time()
#     cwd = Path(__file__).parent

#     dolfin.set_log_level(dolfin.LogLevel.INFO)
#     logger = logging.getLogger(__name__)
#     FORMAT = "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]: %(message)s"
#     logging.basicConfig(format=FORMAT, level=logging.DEBUG)

#     params_flow = flowsolverparameters.ParamFlow(Re=Re, uinf=1)
#     params_flow.user_data["D"] = 1.0

#     params_time = flowsolverparameters.ParamTime(num_steps=num_steps, dt=0.005, Tstart=0.0)

#     params_save = flowsolverparameters.ParamSave(
#         save_every=20, path_out=save_dir
#     )

#     params_solver = flowsolverparameters.ParamSolver(
#         throw_error=True, is_eq_nonlinear=True, shift=0.0
#     )

#     params_mesh = flowsolverparameters.ParamMesh(
#         meshpath=cwd / "data_input" / "lidcavity_3.xdmf"
#     )
#     params_mesh.user_data["yup"] = 1
#     params_mesh.user_data["ylo"] = 0
#     params_mesh.user_data["xri"] = 1
#     params_mesh.user_data["xle"] = 0

#     params_restart = flowsolverparameters.ParamRestart()

#     actuator_bc_up = ActuatorBCUniformU()
#     sensor_1 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([0.05, 0.5]))
#     sensor_2 = SensorPoint(sensor_type=SENSOR_TYPE.U, position=np.array([0.5, 0.95]))
#     sensor_3 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([0.5, 0.95]))
#     params_control = flowsolverparameters.ParamControl(
#         sensor_list=[sensor_1, sensor_2, sensor_3],
#         actuator_list=[actuator_bc_up],
#     )

#     params_ic = flowsolverparameters.ParamIC(
#         xloc=xloc, yloc=yloc, radius=radius, amplitude=amplitude
#     )

#     fs = LidCavityFlowSolver(
#         params_flow=params_flow,
#         params_time=params_time,
#         params_save=params_save,
#         params_solver=params_solver,
#         params_mesh=params_mesh,
#         params_restart=params_restart,
#         params_control=params_control,
#         params_ic=params_ic,
#         verbose=10,
#     )

#     logger.info("__init__(): successful!")

#     logger.info("Exporting subdomains...")
#     flu.export_subdomains(
#         fs.mesh, fs.boundaries.subdomain, save_dir / "subdomains.xdmf"
#     )

#     logger.info("Load steady state...")
#     fs.load_steady_state(
#         path_u_p=[
#             cwd / "data_output" / "steady" / f"U0_Re={Re}.xdmf",
#             cwd / "data_output" / "steady" / f"P0_Re={Re}.xdmf",
#         ]
#     )

#     logger.info("Init time-stepping")
#     fs.initialize_time_stepping(ic=None)

#     logger.info("Step several times")
#     for _ in range(fs.params_time.num_steps):
#         y_meas = flu.MpiUtils.mpi_broadcast(fs.y_meas)
#         # This is still for unforced simulations
#         # TODO: Add forcing to the workflow
#         u_ctrl = [0.01 * np.sin(1 * np.pi * fs.t) + 0 * y_meas[0]]
#         # u_ctrl = [0 * y_meas[0]]
#         fs.step(u_ctrl=[u_ctrl[0]])

#     xdmf_files = [
#     save_dir / "@_restart0,0.xdmf",
#     # cwd / "data_output" / "@_restart0,0.xdmf",
#     ]

#     def make_file_name_for_field(field_name, original_file_name):
#         "Make file name for field U or P, using provided original"
#         "file name with placeholder @"
#         return str(original_file_name).replace("@", field_name)

#     # Data will be stored as 3D arrays
#     # of size [ndof*, nsnapshots, nfiles]
#     nsnapshots = fs.params_time.num_steps // fs.params_save.save_every
#     ndof_u = fs.V.dim()
#     ndof_p = fs.P.dim()
#     ndof = fs.W.dim()
#     nfiles = len(xdmf_files)

#     # Allocate arrays for 1 trajectory (= 1 file)
#     U_field_data = np.empty((ndof_u, nsnapshots))
#     P_field_data = np.empty((ndof_p, nsnapshots))
#     UP_field_data = np.empty((ndof, nsnapshots))

#     # Allocate arrays for all data (stack files on axis 2)
#     U_field_alldata = np.empty((ndof_u, nsnapshots, nfiles))
#     P_field_alldata = np.empty((ndof_p, nsnapshots, nfiles))
#     UP_field_alldata = np.empty((ndof, nsnapshots, nfiles))

#     # Allocate empty dolfin.Function
#     U_field = dolfin.Function(fs.V)
#     P_field = dolfin.Function(fs.P)
#     UP_field = dolfin.Function(fs.W)

#     for jfile in range(nfiles):
#         print(f"* Reading file nr={jfile}, name={xdmf_files[jfile]}")

#         file_name_U = make_file_name_for_field("U", xdmf_files[jfile])
#         file_name_P = make_file_name_for_field("P", xdmf_files[jfile])

#         for icounter in range(nsnapshots):
#             print(f"\t counter={icounter}")

#             try:
#                 flu.read_xdmf(file_name_U, U_field, "U", counter=icounter)
#                 flu.read_xdmf(file_name_P, P_field, "P", counter=icounter)
#                 UP_field = fs.merge(U_field, P_field)
#             except RuntimeError:
#                 print("\t *** EOF -- Reached End Of File")
#                 break

#             U_field_data[:, icounter] = np.copy(U_field.vector().get_local())
#             P_field_data[:, icounter] = np.copy(P_field.vector().get_local())
#             UP_field_data[:, icounter] = np.copy(UP_field.vector().get_local())

#         U_field_alldata[:, :, jfile] = np.copy(U_field_data)
#         P_field_alldata[:, :, jfile] = np.copy(P_field_data)
#         UP_field_alldata[:, :, jfile] = np.copy(UP_field_data)

#         print(f"\t -> Reached snapshot = {icounter} - Fetching next file")

#     print("Finished reading trajectores")

#     ##########################################################
#     # Steady-state
#     ##########################################################
#     U0 = dolfin.Function(fs.V)
#     P0 = dolfin.Function(fs.P)

#     # file_name_U0 = "src/examples/lidcavity/data_output/steady/U0.xdmf"
#     # file_name_P0 = "src/examples/lidcavity/data_output/steady/P0.xdmf"

#     file_name_U0 = cwd / "data_output" / "steady" / "U0.xdmf"
#     file_name_P0 = cwd / "data_output" / "steady" / "P0.xdmf"

#     print(f"* Reading steady-states at {file_name_U0}, {file_name_P0}")
#     flu.read_xdmf(file_name_U0, U0, "U0")
#     flu.read_xdmf(file_name_P0, P0, "P0")
#     UP0 = fs.merge(U0, P0)

#     U0_field_data = U0.vector().get_local()
#     P0_field_data = P0.vector().get_local()
#     UP0_field_data = UP0.vector().get_local()

#     print("Finished reading steady-states")

#     np.save(save_dir / "U_field_alldata.npy", U_field_alldata)
#     np.save(save_dir / "P_field_alldata.npy", P_field_alldata)
#     np.save(save_dir / "UP_field_alldata.npy", UP_field_alldata)

#     np.save(save_dir / "U0_field_data.npy", U0_field_data)
#     np.save(save_dir / "P0_field_data.npy", P0_field_data)
#     np.save(save_dir / "UP0_field_data.npy", UP0_field_data)

#     ##########################################################
#     # Extract mesh coordinates corresponding to DOF ordering
#     ##########################################################
#     print("Extracting mesh coordinates...")
    
#     # Get DOF coordinates for velocity space (V)
#     V_coords = fs.V.tabulate_dof_coordinates()
    
#     # Get DOF coordinates for pressure space (P)
#     P_coords = fs.P.tabulate_dof_coordinates()
    
#     # Get DOF coordinates for mixed space (W)
#     W_coords = fs.W.tabulate_dof_coordinates()
    
#     print(f"V DOF coordinates shape: {V_coords.shape}")
#     print(f"P DOF coordinates shape: {P_coords.shape}")
#     print(f"W DOF coordinates shape: {W_coords.shape}")

#     # Save mesh coordinates
#     np.save(save_dir / "V_dof_coordinates.npy", V_coords)
#     np.save(save_dir / "P_dof_coordinates.npy", P_coords)
#     np.save(save_dir / "W_dof_coordinates.npy", W_coords)

#     ##########################################################
#     # DOF Indices and Mixed Space Mapping
#     ##########################################################
#     print("Computing DOF indices and mixed space mappings...")

#     # Get velocity and pressure DOF indices in the mixed space
#     vel_dofs_in_mixed = np.array(fs.W.sub(0).dofmap().dofs())
#     pres_dofs_in_mixed = np.array(fs.W.sub(1).dofmap().dofs())

#     # Get component DOFs for both spaces
#     V_u_dofs = fs.V.sub(0).dofmap().dofs()
#     V_v_dofs = fs.V.sub(1).dofmap().dofs()
#     W_u_dofs = fs.W.sub(0).sub(0).dofmap().dofs()
#     W_v_dofs = fs.W.sub(0).sub(1).dofmap().dofs()

#     # Get coordinates for mapping
#     # V_coords = fs.V.tabulate_dof_coordinates()
#     # W_coords = fs.W.tabulate_dof_coordinates()

#     # Build coordinate-based mapping between V and W velocity spaces
#     V_u_coords = V_coords[V_u_dofs]
#     V_v_coords = V_coords[V_v_dofs]
#     W_u_coords = W_coords[W_u_dofs]
#     W_v_coords = W_coords[W_v_dofs]

#     # Create KDTrees for coordinate matching
#     tree_u = cKDTree(W_u_coords)
#     tree_v = cKDTree(W_v_coords)

#     # Find coordinate mappings
#     _, u_mapping = tree_u.query(V_u_coords, k=1)
#     _, v_mapping = tree_v.query(V_v_coords, k=1)

#     # Create full mapping from V to W velocity DOFs
#     V_to_W_vel_mapping = np.zeros(len(U_field.vector().get_local()), dtype=int)

#     # Map u-components
#     for i, v_dof in enumerate(V_u_dofs):
#         V_to_W_vel_mapping[v_dof] = W_u_dofs[u_mapping[i]]

#     # Map v-components  
#     for i, v_dof in enumerate(V_v_dofs):
#         V_to_W_vel_mapping[v_dof] = W_v_dofs[v_mapping[i]]

#     # Test the mapping
#     UP_test_vec = UP_field.vector().get_local()
#     U_extracted_mapped = UP_test_vec[V_to_W_vel_mapping]
#     U_direct = U_field.vector().get_local()

#     mapping_test = np.allclose(U_extracted_mapped, U_direct, atol=1e-12)
#     print(f"Velocity extraction test passed: {mapping_test}")
#     print(f"Max difference: {np.max(np.abs(U_extracted_mapped - U_direct))}")

#     if mapping_test:
#         print("✓ Velocity extraction from mixed space works correctly!")
        
#         # Save all DOF mappings and indices
#         np.save(save_dir / "V_to_W_vel_mapping.npy", V_to_W_vel_mapping)
#         np.save(save_dir / "vel_dofs_in_mixed.npy", vel_dofs_in_mixed)
#         np.save(save_dir / "pres_dofs_in_mixed.npy", pres_dofs_in_mixed)
#         # np.save(save_dir / "W_u_dofs.npy", W_u_dofs)
#         # np.save(save_dir / "W_v_dofs.npy", W_v_dofs)
#         # np.save(save_dir / "V_u_dofs.npy", V_u_dofs)
#         # np.save(save_dir / "V_v_dofs.npy", V_v_dofs)
        
#         # Create utility functions file
#         utility_code = '''# Utility functions for velocity extraction/insertion
#     import numpy as np

#     def extract_velocity_from_mixed(mixed_vector, V_to_W_mapping):
#         """Extract velocity components from mixed vector using correct mapping"""
#         return mixed_vector[V_to_W_mapping]

#     def insert_velocity_into_mixed(mixed_vector, velocity_vector, V_to_W_mapping):
#         """Insert velocity components into mixed vector using correct mapping"""
#         mixed_copy = mixed_vector.copy()
#         mixed_copy[V_to_W_mapping] = velocity_vector
#         return mixed_copy

#     # Example usage:
#     # V_to_W_mapping = np.load("V_to_W_vel_mapping.npy")
#     # mixed_data = np.load("UP_field_alldata.npy")
#     # velocity_data = extract_velocity_from_mixed(mixed_data[:, 0, 0], V_to_W_mapping)
#     '''
        
#         with open(save_dir / "velocity_extraction_utils.py", "w") as f:
#             f.write(utility_code)
        
#         print("✓ DOF mappings and utility functions saved!")
        
#     else:
#         raise RuntimeError("Velocity extraction mapping failed!")

#     print(f"Velocity DOFs in mixed space: {len(vel_dofs_in_mixed)}")
#     print(f"Pressure DOFs in mixed space: {len(pres_dofs_in_mixed)}")

#     # Plot to verify saved data - final trajectory snapshot
#     plot_to_check = False
#     if plot_to_check:
#         import matplotlib.pyplot as plt
#         U_final = dolfin.Function(fs.V)
#         U_final.vector().set_local(U_field_alldata[:, -1, 0])  # Last snapshot from saved data
        
#         V_scalar = dolfin.FunctionSpace(fs.mesh, "CG", 1)
#         velocity_mag = dolfin.project(dolfin.sqrt(dolfin.dot(U_final, U_final)), V_scalar)
        
#         plt.figure(figsize=(8, 6))
#         c = dolfin.plot(velocity_mag)
#         plt.colorbar(c)
#         plt.title(f'Velocity Magnitude (from saved data) - xloc={xloc:.3f}, yloc={yloc:.3f}')
#         plt.savefig(save_dir / "velocity_magnitude_from_saved_data.png", dpi=150, bbox_inches='tight')
#         plt.show()

if __name__ == "__main__":
    # Adapt to wherever you want to save the results
    base_dir = Path("/Users/jaking/Desktop/PhD/lid_driven_cavity")
    parent_dir = base_dir / f"Re{Re}_forced"
    parent_dir.mkdir(parents=True, exist_ok=True)

    phase_angles = np.linspace(0,2*np.pi, 1)
    amplitudes = [0.005]  # Different perturbation amplitudes
    num_steps = 10000
    count = 1
    for phase_angle in phase_angles:
        for amp in amplitudes:
            save_dir = parent_dir / f"run{count}"
            save_dir.mkdir(parents=True, exist_ok=True)

            print(f"Running simulation {count} with angle {phase_angle}, amplitude {amp}")
            print(f"  -> Save directory: {save_dir}")
            
            run_lidcavity_with_eigenvector_ic(Re, phase_angle, amp, 1, 0.01, save_dir, num_steps)
            
            print(f"Finished simulation {count}")
            count += 1