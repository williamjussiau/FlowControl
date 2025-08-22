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
from scipy.spatial import cKDTree

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

def save_data(fs, save_dir, cwd, logger):
    """Save simulation data, coordinates, and DOF mappings"""
    
    xdmf_files = [
        save_dir / "@_restart0,0.xdmf",
    ]

    logger.info("Cleaning up redundant prev files")
    cleanup_redundant_files(save_dir)
    logger.info("Finished cleaning up")

    def make_file_name_for_field(field_name, original_file_name):
        """Make file name for field U or P, using provided original file name with placeholder @"""
        return str(original_file_name).replace("@", field_name)

    # Data will be stored as 3D arrays of size [ndof*, nsnapshots, nfiles]
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

    print("Finished reading trajectories")

    ##########################################################
    # Steady-state
    ##########################################################
    U0 = dolfin.Function(fs.V)
    P0 = dolfin.Function(fs.P)

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

    # Save field data
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
    
    V_coords = fs.V.tabulate_dof_coordinates()
    P_coords = fs.P.tabulate_dof_coordinates()
    W_coords = fs.W.tabulate_dof_coordinates()
    
    print(f"V DOF coordinates shape: {V_coords.shape}")
    print(f"P DOF coordinates shape: {P_coords.shape}")
    print(f"W DOF coordinates shape: {W_coords.shape}")

    # Save mesh coordinates
    np.save(save_dir / "V_dof_coordinates.npy", V_coords)
    np.save(save_dir / "P_dof_coordinates.npy", P_coords)
    np.save(save_dir / "W_dof_coordinates.npy", W_coords)

    ##########################################################
    # Extract Actuator Boundary DOFs
    ##########################################################
    print("Extracting actuator boundary DOFs...")
    
    # Get actuator boundary conditions in mixed space (W)
    bcu_actuation_up = dolfin.DirichletBC(
        fs.W.sub(0), dolfin.Constant((0, 0)), fs.get_subdomain("actuator_up"))
    bcu_actuation_lo = dolfin.DirichletBC(
        fs.W.sub(0), dolfin.Constant((0, 0)), fs.get_subdomain("actuator_lo"))
    
    actuator_up_dofs_W = list(bcu_actuation_up.get_boundary_values().keys())
    actuator_lo_dofs_W = list(bcu_actuation_lo.get_boundary_values().keys())
    actuator_dofs_W = np.array(actuator_up_dofs_W + actuator_lo_dofs_W)
    
    # Get actuator boundary conditions in velocity space (V)
    bcu_actuation_up_V = dolfin.DirichletBC(
        fs.V, dolfin.Constant((0, 0)), fs.get_subdomain("actuator_up"))
    bcu_actuation_lo_V = dolfin.DirichletBC(
        fs.V, dolfin.Constant((0, 0)), fs.get_subdomain("actuator_lo"))

    actuator_up_dofs_V = list(bcu_actuation_up_V.get_boundary_values().keys())
    actuator_lo_dofs_V = list(bcu_actuation_lo_V.get_boundary_values().keys())
    actuator_dofs_V = np.array(actuator_up_dofs_V + actuator_lo_dofs_V)

    print(f"Actuator up DOFs W: {len(actuator_up_dofs_W)}")
    print(f"Actuator lo DOFs W: {len(actuator_lo_dofs_W)}")
    print(f"Total actuator DOFs in W space: {len(actuator_dofs_W)}")
    print(f"Actuator up DOFs V: {len(actuator_up_dofs_V)}")
    print(f"Actuator lo DOFs V: {len(actuator_lo_dofs_V)}")
    print(f"Total actuator DOFs in V space: {len(actuator_dofs_V)}")

    # Save actuator DOF information
    np.save(save_dir / "actuator_up_dofs_W.npy", np.array(actuator_up_dofs_W))
    np.save(save_dir / "actuator_lo_dofs_W.npy", np.array(actuator_lo_dofs_W))
    np.save(save_dir / "actuator_dofs_W.npy", actuator_dofs_W)
    np.save(save_dir / "actuator_up_dofs_V.npy", np.array(actuator_up_dofs_V))
    np.save(save_dir / "actuator_lo_dofs_V.npy", np.array(actuator_lo_dofs_V))
    np.save(save_dir / "actuator_dofs_V.npy", actuator_dofs_V)

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
        save_every=10, path_out=save_dir
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

    save_data(fs, save_dir, cwd, logger)
    return

    # xdmf_files = [
    # save_dir / "@_restart0,0.xdmf",
    # # cwd / "data_output" / "@_restart0,0.xdmf",
    # ]

    # logger.info("Cleanung up redundant prev files")
    # cleanup_redundant_files(save_dir)
    # logger.info("Finished cleaning up")

    # def make_file_name_for_field(field_name, original_file_name):
    #     "Make file name for field U or P, using provided original"
    #     "file name with placeholder @"
    #     return str(original_file_name).replace("@", field_name)

    # # Data will be stored as 3D arrays
    # # of size [ndof*, nsnapshots, nfiles]
    # nsnapshots = fs.params_time.num_steps // fs.params_save.save_every
    # ndof_u = fs.V.dim()
    # ndof_p = fs.P.dim()
    # ndof = fs.W.dim()
    # nfiles = len(xdmf_files)

    # # Allocate arrays for 1 trajectory (= 1 file)
    # U_field_data = np.empty((ndof_u, nsnapshots))
    # P_field_data = np.empty((ndof_p, nsnapshots))
    # UP_field_data = np.empty((ndof, nsnapshots))

    # # Allocate arrays for all data (stack files on axis 2)
    # U_field_alldata = np.empty((ndof_u, nsnapshots, nfiles))
    # P_field_alldata = np.empty((ndof_p, nsnapshots, nfiles))
    # UP_field_alldata = np.empty((ndof, nsnapshots, nfiles))

    # # Allocate empty dolfin.Function
    # U_field = dolfin.Function(fs.V)
    # P_field = dolfin.Function(fs.P)
    # UP_field = dolfin.Function(fs.W)

    # for jfile in range(nfiles):
    #     print(f"* Reading file nr={jfile}, name={xdmf_files[jfile]}")

    #     file_name_U = make_file_name_for_field("U", xdmf_files[jfile])
    #     file_name_P = make_file_name_for_field("P", xdmf_files[jfile])

    #     for icounter in range(nsnapshots):
    #         print(f"\t counter={icounter}")

    #         try:
    #             flu.read_xdmf(file_name_U, U_field, "U", counter=icounter)
    #             flu.read_xdmf(file_name_P, P_field, "P", counter=icounter)
    #             UP_field = fs.merge(U_field, P_field)
    #         except RuntimeError:
    #             print("\t *** EOF -- Reached End Of File")
    #             break

    #         U_field_data[:, icounter] = np.copy(U_field.vector().get_local())
    #         P_field_data[:, icounter] = np.copy(P_field.vector().get_local())
    #         UP_field_data[:, icounter] = np.copy(UP_field.vector().get_local())

    #     U_field_alldata[:, :, jfile] = np.copy(U_field_data)
    #     P_field_alldata[:, :, jfile] = np.copy(P_field_data)
    #     UP_field_alldata[:, :, jfile] = np.copy(UP_field_data)

    #     print(f"\t -> Reached snapshot = {icounter} - Fetching next file")

    # print("Finished reading trajectores")

    # ##########################################################
    # # Steady-state
    # ##########################################################
    # U0 = dolfin.Function(fs.V)
    # P0 = dolfin.Function(fs.P)

    # file_name_U0 = cwd / "data_output" / "steady" / "U0.xdmf"
    # file_name_P0 = cwd / "data_output" / "steady" / "P0.xdmf"

    # print(f"* Reading steady-states at {file_name_U0}, {file_name_P0}")
    # flu.read_xdmf(file_name_U0, U0, "U0")
    # flu.read_xdmf(file_name_P0, P0, "P0")
    # UP0 = fs.merge(U0, P0)

    # U0_field_data = U0.vector().get_local()
    # P0_field_data = P0.vector().get_local()
    # UP0_field_data = UP0.vector().get_local()

    # print("Finished reading steady-states")

    # np.save(save_dir / "U_field_alldata.npy", U_field_alldata)
    # np.save(save_dir / "P_field_alldata.npy", P_field_alldata)
    # np.save(save_dir / "UP_field_alldata.npy", UP_field_alldata)

    # np.save(save_dir / "U0_field_data.npy", U0_field_data)
    # np.save(save_dir / "P0_field_data.npy", P0_field_data)
    # np.save(save_dir / "UP0_field_data.npy", UP0_field_data)

    # ##########################################################
    # # Extract mesh coordinates corresponding to DOF ordering
    # ##########################################################
    # print("Extracting mesh coordinates...")
    
    # # Get DOF coordinates for velocity space (V)
    # V_coords = fs.V.tabulate_dof_coordinates()
    
    # # Get DOF coordinates for pressure space (P)
    # P_coords = fs.P.tabulate_dof_coordinates()
    
    # # Get DOF coordinates for mixed space (W)
    # W_coords = fs.W.tabulate_dof_coordinates()
    
    # print(f"V DOF coordinates shape: {V_coords.shape}")
    # print(f"P DOF coordinates shape: {P_coords.shape}")
    # print(f"W DOF coordinates shape: {W_coords.shape}")

    # # Save mesh coordinates
    # np.save(save_dir / "V_dof_coordinates.npy", V_coords)
    # np.save(save_dir / "P_dof_coordinates.npy", P_coords)
    # np.save(save_dir / "W_dof_coordinates.npy", W_coords)

    # ##########################################################
    # # Extract Actuator Boundary DOFs
    # ##########################################################
    # print("Extracting actuator boundary DOFs...")
    
    # # Get actuator boundary conditions directly from the flow solver
    # bcu_actuation_up = dolfin.DirichletBC(
    #     fs.W.sub(0),
    #     dolfin.Constant((0, 0)),
    #     fs.get_subdomain("actuator_up"),
    # )
    # bcu_actuation_lo = dolfin.DirichletBC(
    #     fs.W.sub(0),
    #     dolfin.Constant((0, 0)),
    #     fs.get_subdomain("actuator_lo"),
    # )
    
    # # Extract DOF indices from boundary conditions
    # actuator_up_dofs = list(bcu_actuation_up.get_boundary_values().keys())
    # actuator_lo_dofs = list(bcu_actuation_lo.get_boundary_values().keys())
    # actuator_dofs_W = np.array(actuator_up_dofs + actuator_lo_dofs)
    
    # print(f"Actuator up DOFs: {len(actuator_up_dofs)}")
    # print(f"Actuator lo DOFs: {len(actuator_lo_dofs)}")
    # print(f"Total actuator DOFs in W space: {len(actuator_dofs_W)}")
    
    # # Save actuator DOF information
    # np.save(save_dir / "actuator_up_dofs_W.npy", np.array(actuator_up_dofs))
    # np.save(save_dir / "actuator_lo_dofs_W.npy", np.array(actuator_lo_dofs))
    # np.save(save_dir / "actuator_dofs_W.npy", actuator_dofs_W)

    # # Get actuator boundary conditions in velocity space (V)
    # bcu_actuation_up_V = dolfin.DirichletBC(
    #     fs.V,
    #     dolfin.Constant((0, 0)),
    #     fs.get_subdomain("actuator_up"),
    # )
    # bcu_actuation_lo_V = dolfin.DirichletBC(
    #     fs.V,
    #     dolfin.Constant((0, 0)),
    #     fs.get_subdomain("actuator_lo"),
    # )

    # # Extract DOF indices from boundary conditions in V space
    # actuator_up_dofs_V = list(bcu_actuation_up_V.get_boundary_values().keys())
    # actuator_lo_dofs_V = list(bcu_actuation_lo_V.get_boundary_values().keys())
    # actuator_dofs_V = np.array(actuator_up_dofs_V + actuator_lo_dofs_V)

    # print(f"Actuator up DOFs V: {len(actuator_up_dofs_V)}")
    # print(f"Actuator lo DOFs V: {len(actuator_lo_dofs_V)}")
    # print(f"Total actuator DOFs in V space: {len(actuator_dofs_V)}")

    # # Save actuator DOF information
    # np.save(save_dir / "actuator_up_dofs_V.npy", np.array(actuator_up_dofs_V))
    # np.save(save_dir / "actuator_lo_dofs_V.npy", np.array(actuator_lo_dofs_V))
    # np.save(save_dir / "actuator_dofs_V.npy", actuator_dofs_V)

    # # For velocity space DOFs only
    # # bcu_actuation_up_V = dolfin.DirichletBC(
    # #     fs.V,
    # #     dolfin.Constant((0, 0)),
    # #     fs.get_subdomain("actuator_up"),
    # # )
    # # actuator_up_dofs_V = list(bcu_actuation_up_V.get_boundary_values().keys())

    # ##########################################################
    # # DOF Indices and Mixed Space Mapping
    # ##########################################################
    # print("Computing DOF indices and mixed space mappings...")

    # # Get velocity and pressure DOF indices in the mixed space
    # vel_dofs_in_mixed = np.array(fs.W.sub(0).dofmap().dofs())
    # pres_dofs_in_mixed = np.array(fs.W.sub(1).dofmap().dofs())

    # # Get component DOFs for both spaces
    # V_u_dofs = fs.V.sub(0).dofmap().dofs()
    # V_v_dofs = fs.V.sub(1).dofmap().dofs()
    # W_u_dofs = fs.W.sub(0).sub(0).dofmap().dofs()
    # W_v_dofs = fs.W.sub(0).sub(1).dofmap().dofs()

    # # Build coordinate-based mapping between V and W velocity spaces
    # V_u_coords = V_coords[V_u_dofs]
    # V_v_coords = V_coords[V_v_dofs]
    # W_u_coords = W_coords[W_u_dofs]
    # W_v_coords = W_coords[W_v_dofs]

    # # Create KDTrees for coordinate matching
    # tree_u = cKDTree(W_u_coords)
    # tree_v = cKDTree(W_v_coords)

    # # Find coordinate mappings
    # _, u_mapping = tree_u.query(V_u_coords, k=1)
    # _, v_mapping = tree_v.query(V_v_coords, k=1)

    # # Create full mapping from V to W velocity DOFs
    # V_to_W_vel_mapping = np.zeros(len(U_field.vector().get_local()), dtype=int)

    # # Map u-components
    # for i, v_dof in enumerate(V_u_dofs):
    #     V_to_W_vel_mapping[v_dof] = W_u_dofs[u_mapping[i]]

    # # Map v-components  
    # for i, v_dof in enumerate(V_v_dofs):
    #     V_to_W_vel_mapping[v_dof] = W_v_dofs[v_mapping[i]]

    # # Test the mapping
    # UP_test_vec = UP_field.vector().get_local()
    # U_extracted_mapped = UP_test_vec[V_to_W_vel_mapping]
    # U_direct = U_field.vector().get_local()

    # mapping_test = np.allclose(U_extracted_mapped, U_direct, atol=1e-12)
    # print(f"Velocity extraction test passed: {mapping_test}")
    # print(f"Max difference: {np.max(np.abs(U_extracted_mapped - U_direct))}")

    # if mapping_test:
    #     print("✓ Velocity extraction from mixed space works correctly!")
        
    #     # Save all DOF mappings and indices
    #     np.save(save_dir / "V_to_W_vel_mapping.npy", V_to_W_vel_mapping)
    #     np.save(save_dir / "vel_dofs_in_mixed.npy", vel_dofs_in_mixed)
    #     np.save(save_dir / "pres_dofs_in_mixed.npy", pres_dofs_in_mixed)
    #     # np.save(save_dir / "W_u_dofs.npy", W_u_dofs)
    #     # np.save(save_dir / "W_v_dofs.npy", W_v_dofs)
    #     # np.save(save_dir / "V_u_dofs.npy", V_u_dofs)
    #     # np.save(save_dir / "V_v_dofs.npy", V_v_dofs)
        
    #     # Create utility functions file
    #     utility_code = '''# Utility functions for velocity extraction/insertion
    # import numpy as np

    # def extract_velocity_from_mixed(mixed_vector, V_to_W_mapping):
    #     """Extract velocity components from mixed vector using correct mapping"""
    #     return mixed_vector[V_to_W_mapping]

    # def insert_velocity_into_mixed(mixed_vector, velocity_vector, V_to_W_mapping):
    #     """Insert velocity components into mixed vector using correct mapping"""
    #     mixed_copy = mixed_vector.copy()
    #     mixed_copy[V_to_W_mapping] = velocity_vector
    #     return mixed_copy

    # # Example usage:
    # # V_to_W_mapping = np.load("V_to_W_vel_mapping.npy")
    # # mixed_data = np.load("UP_field_alldata.npy")
    # # velocity_data = extract_velocity_from_mixed(mixed_data[:, 0, 0], V_to_W_mapping)
    # '''
        
    #     with open(save_dir / "velocity_extraction_utils.py", "w") as f:
    #         f.write(utility_code)
        
    #     print("✓ DOF mappings and utility functions saved!")
        
    # else:
    #     raise RuntimeError("Velocity extraction mapping failed!")

    # print(f"Velocity DOFs in mixed space: {len(vel_dofs_in_mixed)}")
    # print(f"Pressure DOFs in mixed space: {len(pres_dofs_in_mixed)}")

    return


def main():

    base_dir = Path("/Users/jaking/Desktop/PhD/cylinder")
    parent_dir = base_dir / f"Re{Re}_short"
    parent_dir.mkdir(parents=True, exist_ok=True)

    # x_vals = np.linspace(0.2, 0.8, 3)
    # y_vals = np.linspace(0.2, 0.8, 3)
    # x_vals = np.linspace(0.2, 0.2, 1)
    # y_vals = np.linspace(0.2, 0.2, 1)
    x_vals = [2.0]
    y_vals = [0.0]
    radius = 0.5
    amplitude = 1.0
    num_steps = 200
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
