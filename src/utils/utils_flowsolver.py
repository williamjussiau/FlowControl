"""
Utilitary functions for FlowSolver
"""

# FlowSolver direct utility #########################################
# Write CPP utils (for boundaries) ##################################
# Dolfin utility ####################################################
from utils.fem import (  # noqa: F401, E402
    and_cpp,
    apply_fun,
    between_cpp,
    expression_to_dolfin_function,
    get_subspace_dofs,
    near_cpp,
    on_boundary_cpp,
    or_cpp,
    print0,
    projectm,
    show_max,
    summarize_timings,
)

# Export utility ########################################################
from utils.io import (  # noqa: F401, E402
    export_boundary_field,
    export_boundary_forces,
    export_complex_field,
    export_dof_map,
    export_npz_to_mat,
    export_sparse_matrix,
    export_square_operators,
    export_stress_tensor,
    export_subdomains,
    plot_Hw,
    read_xdmf,
    save_Hw,
    write_xdmf,
)

# Frequency response utility ############################################
# Eig utility ###########################################################
# PETSc, scipy.sparse utility ###########################################
from utils.linalg import (  # noqa: F401, E402
    dolfin_to_petsc,
    dolfin_to_scipy,
    get_all_enum_petsc_slepc,
    get_field_response,
    get_frequency_response_mpi,
    get_frequency_response_parallel,
    get_frequency_response_sequential,
    get_Hw_lifting,
    get_mat_vp_slepc,
    load_mat_from_file,
    make_mat_to_test_slepc,
    numpy_to_petsc,
    petsc_to_scipy,
    scipy_to_petsc,
)

# Controller utility ####################################################
from utils.lticontrol import (  # noqa: F401, E402
    balreal,
    balred_rel,
    baltransform,
    basis_laguerre,
    basis_laguerre_canonical,
    basis_laguerre_canonical_ss,
    basis_laguerre_K00,
    basis_laguerre_ss,
    build_block_Psi,
    compare_controllers,
    condswitch,
    controller_residues,
    controller_residues_getidx,
    controller_residues_wrapper,
    export_controller,
    hinfsyn_mref,
    isstable,
    isstablecl,
    lncf,
    lqg_regulator,
    make_tf_real,
    norm,
    read_matfile,
    read_regulator,
    read_ss,
    reduceorder,
    rncf,
    show_ss,
    sigma_trivial,
    slowfast,
    ss_blkdiag_list,
    ss_hstack,
    ss_hstack_list,
    ss_inv,
    ss_one,
    ss_transpose,
    ss_vstack,
    ss_vstack_list,
    ss_zero,
    ssdata,
    sys_hsv,
    test_controller_residues,
    write_ss,
    youla,
    youla_laguerre,
    youla_laguerre_K00,
    youla_laguerre_mimo,
    youla_left_coprime,
    youla_lqg,
    youla_lqg_lftmat,
    youla_Q0b,
    youla_Qab,
    youla_right_coprime,
)

# MPI utility ###########################################################
from utils.mpi import MpiUtils  # noqa: F401, E402

# Cross optimization-Flowsolver utility #################################
from utils.optim import compute_cost, write_optim_csv  # noqa: F401, E402

# Signal processing and array utility ###################################
from utils.signal import (  # noqa: F401, E402
    compute_signal_frequency,
    pad_upto,
    sample_lco,
    saturate,
)
