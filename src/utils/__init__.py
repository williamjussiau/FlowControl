"""Flow control utility package.

Submodules
----------
fem         -- FEniCS/dolfin helpers (projection, subspace DOFs, C++ snippets)
io          -- Field export/import, frequency-response I/O, DOF map export
linalg      -- PETSc/SciPy sparse utilities, frequency response, eigenvalues
lticontrol  -- LTI control: SS algebra, Youla, LQG, H-inf, Laguerre, coprime
mesh        -- Mesh loading and parameter helpers
mpi         -- MPI rank/comm utilities
optim       -- Cost function and CSV logging for optimisation loops
physics     -- Vorticity, divergence, stress tensor, divergence-free ICs
signal      -- Multisine generation, crest-factor optimisation, JSON export
"""

__all__ = [
    "fem",
    "io",
    "linalg",
    "lticontrol",
    "mesh",
    "mpi",
    "mesh_generation",
    "optim",
    "physics",
    "signal",
]
