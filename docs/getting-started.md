# Getting Started 🛠️

## Installation

### Conda

The conda environment required to run the code is defined in `environment.yml`.

Create and activate the environment:

```bash
conda env create -n fenics --file environment.yml
conda activate fenics
conda develop src
```

**Requirements:**
- Python >= 3.10 (tested with 3.10, 3.11, 3.12)
- FEniCS 2019.1.0 (only available via conda)

The folder `src` in the root must be appended to the Python system path. The `conda develop src` command above handles this automatically.

> **Note:** Additional path tweaking is sometimes required for FEniCS to be found through the `dolfin` module (see [this problem with PKG_CONFIG](https://fenicsproject.discourse.group/t/problem-with-fenics-and-macos-catalina/2106)).

### Optional Dependencies

- **optim_algs**: Optimization algorithms (SMT, blackbox_opt)
- **mesh**: Mesh generation utilities (gmsh, meshio)

Install with: `pip install flowcontrol[optim_algs,mesh]` or `pip install flowcontrol[all]`

## Run Examples

After setting up the conda environment, test the installation:

1. Try importing FEniCS: `import dolfin`
2. Run examples from the `examples` folder:
   - `cylinder/run_cylinder_example.py` and `cavity/run_cavity_example.py` for time simulations
   - `mpitest/demo_poisson.py` with MPI (e.g., `mpirun -np 2 python demo_poisson.py`) to check parallel tools

> ⚠️ If you observe that `mpirun` is excessively slow, see [this thread about MPI and OpenMP](https://fenicsproject.discourse.group/t/mumps-in-parallel-slower-than-in-serial/662).

## Docker :whale:

[Coming soon]
