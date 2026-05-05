# Installation

## Prerequisites

- Python >= 3.10
- Conda (Anaconda or Miniconda)

## Setup

```bash
# Clone the repository
git clone https://github.com/williamjussiau/FlowControl.git
cd FlowControl

# Create the conda environment
conda env create -n fenics --file environment.yml

# Activate and install in development mode
conda activate fenics
conda develop src
```

## Optional Dependencies

- `optim`: Optimization algorithms (SMT, blackbox_opt)
- `mesh`: Mesh generation utilities (gmsh, meshio)
- Install with: `pip install flowcontrol[optim,mesh]` or `pip install flowcontrol[all]`
