# Usage

## Running Tests

```bash
# All tests
pytest tests/

# Fast tests only (skip slow and MPI)
pytest tests/ -m "not slow and not mpi"

# Single test file
pytest tests/test_controller.py
```

## Running Simulations

```bash
# Serial simulation
python src/examples/cylinder/run_cylinder_example.py

# Parallel simulation (MPI)
mpirun -n 4 python src/examples/cylinder/run_cylinder_example.py
```

## Configuration

Flow configurations are defined in YAML files. See examples in `src/examples/*/` for reference.
