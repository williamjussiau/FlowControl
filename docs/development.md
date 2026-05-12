# Development

## Code Structure

```
src/flowcontrol/          # Core classes
├── __init__.py
├── actuator.py         # Actuator classes (BC, force)
├── controller.py       # LTI controller with ZOH stepping
├── exporter.py         # Field and timeseries I/O
├── flowfield.py        # FlowField, FlowFieldCollection, BoundaryConditions, SimPaths
├── flowsolver.py       # Main solver interface (abstract base)
├── flowsolverparameters.py  # Param* dataclasses
├── nsforms.py          # Navier-Stokes weak forms
├── operatorgetter.py   # State-space operator extraction (A, B, C, E)
└── steadystate.py      # Newton/Picard steady-state solver

src/utils/             # Utilities
├── __init__.py
├── fem.py              # FEniCS helpers
├── io.py               # XDMF/HDF5 I/O and sparse matrix utilities
├── linalg.py           # Linear algebra: freq response, eigensolvers
├── mpi.py              # MPI utilities (mpi_broadcast, peval, ...)
├── physics.py          # Flow post-processing (energy, wall shear)
├── signal.py           # Signal processing utilities
└── mesh_generation/    # Mesh generators

tests/                 # Test suite
├── conftest.py         # Shared fixtures
├── integration/        # Integration tests
│   └── conftest.py     # Coarse mesh fixtures
└── test_*.py           # Unit tests

.github/workflows/    # CI/CD
├── test-fast.yml     # Fast tests (push/PR)
├── test-full.yml     # All tests (manual)
└── test-mpi.yml      # MPI tests (push/PR)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Open a Pull Request

## CI Workflows

- **test-fast.yml**: Runs on every push/PR — unit tests + fast integration tests
- **test-full.yml**: Manual trigger — all non-MPI tests
- **test-mpi.yml**: Runs on push/PR — MPI-specific tests
