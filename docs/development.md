# Development

## Code Structure

```
src/flowcontrol/    # Core classes
├── flowsolver.py     # Main solver interface
├── nsforms.py        # Navier-Stokes weak forms
├── controller.py     # Control system classes
└── ...

src/utils/           # Utilities
├── fem.py            # FEniCS helpers
├── physics.py        # Flow post-processing
├── mesh_generation/  # Mesh generators
└── ...

tests/               # Test suite
├── conftest.py       # Shared fixtures
├── integration/      # Integration tests
│   └── conftest.py   # Coarse mesh fixtures
└── test_*.py         # Unit tests

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
