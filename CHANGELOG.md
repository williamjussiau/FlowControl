# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `NSForms`, `SteadyStateSolver`, `FlowExporter` extracted as dedicated classes
- `utils/` partitioned into focused modules: `fem`, `io`, `linalg`, `lticontrol`, `mesh`, `mpi`, `optim`, `physics`, `signal`
- gmsh-based mesh generators for all four benchmark flows (`mesh_generation/`)
- Comprehensive unit and integration test suite with `slow` and `mpi` markers
- GitHub Actions CI: fast tests, full tests, MPI tests
- `pyproject.toml` with optional dependency groups (`optim_algs`, `mesh`)
- `py.typed` marker for PEP 561 compliance

### Changed
- `FlowSolver` is now an abstract base class; benchmark solvers are subclasses
- Parameter objects refactored to dataclasses (`ParamFlow`, `ParamTime`, etc.)

## [0.1.0] - 2024-01-01

Initial public release.
