# Utility Tools

## Operator Extraction

The linearized system matrices $(A, B, C, E)$ around a base flow can be extracted using `OperatorGetter` from `flowcontrol.operatorgetter`. A converged steady state must be available on the `FlowSolver` before calling any of these methods.

```python
from flowcontrol.operatorgetter import OperatorGetter

opget = OperatorGetter(fs)
A = opget.get_A(autodiff=True)   # linearized NS operator    — dolfin.PETScMatrix (n × n)
E = opget.get_mass_matrix()      # velocity mass matrix      — dolfin.PETScMatrix (n × n)
B = opget.get_B()                # input (control→state)     — np.ndarray (n, n_act)
C = opget.get_C()                # output (state→sensor)     — np.ndarray (n_sens, n)
```

`get_A` accepts `autodiff=True` (UFL automatic differentiation, default) or `autodiff=False` (manual linearization); both paths should agree to machine precision.

The matrices satisfy the linearized system: $E \dot{q} = A q + B u$, $y = C q$, where $q$ is the state (velocity perturbation DOFs), $u$ the control input, and $y$ the sensor output.


## Frequency Response

Given the operators above, the frequency response $H(j\omega) = C\,(j\omega E - A)^{-1} B$ can be computed with `utils.linalg`:

```python
import numpy as np
from utils.linalg import get_frequency_response_sequential, get_frequency_response_parallel

ww = np.logspace(-2, 2, 100)  # angular frequencies [rad/s]

# Single-process (sequential solves, suitable for moderate system sizes)
H, ww = get_frequency_response_sequential(A, B, C, Q=E, ww=ww)

# Multi-process (joblib parallelism over frequencies)
H, ww = get_frequency_response_parallel(A, B, C, Q=E, ww=ww, n_jobs=4)
```

`A` and `Q` (`=E`) accept either `dolfin.PETScMatrix` or `scipy.sparse` matrices. `B` and `C` are dense `np.ndarray`.

`H` has shape `(ny, nu, nw)` complex, where `ny` is the number of sensors, `nu` the number of actuators, and `nw = len(ww)`. For a SISO system `H[0, 0, :]` is the scalar transfer function sampled at each frequency.

### Implementation: real-arithmetic splitting

PETSc supports complex arithmetic, but the FEniCS 2019.1.0 conda package is built against the real PETSc configuration. To work around this, the solve $(j\omega E - A)^{-1} B$ is cast into a real system. Writing the solution as $x_r + j x_i$ and separating real and imaginary parts gives:

$$\begin{bmatrix} -A & -\omega E \\ \omega E & -A \end{bmatrix} \begin{bmatrix} x_r \\ x_i \end{bmatrix} = \begin{bmatrix} B \\ 0 \end{bmatrix}$$

This $2n \times 2n$ real system is solved once per frequency with a sparse LU factorization, then $H(\omega) = C x_r + j\, C x_i$.

A third variant `get_frequency_response_mpi` uses PETSc/MUMPS for parallel solves and must be launched with `mpirun`.


## Export and Debug Utilities

- `export_subdomains(fs.mesh, fs.boundaries.subdomain, path)` — export boundary subdomain markers to XDMF for inspection in Paraview. Import directly from `utils.io`, or via the convenience aggregator `import utils.utils_flowsolver as flu`.
- `utils.io` — save/load sparse matrices and frequency response data to disk.
- `utils.mpi.MpiUtils` — MPI broadcast helpers (e.g. `MpiUtils.mpi_broadcast` to gather sensor values on all ranks).


## Optimization Backend

The toolbox can be used as a black-box simulation backend inside an optimization loop. See [Jussiau et al. (2025)](https://ieeexplore.ieee.org/abstract/document/10884641/) for an example of closed-loop controller optimization using this setup.
