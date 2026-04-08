import time

import numpy as np
import petsc4py
from mpi4py import MPI
from petsc4py import PETSc

# Initialize PETSc and MPI
petsc4py.init()
comm = PETSc.COMM_WORLD
rank = comm.Get_rank()


# --- Helper function to create PETSc matrix from NumPy ---
def numpy_to_petsc_mat(arr):
    mat = PETSc.Mat().createAIJ(size=arr.shape, comm=comm)
    mat.setValues(range(arr.shape[0]), range(arr.shape[1]), arr.flatten())
    mat.assemble()
    return mat


# --- Test matrices ---
# A_np = np.array([[-1.758, 0.8539], [-0.9695, 0.3192]])
# B_np = np.array([[0.3005, 0.8155], [-0.3731, 0.0]])
# C_np = np.array([[0.1202, 0.4128], [0.5712, 0.0]])
# Q_np = np.eye(2)
n = 100
nu = 2
ny = 3
A_np = np.random.randn(n, n)
B_np = np.random.randn(n, nu)
C_np = np.random.randn(ny, n)
Q_np = np.eye(n, n)


# Convert to PETSc matrices
A = numpy_to_petsc_mat(A_np)
B = numpy_to_petsc_mat(B_np)
C = numpy_to_petsc_mat(C_np)
Q = numpy_to_petsc_mat(Q_np)

n_x = A.size[0]
n_y = C.size[0]
n_u = B.size[1]
n_w = 100
omega = np.logspace(-2, 2, num=n_w)


# --- Benchmarking functions ---
def loop_over_nw_real_imag():
    Hw = np.zeros((n_y, n_u, n_w), dtype=complex)
    for u in range(n_u):
        for i, w in enumerate(omega):
            # Construct the block matrix
            M = PETSc.Mat().create(comm=comm)
            M.setType("aij")
            M.setSizes([2 * n_x, 2 * n_x])
            M.setPreallocationNNZ((3 * n_x, 3 * n_x))
            M.setUp()

            # Fill the block matrix
            # Top-left block: -A
            for row in range(n_x):
                for col in range(n_x):
                    M.setValue(row, col, -A.getValue(row, col))

            # Top-right block: -w*Q
            for row in range(n_x):
                for col in range(n_x):
                    M.setValue(row, n_x + col, -w * Q.getValue(row, col))

            # Bottom-left block: w*Q
            for row in range(n_x):
                for col in range(n_x):
                    M.setValue(n_x + row, col, w * Q.getValue(row, col))

            # Bottom-right block: -A
            for row in range(n_x):
                for col in range(n_x):
                    M.setValue(n_x + row, n_x + col, -A.getValue(row, col))

            M.assemble()

            # Construct the RHS vector
            b = PETSc.Vec().createMPI(2 * n_x, comm=comm)
            b_real = PETSc.Vec().createSeq(n_x, comm=PETSc.COMM_SELF)
            b_imag = PETSc.Vec().createSeq(n_x, comm=PETSc.COMM_SELF)

            b_real.setValues(range(n_x), B_np[:, u])
            b_imag.setValues(range(n_x), np.zeros(n_x))

            b.setValues(range(n_x), b_real)
            b.setValues(range(n_x, 2 * n_x), b_imag)
            b.assemble()

            # Solve the system
            ksp = PETSc.KSP().create(comm=comm)
            ksp.setOperators(M)
            ksp.setType(PETSc.KSP.Type.PREONLY)
            ksp.getPC().setType(PETSc.PC.Type.LU)
            x = b.copy()
            ksp.solve(b, x)

            # Extract real and imaginary parts
            x_array = x.getArray()
            x_real = x_array[:n_x]
            x_imag = x_array[n_x:]

            # Compute Hw
            Hw[:, u, i] = (C_np @ x_real) + 1j * (C_np @ x_imag)
    return Hw


def vectorized_real_imag():
    # Create block-diagonal matrix M
    M = PETSc.Mat().create(comm=comm)
    M.setType("aij")
    M.setSizes([2 * n_w * n_x, 2 * n_w * n_x])
    M.setPreallocationNNZ((3 * n_x, 3 * n_x))
    M.setUp()

    # Fill block-diagonal matrix
    for i, w in enumerate(omega):
        # Top-left block: -A
        for row in range(n_x):
            for col in range(n_x):
                M.setValue(2 * i * n_x + row, 2 * i * n_x + col, -A.getValue(row, col))

        # Top-right block: -w*Q
        for row in range(n_x):
            for col in range(n_x):
                M.setValue(
                    2 * i * n_x + row,
                    2 * i * n_x + n_x + col,
                    -w * Q.getValue(row, col),
                )

        # Bottom-left block: w*Q
        for row in range(n_x):
            for col in range(n_x):
                M.setValue(
                    2 * i * n_x + n_x + row, 2 * i * n_x + col, w * Q.getValue(row, col)
                )

        # Bottom-right block: -A
        for row in range(n_x):
            for col in range(n_x):
                M.setValue(
                    2 * i * n_x + n_x + row,
                    2 * i * n_x + n_x + col,
                    -A.getValue(row, col),
                )

    M.assemble()

    # Create stacked RHS vector B_stack for each column of B
    Hw = np.zeros((n_y, n_u, n_w), dtype=complex)
    for u in range(n_u):
        B_stack = PETSc.Vec().createMPI(2 * n_w * n_x, comm=comm)
        B_local = PETSc.Vec().createSeq(B_stack.getLocalSize(), comm=PETSc.COMM_SELF)
        B_stack.getLocalVector(B_local)

        # Set local values for u-th column of B
        local_values_real = np.tile(B_np[:, u], n_w)[: B_local.getSize() // 2]
        local_values_imag = np.zeros(B_local.getSize() // 2)
        local_values = np.concatenate((local_values_real, local_values_imag))

        B_local.setValues(range(B_local.getSize()), local_values)
        B_stack.restoreLocalVector(B_local)
        B_stack.assemble()

        # Solve M * X = B_stack
        ksp = PETSc.KSP().create(comm=comm)
        ksp.setOperators(M)
        ksp.setType(PETSc.KSP.Type.PREONLY)
        ksp.getPC().setType(PETSc.PC.Type.LU)
        X = B_stack.copy()
        ksp.solve(B_stack, X)

        # Reshape and compute Hw for u-th column
        X_array = X.getArray()
        X_mat_real = X_array[: n_w * n_x].reshape(n_w, n_x)
        X_mat_imag = X_array[n_w * n_x :].reshape(n_w, n_x)

        Hw[:, u, :] = (C_np @ X_mat_real.T) + 1j * (C_np @ X_mat_imag.T)
    return Hw


# --- Benchmarking ---
def benchmark(func, name):
    if rank == 0:
        print(f"\nBenchmarking {name}...")
    comm.Barrier()
    start = time.time()
    result = func()
    end = time.time()
    elapsed = end - start
    if rank == 0:
        print(f"{name} elapsed time: {elapsed:.4f} seconds")
    return elapsed, result


# --- Run benchmarks ---
elapsed_nw, Hw_nw = benchmark(loop_over_nw_real_imag, "Loop over nw (Real/Imag)")
elapsed_vec, Hw_vec = benchmark(vectorized_real_imag, "Vectorized (Real/Imag)")

# --- Results ---
if rank == 0:
    print("\n--- Results ---")
    print(f"Loop over nw H(w) shape: {Hw_nw.shape}")
    print(f"Vectorized H(w) shape: {Hw_vec.shape}")
    print(f"Loop over nw H(w) sample:\n{Hw_nw[:, 0, :5]}")
    print(f"Vectorized H(w) sample:\n{Hw_vec[:, 0, :5]}")
    print(f"\nTiming summary:")
    print(f"Loop over nw (Real/Imag): {elapsed_nw:.4f} seconds")
    print(f"Vectorized (Real/Imag):   {elapsed_vec:.4f} seconds")
    print(f"\nResults match: {np.allclose(Hw_nw, Hw_vec, atol=1e-10)}")
