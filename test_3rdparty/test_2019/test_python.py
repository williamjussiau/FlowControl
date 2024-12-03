# Poisson equation
# ================
# 
# This demo is implemented in a single Python file,
# :download:`demo_poisson.py`, which contains both the variational forms
# and the solver.
# 
# This demo illustrates how to:
# 
# * Solve a linear partial differential equation
# * Create and apply Dirichlet boundary conditions
# * Define Expressions
# * Define a FunctionSpace
# * Create a SubDomain

from dolfin import *
set_log_level(LogLevel.TRACE)

import time

t0 = time.time()

# Create mesh and define function space
nz = 400
mesh = UnitSquareMesh(nz, nz)
V = FunctionSpace(mesh, "Lagrange", 2)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
g = Expression("sin(5*x[0])", degree=2)
a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds

# Compute solution
u = Function(V)
# solver_param = {"linear_solver":"mumps"}
solver_param = {"linear_solver":"bicgstab", "preconditioner":"hypre_amg"}
solve(a == L, u, bc, solver_parameters=solver_param)

# Save solution in VTK format
# file = File("/stck/wjussiau/fenics-python/test_2019/results/poisson.pvd")
# file << u

print('----------------------- Elasped (Python) time is: ', time.time() - t0)





