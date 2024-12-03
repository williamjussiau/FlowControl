# from scipy.sparse import csc_matrix
# from scipy.sparse.linalg import inv
# A = csc_matrix([[1., 0.], [1., 2.]])
# Ainv = inv(A)
# Ainv
# A.dot(Ainv)
# A.dot(Ainv).todense()


# It is very expensive to compute the inverse of a matrix and very rarely needed in practice.
# We highly recommend avoiding algorithms that need it. 
# The inverse of a matrix (dense or sparse) is essentially always dense, so begin by creating a
# dense matrix B and fill it with the identity matrix (ones along the diagonal), 
# also create a dense matrix X of the same size that will hold the solution. 
# Then factor the matrix you wish to invert with MatLUFactor() or MatCholeskyFactor(),
# call the result A. 
# Then call MatMatSolve(A,B,X) to compute the inverse into X. See also.

from petsc4py import PETSc

# A = PETSc.Mat()
# 
# 
# B = A.copy()
# petsc_mat = as_backend_type(B).mat()
# petsc_mat.transpose()
# A_adj_petsc = PETScMatrix(petsc_mat)
# 
# # see: PETSc.Mat.solve(self, vec b, vec x), PETSc.Mat.factorLU-factorCholesky

n = 30

# matrix to be inverted
A = PETSc.Mat().create()
A.setSizes([n, n])
A.setFromOptions()
A.setUp()

rstart, rend = A.getOwnershipRange()

# first row
if rstart == 0:
    A[0, :2] = [2, -1]
    rstart += 1
# last row
if rend == n:
    A[n-1, -2:] = [-1, 2]
    rend -= 1
# other rows
for i in range(rstart, rend):
    A[i, i-1:i+2] = [-1, 2, -1]

A.assemble()

# identity
B = PETSc.Mat().create()
B.setSizes([n, n])
B.setFromOptions()
B.setUp()

rsb, reb = B.getOwnershipRange()
for i in range(rsb, reb):
    B[i, i] = 1
B.assemble()

# solution, X = inv(A)
X = PETSc.Mat().create()
X.setSizes([n, n])
X.setFromOptions()
X.setUp()

# factor A
#myis = PETSc.IS(comm=MPI.comm_world)
#C = A.factorLU(myis, myis)

PETSc.Mat.solve(A, B, X)









