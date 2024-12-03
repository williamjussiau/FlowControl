import petsc4py
import sys
import time
import numpy as np

petsc4py.init(sys.argv)

# import tools
print('Importing petsc4py.PETSc')
from petsc4py import PETSc



if 0:
    # grid size
    print('Defining grid sizes')
    m, n = 32, 32
    hx = 1.0/(m-1)
    hy = 1.0/(n-1)
    sz = m*n
    
    # create mat
    print('Creating matrix')
    M = PETSc.Mat()
    M.create(PETSc.COMM_WORLD)
    M.setSizes([sz, sz])
    M.setType('aij') # sparse
    M.setUp()
    
    # compute entries
    print('Computing entries')
    diagv = 2.0/hx**2 + 2.0/hy**2
    offdx = -1.0/hx**2
    offdy = -1.0/hy**2
    
    # assemble matrix
    t00 = time.time()
    print('Assembling matrix')
    Istart, Iend = M.getOwnershipRange()
    
    is_set_value = True
    
    for I in range(Istart, Iend):
        M.setValue(I, I, diagv)
        i = I//n
        j = I - i*n
        if i>0:
            J = I-n
            M.setValue(I, J, offdx)
        if i<m-1:
            J = I+n
            M.setValue(I, J, offdx)
        if j>0:
            J =I-1
            M.setValue(I, J, offdy)
        if j<n-1:
            J = I+1
            M.setValue(I, J, offdy)
    
    M.assemblyBegin()
    M.assemblyEnd()
    
    w = 4.3
    
    # identity matrix
    Z = PETSc.Mat()
    Z.create(PETSc.COMM_WORLD)
    Z.setSizes([sz, sz])
    Z.setType('aij')
    Z.setUp()
    
    #VV = PETSc.Vec()
    #VV.create(PETSc.COMM_WORLD)
    #VV.setSizes(sz)
    #VV.setUp()
    #VV.setArray(w*np.ones(sz))
    #Z.setDiagonal(VV)
    
    Z.assemblyBegin()
    Z.assemblyEnd()
    
    print('Total assembly time is: ', time.time() - t00)




####################################################
####################################################
####################################################
####################################################
# block matrix
###BLK = PETSc.Mat()
###BLK.createNest([[M, Z, Z, M]])
###BLK.setType('aij')
###BLK.setUp()
###BLK.assemblyBegin()
###BLK.assemblyEnd()

# GOTO1

# ce que l'on veut faire ici:
# montrer qu'il existe des techniques plus 
# rapides que "loop over rows + setValues"
# pour remplir une grosse matrice
# plus : on veut definir cette matrice par 
# blocs afin d'etre au plus pres du cas
# d'utilisation final

# prop 1: create csr
# prop 2: pre allocate nnz
# prop 3: pre allocate csr

# first create & measure runtime
# then check that result is correct


# grid size
print('Defining grid sizes')
sz = 500
rho = 0.01

import scipy.sparse as sp

# true data
# c, s, r = get_A().mat().getValuesCSR()
# dummy data
A = sp.random(sz, sz, density=rho, format='csr', random_state=1)
col_idx = A.indices
row_idx = A.indptr
val = A.data
nnzA = A.nnz
# Ad = A.toarray()
Acsr = (A.indptr, A.indices, A.data)

COMM = PETSc.COMM_WORLD

# prop 1: pre allocate aij
print('Creating matrix --- 1')
t1 = time.time()
R1 = PETSc.Mat()
R1.createAIJWithArrays(size=[sz, sz], csr=Acsr, comm=COMM)
R1.assemblyBegin()
R1.assemblyEnd()
print('Prop 1: ', time.time() - t1)

# prop 2: pre allocate nnz
print('Creating matrix --- 2')
t2 = time.time()
R2 = PETSc.Mat()
R2.create(comm=COMM)
R2.setSizes([sz, sz])
R2.setType('aij') # sparse
R2.setUp()
R2.setPreallocationNNZ(nnzA)
Istart, Iend = R2.getOwnershipRange()
for i in range(Istart, Iend):
    rowi = A.getrow(i)
    R2.setValues(i, A.getrow(i).indices, A.getrow(i).data)
R2.assemblyBegin()
R2.assemblyEnd()
print('Prop 2: ', time.time() - t2)

# prop 3: pre allocate csr (aij)
print('Creating matrix --- 3')
t3 = time.time()
R3 = PETSc.Mat()
R3.create(comm=COMM)
R3.setSizes([sz, sz])
R3.setType('aij') # sparse
R3.setUp()
R3.setPreallocationCSR(csr=Acsr)
Istart, Iend = R2.getOwnershipRange()
for i in range(Istart, Iend):
    rowi = A.getrow(i)
    R3.setValues(i, A.getrow(i).indices, A.getrow(i).data)
R3.assemblyBegin()
R3.assemblyEnd()
print('Prop 3: ', time.time() - t3)



# check
print('Norm of dense matrix         : ', np.linalg.norm(A.data))
print('Norm of createAIJ            : ', R1.norm())
print('Norm of setValues (alloc NNZ): ', R2.norm())
print('Norm of setValues (alloc CSR): ', R3.norm())


###########################################################
# Prop 1 is the fastest
# Now: make block matrix
###########################################################
# A is a sp matrix here

w = 0.31
print('')
I = sp.identity(sz)
Ablk = sp.vstack([sp.hstack([-A, -w*I]), sp.hstack([w*I, -A])])
Ablk = Ablk.tocsr()
Ablk_csr = (Ablk.indptr, Ablk.indices, Ablk.data)
print('Creating matrix --- BLOCK')
tb = time.time()
Rb = PETSc.Mat()
Rb.createAIJWithArrays(size=[2*sz, 2*sz], csr=Ablk_csr, comm=COMM)
Rb.assemblyBegin()
Rb.assemblyEnd()
print('Making block: ', time.time() - tb)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.spy(Ablk, markersize=1)
fig.savefig('spy.png')


sys.exit()

# prop 3: ?
# compute entries
print('Computing entries')
w = 0.1

# assemble matrix
t00 = time.time()
print('Assembling matrix')
Istart, Iend = R.getOwnershipRange()

# Block matrix
# R = [  -A   |  -wI  ]
#     [  wI   |   -A  ]
blk_next = sz

# this is done once
for i in range(Istart, Iend//2):
    print('Filling row: {0}/{1}'.format(i, Iend//2))
    R.setValues(i, A.getrow(i)[0].astype(np.int32), -A.getrow(i)[1]) 

for i in range(Iend//2, Iend):
    print('Filling row: {0}/{1}'.format(i, Iend//2))
    R.setValues(i+blk_next, A.getrow(i)[0].astype(np.int32)+blk_next, -A.getrow(i)[1]) 

# fill in diagonal
for i in range(Istart, Iend//2):
    # fill upper right 
    R.setValue(i, i+blk_next, -w)

for i in range(Iend//2, Iend):
    # fill lower left
    R.setValue(i, i, w)


















