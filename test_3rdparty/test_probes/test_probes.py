from dolfin import *
from fenicstools.Probe import *

# Test the probe functions:
set_log_level(20)

nn = 16
mesh = UnitCubeMesh(nn, nn, nn)
Ve = FiniteElement('CG', mesh.ufl_cell(), 1)
Vve = VectorElement('CG', mesh.ufl_cell(), 1)
We = MixedElement([Ve, Vve])
W = FunctionSpace(mesh, We) 

# Just create some random data to be used for probing
w0 = interpolate(Expression(('x[0]', 'x[1]', 'x[2]', 'x[1]*x[2]'), degree=2), W)

x = array([[0.5, 0.5, 0.5],
           [0.2, 0.3, 0.4],
           [0.8, 0.9, 1.0],
           [20, 10.1, 4.3]])
p = Probes(x.flatten(), W)
#x = x*0.9 
#p.add_positions(x.flatten(), W)
p.get_total_number_probes())
for i in range(2): # probe n times 
    p(w0)


print('probes at:\n', x)
print('value is: \n', p.array(2, "testarray"))         # dump snapshot 2
#print(p.array(filename="testarray"))   # dump all snapshots
#print(p.dump("testarray"))


