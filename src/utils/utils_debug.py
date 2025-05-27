import logging

import dolfin
import numpy as np
from dolfin import dot

# import utils_flowsolver as flu

# i#mport scipy.sparse as spr
# import sympy as sp


logger = logging.getLogger(__name__)


def check_mass_matrix(self, up=0, vq=0, random=True):
    """Given two vectors u, v (Functions on self.W),
    compute assemble(dot(u,v)*dx) v.s. u.local().T @ Q @ v.local()
    The result should be the same"""
    if random:
        logger.info("Creating random vectors")

        def createrandomfun():
            up = dolfin.Function(self.W)
            up.vector().set_local((np.random.randn(self.W.dim(), 1)))
            up.vector().apply("insert")
            return up

        up = createrandomfun()
        vq = createrandomfun()

    fa = dolfin.FunctionAssigner([self.V, self.P], self.W)
    u = dolfin.Function(self.V)  # velocity only
    p = dolfin.Function(self.P)
    v = dolfin.Function(self.V)
    q = dolfin.Function(self.P)
    fa.assign([u, p], up)
    fa.assign([v, q], vq)

    # True integral of velocities
    d1 = dolfin.assemble(dot(u, v) * self.dx)

    # Discretized dot product (scipy)
    Q = self.get_mass_matrix(sparse=True)
    d2 = up.vector().get_local().T @ Q @ vq.vector().get_local()
    ## Note: u.T @ Qv = (Qv).T @ u
    # d2 = (Q @ v.vector().get_local()).T @ u.vector().get_local()

    # Discretized dot product (petsc)
    QQ = self.get_mass_matrix(sparse=False)
    uu = dolfin.Vector(up.vector())
    vv = dolfin.Vector(vq.vector())
    ww = dolfin.Vector(up.vector())  # intermediate result
    QQ.mult(vv, ww)  # ww = QQ*vv
    d3 = uu.inner(ww)

    return {"integral": d1, "dot_scipy": d2, "dot_petsc": d3}
