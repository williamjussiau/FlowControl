"""Class dedicated to operator computation on flows
For now: not working, just copied-pasted operator getters from FlowSolver to make room"""

import logging
import time
from abc import ABC, abstractmethod

import dolfin
import flowsolver
import numpy as np
import utils_flowsolver as flu
from dolfin import div, dot, dx, inner, nabla_grad

logger = logging.getLogger(__name__)
FORMAT = "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)


class OperatorGetter(ABC):
    def __init__(self, flowsolver: flowsolver.FlowSolver):
        self.flowsolver.flowsolver = flowsolver

    @abstractmethod
    def get_A(self):
        pass

    @abstractmethod
    def get_B(self):
        pass

    @abstractmethod
    def get_C(self):
        pass


class CylinderOperatorGetter(OperatorGetter):
    # Matrix computations
    def get_A(
        self, perturbations=True, shift=0.0, timeit=True, UP0=None
    ):  # TODO idk, merge with make_mixed_form?
        """Get state-space dynamic matrix A linearized around some field UP0"""
        logger.info("Computing jacobian A...")

        if timeit:
            t0 = time.time()

        Jac = dolfin.PETScMatrix()
        v, q = dolfin.TestFunctions(self.flowsolver.flowsolver.W)
        iRe = dolfin.Constant(1 / self.flowsolver.flowsolver.params_flow.Re)
        shift = dolfin.Constant(shift)

        if UP0 is None:
            UP_ = self.flowsolver.fields.UP0  # base flow
        else:
            UP_ = UP0
        U_, p_ = UP_.split()

        if perturbations:  # perturbation equations linearized
            up = dolfin.TrialFunction(self.flowsolver.W)
            u, p = dolfin.split(up)
            dF0 = (
                -dot(dot(U_, nabla_grad(u)), v) * dx
                - dot(dot(u, nabla_grad(U_)), v) * dx
                - iRe * inner(nabla_grad(u), nabla_grad(v)) * dx
                + p * div(v) * dx
                + div(u) * q * dx
                - shift * dot(u, v) * dx
            )  # sum u, v but not p
            # create zeroBC for perturbation formulation
            bcu_inlet = dolfin.DirichletBC(
                self.flowsolver.W.sub(0),
                dolfin.Constant((0, 0)),
                self.flowsolver.boundaries.loc["inlet"].subdomain,
            )
            bcu_walls = dolfin.DirichletBC(
                self.flowsolver.W.sub(0).sub(1),
                dolfin.Constant(0),
                self.flowsolver.boundaries.loc["walls"].subdomain,
            )
            bcu_cylinder = dolfin.DirichletBC(
                self.flowsolver.W.sub(0),
                dolfin.Constant((0, 0)),
                self.flowsolver.boundaries.loc["cylinder"].subdomain,
            )
            bcu_actuation_up = dolfin.DirichletBC(
                self.flowsolver.W.sub(0),
                self.flowsolver.actuator_expression,
                self.flowsolver.boundaries.loc["actuator_up"].subdomain,
            )
            bcu_actuation_lo = dolfin.DirichletBC(
                self.flowsolver.W.sub(0),
                self.flowsolver.actuator_expression,
                self.flowsolver.boundaries.loc["actuator_lo"].subdomain,
            )
            bcu = [
                bcu_inlet,
                bcu_walls,
                bcu_cylinder,
                bcu_actuation_up,
                bcu_actuation_lo,
            ]
            self.flowsolver.actuator_expression.ampl = 0.0
            bcs = bcu
        else:
            F0 = (
                -dot(dot(U_, nabla_grad(U_)), v) * dx
                - iRe * inner(nabla_grad(U_), nabla_grad(v)) * dx
                + p_ * div(v) * dx
                + q * div(U_) * dx
                - shift * dot(U_, v) * dx
            )
            # prepare derivation
            du = dolfin.TrialFunction(self.flowsolver.W)
            dF0 = dolfin.derivative(F0, UP_, du=du)
            # import pdb
            # pdb.set_trace()
            ## shift
            # dF0 = dF0 - shift*dot(U_,v)*dx
            # bcs)
            self.flowsolver.actuator_expression.ampl = 0.0
            bcs = self.flowsolver.bc["bcu"]

        dolfin.assemble(dF0, tensor=Jac)
        [bc.apply(Jac) for bc in bcs]

        if timeit:
            logger.info(f"Elapsed time: {time.time() - t0}")

        return Jac

    def get_B(self, export=False, timeit=True):  # TODO keep here
        """Get actuation matrix B"""
        logger.info("Computing actuation matrix B...")

        if timeit:
            t0 = time.time()

        # for an exponential actuator -> just evaluate actuator_exp on every coordinate, kinda?
        # for a boundary actuator -> evaluate actuator on boundary
        actuator_ampl_old = self.flowsolver.actuator_expression.ampl
        self.flowsolver.actuator_expression.ampl = 1.0

        # Method 1
        # restriction of actuation of boundary
        class RestrictFunction(dolfin.UserExpression):
            def __init__(self, boundary, fun, **kwargs):
                self.boundary = boundary
                self.fun = fun
                super(RestrictFunction, self).__init__(**kwargs)

            def eval(self, values, x):
                values[0] = 0
                values[1] = 0
                values[2] = 0
                if self.boundary.inside(x, True):
                    evalval = self.fun(x)
                    values[0] = evalval[0]
                    values[1] = evalval[1]

            def value_shape(self):
                return (3,)

        Bi = []
        for actuator_name in ["actuator_up", "actuator_lo"]:
            actuator_restricted = RestrictFunction(
                boundary=self.flowsolver.boundaries.loc[actuator_name].subdomain,
                fun=self.flowsolver.actuator_expression,
            )
            actuator_restricted = dolfin.interpolate(
                actuator_restricted, self.flowsolver.W
            )
            # actuator_restricted = flu.projectm(actuator_restricted, self.flowsolver.W)
            Bi.append(actuator_restricted)

        # this is supposedly B
        B_all_actuator = flu.projectm(sum(Bi), self.flowsolver.W)
        # get vector
        B = B_all_actuator.vector().get_local()
        # remove very small values (should be 0 but are not)
        B = flu.dense_to_sparse(
            B, eliminate_zeros=True, eliminate_under=1e-14
        ).toarray()
        B = B.T  # vertical B

        if export:
            vv = dolfin.Function(self.flowsolver.V)
            ww = dolfin.Function(self.flowsolver.W)
            ww.assign(B_all_actuator)
            vv, pp = ww.split()
            flu.write_xdmf("B.xdmf", vv, "B")

        self.flowsolver.actuator_expression.ampl = actuator_ampl_old

        if timeit:
            logger.info(f"Elapsed time: {time.time() - t0}")

        return B

    def get_C(self, timeit=True, check=False):  # TODO keep here
        """Get measurement matrix C"""
        logger.info("Computing measurement matrix C...")

        if timeit:
            t0 = time.time()

        fspace = self.flowsolver.W
        uvp = dolfin.Function(fspace)
        uvp_vec = uvp.vector()
        dofmap = fspace.dofmap()

        ndof = fspace.dim()
        ns = self.flowsolver.params_flow.sensor_nr
        C = np.zeros((ns, ndof))

        idof_old = 0
        # Iteratively put each DOF at 1 and evaluate measurement on said DOF
        for idof in dofmap.dofs():
            uvp_vec[idof] = 1
            if idof_old > 0:
                uvp_vec[idof_old] = 0
            idof_old = idof
            C[:, idof] = self.flowsolver.make_measurement(mixed_field=uvp)

        # check:
        if check:
            for i in range(ns):
                sensor_types = dict(u=0, v=1, p=2)
                logger.debug(
                    f"True probe: {self.flowsolver.up0(self.flowsolver.sensor_location[i])[sensor_types[self.flowsolver.sensor_type[0]]]}"
                )
                logger.debug(
                    f"\t with fun: {self.flowsolver.make_measurement(mixed_field=self.flowsolver.up0)}"
                )
                logger.debug(
                    f"\t with C@x: {C[i] @ self.flowsolver.up0.vector().get_local()}"
                )

        if timeit:
            logger.info(f"Elapsed time: {time.time() - t0}")

        return C


class CavityOperatorGetter(OperatorGetter):
    def get_A(self, perturbations=True, shift=0.0, timeit=True, up_0=None):
        """Get state-space dynamic matrix A around some state up_0"""
        if timeit:
            print("Computing jacobian A...")
            t0 = time.time()

        Jac = dolfin.PETScMatrix()
        v, q = dolfin.TestFunctions(self.flowsolver.W)
        iRe = dolfin.Constant(1 / self.flowsolver.Re)
        shift = dolfin.Constant(shift)
        self.flowsolver.actuator_expression.ampl = 0.0

        if up_0 is None:
            up_ = self.flowsolver.up0  # base flow
        else:
            up_ = up_0
        u_, p_ = up_.dolfin.split()

        if perturbations:  # perturbation equations lidolfin.nearized
            up = dolfin.TrialFunction(self.flowsolver.W)
            u, p = dolfin.split(up)
            dF0 = (
                -dot(dot(u_, nabla_grad(u)), v) * dx
                - dot(dot(u, nabla_grad(u_)), v) * dx
                - iRe * inner(nabla_grad(u), nabla_grad(v)) * dx
                + p * div(v) * dx
                + div(u) * q * dx
                - shift * dot(u, v) * dx
            )  # sum u, v but not p
            bcu = self.flowsolver.bc_p["bcu"]
            bcs = bcu
        else:  # full ns + derivative
            up_ = self.flowsolver.up0
            u_, p_ = dolfin.split(up_)
            F0 = (
                -dot(dot(u_, nabla_grad(u_)), v) * dx
                - iRe * inner(nabla_grad(u_), nabla_grad(v)) * dx
                + p_ * div(v) * dx
                + q * div(u_) * dx
                - shift * dot(u_, v) * dx
            )
            # prepare derivation
            du = dolfin.TrialFunction(self.flowsolver.W)
            dF0 = dolfin.derivative(F0, up_, du=du)
            bcs = self.flowsolver.bc["bcu"]

        dolfin.assemble(dF0, tensor=Jac)
        [bc.apply(Jac) for bc in bcs]

        if timeit:
            print("Elapsed time: ", time.time() - t0)

        return Jac

    def get_B(self, export=False, timeit=True):
        """Get actuation matrix B"""
        print("Computing actuation matrix B...")

        if timeit:
            t0 = time.time()

        # for an exponential actuator -> just evaluate actuator_exp on every coordinate, kinda?
        # for a boundary actuator -> evaluate actuator on boundary
        actuator_ampl_old = self.flowsolver.actuator_expression.ampl
        self.flowsolver.actuator_expression.ampl = 1.0

        # Method 2
        # Projet actuator expression on W
        class ExpandFunctionSpace(dolfin.UserExpression):
            """Expand function from space [V1, V2] to [V1, V2, P]"""

            def __init__(self, fun, **kwargs):
                self.fun = fun
                super(ExpandFunctionSpace, self).__init__(**kwargs)

            def eval(self, values, x):
                evalval = self.fun(x)
                values[0] = evalval[0]
                values[1] = evalval[1]
                values[2] = 0

            def value_shape(self):
                return (3,)

        actuator_extended = ExpandFunctionSpace(fun=self.actuator_expression)
        actuator_extended = dolfin.interpolate(actuator_extended, self.flowsolver.W)
        B_proj = flu.projectm(actuator_extended, self.flowsolver.W)
        B = B_proj.vector().get_local()

        # remove very small values (should be 0 but are not)
        B = flu.dense_to_sparse(
            B, eliminate_zeros=True, eliminate_under=1e-14
        ).toarray()
        B = B.T  # vertical B

        if export:
            fa = dolfin.FunctionAssigner(
                [self.flowsolver.V, self.flowsolver.P], self.flowsolver.W
            )
            vv = dolfin.Function(self.flowsolver.V)
            pp = dolfin.Function(self.flowsolver.P)
            ww = dolfin.Function(self.flowsolver.W)
            ww.assign(B_proj)
            fa.assign([vv, pp], ww)
            flu.write_xdmf("B.xdmf", vv, "B")

        self.flowsolver.actuator_expression.ampl = actuator_ampl_old

        if timeit:
            print("Elapsed time: ", time.time() - t0)

        return B

    def get_C(self, timeit=True, check=False, verbose=False):
        """Get measurement matrix C"""
        # Solution to make it faster:
        # localize the region of dofs where C is going to be nonzero
        # and only account for dofs in this region
        print("Computing measurement matrix C...")

        if timeit:
            t0 = time.time()

        # Initialize
        fspace = self.flowsolver.W  # function space
        uvp = dolfin.Function(fspace)  # function to store C
        uvp_vec = uvp.vector()  # as vector
        ndof = fspace.dim()  # size of C
        ns = self.flowsolver.params_control.sensor_number
        C = np.zeros((ns, ndof))

        dofmap = fspace.dofmap()  # indices of dofs
        dofmap_x = fspace.tabulate_dof_coordinates()  # coordinates of dofs

        # Box that encapsulates all dofs on sensor
        margin = 0.05
        xmin = 1 - margin
        xmax = 1.1 + margin
        ymin = 0 - margin
        ymax = 0 + margin
        xymin = np.array([xmin, ymin]).reshape(1, -1)
        xymax = np.array([xmax, ymax]).reshape(1, -1)
        # keep dofs with coordinates inside box
        dof_in_box = (
            np.greater_equal(dofmap_x, xymin) * np.less_equal(dofmap_x, xymax)
        ).all(axis=1)
        # retrieve said dof index
        dof_in_box_idx = np.array(dofmap.dofs())[dof_in_box]

        # Iteratively put each DOF at 1
        # And evaluate measurement on said DOF
        idof_old = 0
        ii = 0  # counter of the number of dofs evaluated
        for idof in dof_in_box_idx:
            ii += 1
            if verbose and not ii % 1000:
                print("get_C::eval iter {0} - dof n°{1}/{2}".format(ii, idof, ndof))
            # set field 1 at said dof
            uvp_vec[idof] = 1
            uvp_vec[idof_old] = 0
            idof_old = idof
            # retrieve coordinates
            # dof_x = dofmap_x[idof] # not needed for measurement
            # evaluate measurement
            C[:, idof] = self.flowsolver.make_measurement(mixed_field=uvp)

        # check:
        if check:
            for i in range(ns):
                print(
                    "\t with fun:",
                    self.flowsolver.make_measurement(mixed_field=self.flowsolver.up0),
                )
                print("\t with C@x: ", C[i] @ self.flowsolver.up0.vector().get_local())

        if timeit:
            print("Elapsed time: ", time.time() - t0)

        return C
        return C
