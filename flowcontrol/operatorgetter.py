"""Class dedicated to operator computation on flows
For now: not working, just copied-pasted operator getters from FlowSolver to make room"""

import logging
import time

import dolfin
import flowsolver
import numpy as np
import utils_flowsolver as flu
from actuator import ACTUATOR_TYPE
from dolfin import div, dot, dx, inner, nabla_grad
from sensor import SensorIntegral, SensorPoint

logger = logging.getLogger(__name__)
FORMAT = "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)


class OperatorGetter:
    def __init__(self, flowsolver: flowsolver.FlowSolver):
        self.flowsolver = flowsolver

    def get_A(self, UP0=None, auto=True, u_ctrl=None) -> dolfin.PETScMatrix:
        """Get state-space dynamic matrix A linearized around some field UP0
        with constant input u_ctrl"""
        logger.info("Computing jacobian A...")

        # Prealloc
        Jac = dolfin.PETScMatrix()

        if UP0 is None:
            UP0 = self.flowsolver.fields.UP0

        if auto:
            # dolfin.derivative/ufl.derivative
            F0, UP0 = self.flowsolver._make_varf_steady(initial_guess=UP0)
            du = dolfin.TrialFunction(self.flowsolver.W)
            dF0 = dolfin.derivative(-1 * F0, u=UP0, du=du)
        else:
            # linearize by hand - somehow dot(f,v) not working
            U0, _ = UP0.split()
            u, p = dolfin.TrialFunctions(self.flowsolver.W)
            v, q = dolfin.TestFunctions(self.flowsolver.W)
            iRe = dolfin.Constant(1 / self.flowsolver.params_flow.Re)
            # f = self.flowsolver._gather_actuators_expressions()
            dF0 = (
                -dot(dot(U0, nabla_grad(u)), v) * dx
                - dot(dot(u, nabla_grad(U0)), v) * dx
                - iRe * inner(nabla_grad(u), nabla_grad(v)) * dx
                + p * div(v) * dx
                + div(u) * q * dx
                # + dot(f, v) * dx
            )

        if u_ctrl is None:
            self.flowsolver.flush_actuators_u_ctrl()
        else:
            self.flowsolver.set_actuators_u_ctrl(u_ctrl)

        dolfin.assemble(dF0, tensor=Jac)
        [bc.apply(Jac) for bc in self.flowsolver.bc.bcu]

        return Jac

    def get_mass_matrix(self) -> dolfin.PETScMatrix:
        """Get mass matrix associated to spatial discretization"""
        logger.info("Computing mass matrix E...")
        up = dolfin.TrialFunction(self.flowsolver.W)
        vq = dolfin.TestFunction(self.flowsolver.W)

        E = dolfin.PETScMatrix()

        mf = sum([up[i] * vq[i] for i in range(2)]) * self.flowsolver.dx  # sum u, v but not p
        dolfin.assemble(mf, tensor=E)

        return E

    def get_B(self):  # TODO keep here
        """Get actuation matrix B"""
        logger.info("Computing actuation matrix B...")

        # Save amplitude
        u_ctrl_old = self.flowsolver.get_actuators_u_ctrl()
        self.flowsolver.set_actuators_u_ctrl(self.flowsolver.params_control.actuator_number * [1.0])

        # Embed actuators in W (originally in V) + restrict spatially to boundary
        actuators_expressions = []
        for actuator in self.flowsolver.params_control.actuator_list:
            expr = ExpandFunctionSpace(actuator.expression)
            if actuator.actuator_type is ACTUATOR_TYPE.BC:
                expr = RestrictFunctionToBoundary(actuator.boundary, expr)
            actuators_expressions.append(expr)

        # Project expressions
        B = np.zeros((self.flowsolver.W.dim(), self.flowsolver.params_control.actuator_number))
        for ii, actuator_expression in enumerate(actuators_expressions):
            # Bproj = dolfin.interpolate(actuator_expression, self.flowsolver.W)
            Bproj = flu.projectm(actuator_expression, self.flowsolver.W)
            B[:, ii] = Bproj.vector().get_local()

        # Remove very small values (should be 0 but are not)
        B = flu.dense_to_sparse(B, eliminate_zeros=True, eliminate_under=1e-14)
        B = B.toarray()

        # Retrieve amplitude
        self.flowsolver.set_actuators_u_ctrl(u_ctrl_old)

        return B

    def get_C(self, check=False, fast=True):
        """Get measurement matrix C"""
        logger.info("Computing measurement matrix C...")

        W = self.flowsolver.W
        dofmap = W.dofmap()
        dofs = dofmap.dofs()

        if fast:
            # check that all sensors are SensorPoint or SensorIntegral for optimization
            all_sensors_are_compatible = np.all(
                [
                    isinstance(sensor, SensorPoint) or isinstance(sensor, SensorIntegral)
                    for sensor in self.flowsolver.params_control.sensor_list
                ]
            )

            if not all_sensors_are_compatible:
                logger.warning("Not all sensors compatible with fast implementation. Aborting.")
                return -1

            # Define union of boxes to eval: encompas all locations + subdomains
            dofmap_xy = W.tabulate_dof_coordinates()
            XYMARGIN = 0.1
            dofs_in_box = set()
            dofs_in_subdomain = set()

            for sensor in self.flowsolver.params_control.sensor_list:
                # SensorPoint
                if isinstance(sensor, SensorPoint):
                    sensor_position = sensor.position
                    xymin = (sensor_position - XYMARGIN).reshape(1, -1)
                    xymax = (sensor_position + XYMARGIN).reshape(1, -1)
                    # keep dofs with coordinates inside box
                    dofs_in_box_mask = (np.greater_equal(dofmap_xy, xymin) * np.less_equal(dofmap_xy, xymax)).all(
                        axis=1
                    )
                    # add to set (checks uniqueness)
                    dofs_in_box = dofs_in_box.union(np.asarray(dofs)[dofs_in_box_mask])

                # SensorIntegral
                if isinstance(sensor, SensorIntegral):
                    sensor_subdomain = sensor.subdomain
                    mesh = self.flowsolver.mesh
                    # mesh.entities(0) = vertices
                    # mesh.entities(1) = edges
                    # mesh.entities(2) = cells
                    dofs_in_subdomain = set()

                    # mark boundaries (=edges) in subdomain
                    bnd_markers = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
                    bnd_markers.set_all(0)
                    SUBDOMAIN_INDEX = 200
                    sensor_subdomain.mark(bnd_markers, SUBDOMAIN_INDEX)
                    edges_in_subdomain = bnd_markers.where_equal(SUBDOMAIN_INDEX)

                    vertices_idx_to_keep = set()

                    # for edges in subdomain, save vertices
                    # output: vertices in subdomain
                    for edge in dolfin.edges(mesh):
                        if edge.index() in edges_in_subdomain:
                            vertices_idx_to_keep.update(edge.entities(0))

                    # for all cells, if cell contains vertex in subdomain
                    # then save cell-dofs to dofs in subdomain
                    for cell in dolfin.cells(mesh):
                        if len(vertices_idx_to_keep.intersection(cell.entities(0))):
                            dofs_in_subdomain.update(dofmap.cell_dofs(cell.index()))

            # Union of dofs if mixed BC/force actuators
            dofs_to_parse = dofs_in_box.union(dofs_in_subdomain)

        else:  # no fast = parse all dofs
            dofs_to_parse = dofs

        # Fill matrix C
        uvp = dolfin.Function(W)
        uvp_vec = uvp.vector()
        C = np.zeros((self.flowsolver.params_control.sensor_number, W.dim()))
        idof_old = 0
        # Iteratively put each DOF at 1 and evaluate measurement on said DOF
        for idof in dofs_to_parse:
            uvp_vec[idof] = 1
            if idof_old > 0:
                uvp_vec[idof_old] = 0
            idof_old = idof
            C[:, idof] = self.flowsolver.make_measurement(uvp)

        # check:
        if check:
            for i in range(self.flowsolver.params_control.sensor_number):
                logger.debug(f"with fun: {self.flowsolver.make_measurement(self.flowsolver.fields.UP0)}")
                logger.debug(f"with C@x: {C[i, :] @ self.flowsolver.fields.UP0.vector().get_local()}")

        return C


####################################################################################
####################################################################################
####################################################################################
####################################################################################
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
        v, q = dolfin.TestFunctions(self.flowsolver.W)
        iRe = dolfin.Constant(1 / self.flowsolver.params_flow.Re)
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

        # for an exponential actuator -> just evaluate actuator_exp on every coordinate?
        # for a boundary actuator -> evaluate actuator on boundary
        actuator_ampl_old = self.flowsolver.actuator_expression.ampl
        self.flowsolver.actuator_expression.ampl = 1.0

        # Method 1
        # restriction of actuation of boundary
        class RestrictFunctionToBoundary(dolfin.UserExpression):
            def __init__(self, boundary, fun, **kwargs):
                self.boundary = boundary
                self.fun = fun
                super(RestrictFunctionToBoundary, self).__init__(**kwargs)

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
            actuator_restricted = RestrictFunctionToBoundary(
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


class RestrictFunctionToBoundary(dolfin.UserExpression):
    def __init__(self, boundary, fun, **kwargs):
        self.boundary = boundary
        self.fun = fun
        super(RestrictFunctionToBoundary, self).__init__(**kwargs)

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