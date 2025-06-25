"""Class dedicated to operator computation on flows
For now: not working, just copied-pasted operator getters from FlowSolver to make room"""

import logging
from typing import Optional

import dolfin
import numpy as np
from dolfin import div, dot, dx, inner, nabla_grad
from numpy.typing import NDArray

import utils.utils_flowsolver as flu
from flowcontrol import flowsolver
from flowcontrol.actuator import ACTUATOR_TYPE
from flowcontrol.sensor import Sensor, SensorIntegral, SensorPoint

logger = logging.getLogger(__name__)
FORMAT = "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)


class OperatorGetter:
    def __init__(self, flowsolver: flowsolver.FlowSolver):
        self.flowsolver = flowsolver

    def get_A(
        self,
        UP0: Optional[dolfin.Function] = None,
        autodiff: bool = True,
        u_ctrl: Optional[NDArray[np.float64]] = None,
    ) -> dolfin.PETScMatrix:
        """Get state-space dynamic matrix A linearized around some field UP0
        with constant input u_ctrl"""
        logger.info("Computing jacobian A...")

        # Prealloc
        Jac = dolfin.PETScMatrix()

        if UP0 is None:
            UP0 = self.flowsolver.fields.UP0

        if autodiff:  # dolfin.derivative/ufl.derivative
            F0, UP0 = self.flowsolver._make_varf_steady(initial_guess=UP0)
            du = dolfin.TrialFunction(self.flowsolver.W)
            dF0 = dolfin.derivative(-1 * F0, u=UP0, du=du)
        else:  # linearize by hand - somehow dot(f,v) not working
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

        mf = (up[0] * vq[0] + up[1] * vq[1]) * self.flowsolver.dx  # sum u, v but not p
        dolfin.assemble(mf, tensor=E)

        return E

    def get_B(
        self, as_function_list: bool = False, interpolate: bool = True
    ) -> NDArray[np.float64] | list[dolfin.Function]:
        """Get actuation matrix B"""
        logger.info("Computing actuation matrix B...")

        # Save amplitude
        u_ctrl_old = self.flowsolver.get_actuators_u_ctrl()
        self.flowsolver.set_actuators_u_ctrl(
            self.flowsolver.params_control.actuator_number * [1.0]
        )

        try:
            # Embed actuators in W (originally in V) + restrict spatially to boundary
            actuators_expressions = []
            for actuator in self.flowsolver.params_control.actuator_list:
                expr = ExpandFunctionSpace(actuator.expression)
                if actuator.actuator_type is ACTUATOR_TYPE.BC:
                    expr = RestrictFunctionToBoundary(actuator.boundary, expr)
                actuators_expressions.append(expr)

            # Determine local range and size
            mpi_local_size = len(self.flowsolver.W.dofmap().dofs())

            if as_function_list:
                B = []
            else:
                B = np.zeros(
                    (mpi_local_size, self.flowsolver.params_control.actuator_number),
                    dtype=float,
                )

            # Project expressions
            for ii, actuator_expression in enumerate(actuators_expressions):
                if interpolate:
                    Bproj = dolfin.interpolate(actuator_expression, self.flowsolver.W)
                else:
                    Bproj = flu.projectm(actuator_expression, self.flowsolver.W)
                Bproj.vector().apply("insert")

                if as_function_list:
                    B.append(Bproj)
                else:
                    B[:, ii] = Bproj.vector().get_local()

        finally:
            # Retrieve amplitude
            self.flowsolver.set_actuators_u_ctrl(u_ctrl_old)

        return B

    def get_C(self, check: bool = False, fast: bool = True) -> NDArray[np.float64]:
        """Get measurement matrix C"""
        logger.info("Computing measurement matrix C...")

        T0 = dolfin.Timer("Custom/Initialization and optimization of dofs")
        T0.start()
        W = self.flowsolver.W
        dofmap = W.dofmap()
        dofmap_xy = W.tabulate_dof_coordinates()
        dofs = dofmap.dofs()
        sensor_list = self.flowsolver.params_control.sensor_list

        if fast:
            sensors_are_compatible = [
                sensor_is_compatible(sensor) for sensor in sensor_list
            ]

            if not np.all(sensors_are_compatible):
                logger.warning(
                    "Not all sensors compatible with fast implementation. Aborting."
                )
                return -1

            dofs_to_parse = optimize_parsed_dofs(
                dofmap, dofmap_xy, self.flowsolver.mesh, sensor_list
            )
        else:  # no fast = parse all dofs
            dofs_to_parse = dofs

        T0.stop()

        # Fill matrix C
        uvp = dolfin.Function(W)
        uvp_vec = uvp.vector()
        mpi_local_size = len(self.flowsolver.W.dofmap().dofs())
        sensor_number = self.flowsolver.params_control.sensor_number
        C = np.zeros((sensor_number, mpi_local_size))

        # Iteratively put each DOF at 1 and evaluate measurement on said DOF
        mpi_rank = dolfin.MPI.comm_world.Get_rank()
        ownership_range = dofmap.ownership_range()

        logger.info(f"Process {mpi_rank} - dofs has size: {len(dofs_to_parse)}")

        # Synchronize loop content
        dofs_to_parse_global = dolfin.MPI.comm_world.allgather(dofs_to_parse)
        dofs_to_parse_global = sorted(set().union(*dofs_to_parse_global))

        dofs_to_parse_global_dummy = set()
        for ii, elem in enumerate(dofs_to_parse_global):
            dofs_to_parse_global_dummy.add(elem)
            if ii == 200:
                break

        logger.info(
            f"Process {mpi_rank} - dofs GLOBAL has size: {len(dofs_to_parse_global)}"
        )

        T1 = dolfin.Timer("Custom/Set vector to 0")
        T21 = dolfin.Timer("Custom/Check ownership")
        T22 = dolfin.Timer("Custom/Set local")
        T23 = dolfin.Timer("Custom/Insert")
        T3 = dolfin.Timer("Custom/Make measurement")

        # Loop
        jj = 1
        for idof in dofs_to_parse_global:
            T1.start()
            local_array = uvp_vec.get_local()
            local_array[:] = 0.0
            T1.stop()

            T21.start()
            idof_local = None
            if idof in range(*ownership_range):
                # we should know in advance whether it is in process range
                # if it was computed by process, then it is in process range
                # so: if idof in dofs_to_parse (which is local)
                # but it is probably longer
                idof_local = idof - ownership_range[0]
                local_array[idof_local] = 1.0
            T21.stop()

            T22.start()
            uvp_vec.set_local(local_array)  # very costly!
            T22.stop()
            T23.start()
            uvp_vec.apply("insert")  # zero time
            T23.stop()

            T3.start()
            # Ci = np.ones((sensor_number,)) * mpi_rank
            Ci = self.flowsolver.make_measurement(uvp)  # this requires all processes

            if idof_local is not None:
                C[:, idof_local] = Ci
            T3.stop()

            jj += 1
            if not jj % 100:
                logger.info(f"Progress: {jj}/{len(dofs_to_parse_global)}")

        logger.info(f"Finished filling C of size {C.shape} on process {mpi_rank}")

        # check:
        if check:
            pass
            # logger.info("Checking result of computing C with y0 = C @ U0")
            # logger.info(f"From process: {mpi_rank}")
            # for i in range(self.flowsolver.params_control.sensor_number):
            #     # logger.debug(f"with fun: {self.flowsolver.make_measurement(self.flowsolver.fields.UP0)}")
            #     logger.info("Call 1")
            #     Cx_local = C[i, :] @ self.flowsolver.fields.UP0.vector().get_local()
            #     logger.info("Call 2")
            #     Cx_global = np.zeros((1,), dtype=float)
            #     logger.info("Call 3")
            #     dolfin.MPI.comm_world.Allreduce(Cx_local, Cx_global, op=MPI.SUM)
            #     logger.info("Call 4")
            #     logger.debug(f"with C@x: {Cx_global}")

        return C


####################################################################################
####################################################################################


class ExpandFunctionSpace(dolfin.UserExpression):
    """Expand function from space [V1, V2] to [V1, V2, P]"""

    def __init__(self, fun, **kwargs):
        self.fun = fun
        super(ExpandFunctionSpace, self).__init__(**kwargs)

    def eval(self, values, x):
        fun_x = self.fun(x)
        if len(fun_x) != 2:
            raise ValueError(f"Expected 2D vector from fun(x), got: {fun_x}")
        values[0] = fun_x[0]
        values[1] = fun_x[1]
        values[2] = 0

    def value_shape(self):
        return (3,)


class RestrictFunctionToBoundary(dolfin.UserExpression):
    def __init__(self, boundary, fun, **kwargs):
        self.boundary = boundary
        self.fun = fun
        super(RestrictFunctionToBoundary, self).__init__(**kwargs)

    def eval(self, values, x):
        values[:] = 0.0
        if self.boundary.inside(x, True):
            values[:2] = self.fun(x)[:2]

    def value_shape(self):
        return (3,)


def sensor_is_compatible(sensor):
    return isinstance(sensor, SensorPoint) or isinstance(sensor, SensorIntegral)


def find_vertices_from_edges(mesh, edges_idx):
    # for edges in subdomain, save vertices
    # output: vertices in subdomain
    vertices_idx = set()
    for edge in dolfin.edges(mesh):
        if edge.index() in edges_idx:
            vertices_idx.update(edge.entities(0))
    return vertices_idx


def mark_boundaries(mesh, subdomain, SUBDOMAIN_INDEX=200):
    boundary_markers = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    boundary_markers.set_all(0)
    subdomain.mark(boundary_markers, SUBDOMAIN_INDEX)
    return boundary_markers


def cell_owns_at_least_a_vertex(cell, vertices_idx):
    return len(vertices_idx.intersection(cell.entities(0)))


def optimize_parsed_dofs(
    dofmap: dolfin.DofMap,
    dofmap_xy: NDArray[np.float64],
    mesh: dolfin.Mesh,
    sensor_list: list[Sensor],
):
    # Define union of boxes to eval: encompas all locations + subdomains
    # dofmap_xy = W.tabulate_dof_coordinates()
    XYMARGIN = 0.05
    dofs_in_box = set()
    dofs_in_subdomain = set()

    dofs = dofmap.dofs()

    for sensor in sensor_list:
        # SensorPoint
        if isinstance(sensor, SensorPoint):
            sensor_position = sensor.position
            xymin = (sensor_position - XYMARGIN).reshape(1, -1)
            xymax = (sensor_position + XYMARGIN).reshape(1, -1)
            # keep dofs with coordinates inside box
            dofs_in_box_mask = (
                np.greater_equal(dofmap_xy, xymin) * np.less_equal(dofmap_xy, xymax)
            ).all(axis=1)
            # add to set (checks uniqueness)
            dofs_in_box.update(np.asarray(dofs)[dofs_in_box_mask])

        # SensorIntegral
        if isinstance(sensor, SensorIntegral):
            sensor_subdomain = sensor.subdomain
            # mesh.entities(0) = vertices
            # mesh.entities(1) = edges
            # mesh.entities(2) = cells

            SUBDOMAIN_INDEX = 200
            boundary_markers = mark_boundaries(mesh, sensor_subdomain, SUBDOMAIN_INDEX)

            # find edges
            edges_idx_in_subdomain = boundary_markers.where_equal(SUBDOMAIN_INDEX)

            # find vertices
            vertices_idx_in_subdomain = find_vertices_from_edges(
                mesh, edges_idx_in_subdomain
            )

            # find cells owning vertices
            cells_idx_near_subdomain = set()
            for cell in dolfin.cells(mesh):
                if cell_owns_at_least_a_vertex(cell, vertices_idx_in_subdomain):
                    cells_idx_near_subdomain.add(cell.index())

            # find dofs of cells
            dofs_in_subdomain = set()
            for cell_idx in cells_idx_near_subdomain:
                dofs_in_subdomain.update(dofmap.cell_dofs(cell_idx))

    # Union of dofs if mixed BC/force actuators
    logger.info(f"Optimized DOFs in box (SensorPoint): {len(dofs_in_box)}")
    logger.info(
        f"Optimized DOFs in subdomain (SensorIntegral): {len(dofs_in_subdomain)}"
    )
    dofs_to_parse = dofs_in_box.union(dofs_in_subdomain)
    return dofs_to_parse


def get_C_dummy(W):
    mpi_size = dolfin.MPI.comm_world.Get_size()

    global_size = 10
    mpi_local_size = int(global_size / mpi_size)
    sensor_number = 2
    idof_list = [4, 6, 7, 8, 9]

    mpi_rank = dolfin.MPI.comm_world.Get_rank()
    if mpi_rank == 0:
        ownership_range = (0, mpi_local_size)
    if mpi_rank == 1:
        ownership_range = (mpi_local_size, global_size)

    logger.info(f"Process {mpi_rank} - Global size is: {global_size}")
    logger.info(f"Process {mpi_rank} - Local size is: {mpi_local_size}")

    C_local = np.zeros((sensor_number, mpi_local_size))

    uvp = dolfin.Function(W)
    uvp_vec = uvp.vector()

    for idof in idof_list:
        local_array = uvp_vec.get_local()
        local_array[:] = 0.0

        if idof in range(*ownership_range):
            idof_local = idof - ownership_range[0]
            local_array[idof_local] = 1.0

        uvp_vec.set_local(local_array)
        uvp_vec.apply("insert")

        result = my_fun(uvp, sensor_number=sensor_number)

        if idof in range(*ownership_range):
            C_local[:, idof_local] = result

    logger.info(f"Process {mpi_rank} - Finished filling C of size {C_local.shape}")

    return C_local


def my_fun(f, sensor_number):
    return np.ones(sensor_number) * f.vector().norm("l2")


####################################################################################
####################################################################################
