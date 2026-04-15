from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import dolfin
import numpy as np
from numpy.typing import NDArray


class ACTUATOR_TYPE(IntEnum):
    """Enumeration of actuator types.

    Args:
        BC: boundary condition actuation
        FORCE: volumic force in momentum equation
    """

    BC = 1
    FORCE = 2


class CYLINDER_ACTUATION_MODE(IntEnum):
    """Modes for actuation on cylinders. This IntEnum is intended to be used
    by the user, it is never called by the original FlowSolver.

    Args:
        SUCTION: blowing & suction devices at the poles (ex. ActuatorBCParabolicV)
        ROTATION: rotation of whole cylinder (= ActuatorBCRotation)
    """

    SUCTION = 1
    ROTATION = 2


@dataclass(kw_only=True)
class Actuator(ABC):
    """Actuator abstract base class

    Args:
        actuator_type (ACTUATOR_TYPE): boundary condition or force actuation
        expression (dolfin.Expression): mathematical expression of the actuator profile
    """

    actuator_type: ACTUATOR_TYPE
    expression: Optional[dolfin.Expression] = None
    # only on (u,v) not p if type is force, for BC on syntax of DirichletBC

    @abstractmethod
    def _load_expression(
        self, V: dolfin.FunctionSpace, mesh: dolfin.Mesh
    ) -> dolfin.Expression:
        """Build the dolfin.Expression for this actuator.

        Implementations must not access the FlowSolver directly; all mesh
        and function-space information is passed explicitly.

        Args:
            V: velocity function space (used for ``ufl_element()``).
            mesh: computational mesh (needed by force actuators for norm computation).
        """
        pass

    def load_expression(self, flowsolver) -> dolfin.Expression:
        """Resolve the actuator expression against a live FlowSolver."""
        self.expression = self._load_expression(flowsolver.V, flowsolver.mesh)
        return self.expression


@dataclass(kw_only=True)
class ActuatorBC(Actuator):
    """Boundary condition actuator, inherits from abstract base class Actuator.

    The ``boundary_name`` field declares which subdomain (by string key in
    ``FlowSolver.boundaries``) this actuator sits on.  It is resolved to a
    ``dolfin.SubDomain`` automatically during ``load_expression``, so
    ``_make_bcs`` never needs to set ``actuator.boundary`` as a side effect.
    Warning: the ``boundary_name``, if provided, should match a boundary
    defined in the FlowSolver.

    ``boundary`` may still be set manually for backwards compatibility with
    subclasses that do not provide ``boundary_name`` (e.g. examples that set
    directly the actuator boundary in _make_bcs).

    Args:
        boundary_name: key into ``FlowSolver.boundaries`` (e.g. ``"actuator_up"``).
        boundary: resolved subdomain — set automatically from ``boundary_name``
            during ``load_expression``, or manually for legacy callers.
    """

    boundary_name: Optional[str] = None
    boundary: Optional[dolfin.SubDomain] = None

    def load_expression(self, flowsolver) -> dolfin.Expression:
        super().load_expression(flowsolver)
        if self.boundary_name is not None:
            try:
                self.boundary = flowsolver.get_subdomain(self.boundary_name)
            except KeyError:
                available = list(flowsolver.boundaries.index)
                raise KeyError(
                    f"Actuator boundary_name={self.boundary_name!r} not found in "
                    f"FlowSolver.boundaries. Available: {available}"
                ) from None
        return self.expression


@dataclass(kw_only=True)
class ActuatorBCParabolicV(ActuatorBC):
    """Cylinder actuator: parabolic profile depending on first spatial
    coordinate only. Usually located at the poles of a cylinder.
    The width of the actuator can be computed with the static method, given
    the radius of a cylinder and an angular size in degrees.
    """

    width: float = 0.0
    position_x: float = 0.0
    actuator_type: ACTUATOR_TYPE = ACTUATOR_TYPE.BC

    def _load_expression(self, V, mesh):
        expression = dolfin.Expression(
            [
                "0",
                "(x[0]-x0>=L || x[0]-x0<=-L) ? 0 : u_ctrl * -1*(x[0]-x0+L)*(x[0]-x0-L) / (L*L)",
            ],
            element=V.ufl_element(),
            L=self.width,
            x0=self.position_x,
            u_ctrl=0.0,
        )
        return expression

    @staticmethod
    def angular_size_deg_to_width(angular_size_deg, cylinder_radius):
        return cylinder_radius * np.sin(1 / 2 * angular_size_deg * dolfin.pi / 180)


@dataclass(kw_only=True)
class ActuatorBCRotation(ActuatorBC):
    """Cylinder actuator: tangential velocity at a radius D around a center (x0, y0).
    Typically used to model a rotating cylinder.
    """

    position_x: float = 0.0
    position_y: float = 0.0
    diameter: float = 1.0
    actuator_type: ACTUATOR_TYPE = ACTUATOR_TYPE.BC

    def _load_expression(self, V, mesh):
        expression = dolfin.Expression(
            [
                "-sin(atan2(x[1]-y0,x[0]-x0))*u_ctrl*d/2",
                "cos(atan2(x[1]-y0,x[0]-x0))*u_ctrl*d/2",
            ],
            element=V.ufl_element(),
            y0=self.position_y,
            x0=self.position_x,
            u_ctrl=0.0,
            d=self.diameter,
        )
        return expression


@dataclass(kw_only=True)
class ActuatorBCUniformU(ActuatorBC):
    """Lid-driven cavity actuator: uniform profile on u,
    located on top boundary.
    This Actuator has type ACTUATOR_TYPE.BC, which means it is closely linked
    to the definition of boundaries (i.e. FlowSolver._make_boundaries() and
    FlowSolver._make_bcs())"""

    actuator_type: ACTUATOR_TYPE = ACTUATOR_TYPE.BC

    def _load_expression(self, V, mesh):
        expression = dolfin.Expression(
            [
                "u_ctrl",
                "0",
            ],
            element=V.ufl_element(),
            u_ctrl=0.0,
        )
        return expression


@dataclass(kw_only=True)
class ActuatorForceGaussianV(Actuator):
    """Cavity actuator: volumic force with Gaussian profile acting on
    the second component of the velocity, centered at custom position.
    This Actuator has type ACTUATOR_TYPE.FORCE, so it is taken into account
    automatically when building equations (in FlowSolver._make_varfs())."""

    sigma: float
    position: NDArray[np.float64]
    actuator_type: ACTUATOR_TYPE = ACTUATOR_TYPE.FORCE

    def _load_expression(self, V, mesh):
        expression = dolfin.Expression(
            [
                "0",
                "u_ctrl * eta*exp(-0.5*((x[0]-x10)*(x[0]-x10)+(x[1]-x20)*(x[1]-x20))/(sig*sig))",
            ],
            element=V.ufl_element(),
            eta=1,
            sig=self.sigma,
            x10=self.position[0],
            x20=self.position[1],
            u_ctrl=1.0,
        )

        BtB = dolfin.norm(expression, mesh=mesh)
        expression.eta = 1 / BtB
        expression.u_ctrl = 0.0
        return expression


if __name__ == "__main__":
    print("-" * 10)
    try:
        actuator = Actuator(ACTUATOR_TYPE=ACTUATOR_TYPE.BC)  # type: ignore
    except TypeError as e:
        print(e)

    actuator_cylinder = ActuatorBCParabolicV(
        width=ActuatorBCParabolicV.angular_size_deg_to_width(
            angular_size_deg=10, cylinder_radius=3
        ),
        position_x=1.3,
    )
    actuator_cavity = ActuatorForceGaussianV(
        sigma=0.00849, position=np.array([-0.1, 0.02])
    )

    print("-" * 10)
    print(actuator_cylinder)
    print("-" * 10)
    print(actuator_cavity)
    print("-" * 10)
    print("-" * 10)
