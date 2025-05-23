from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum

import dolfin
import numpy as np


class ACTUATOR_TYPE(IntEnum):
    """Enumeration of actuator types

    Args:
        BC: boundary condition actuation
        FORCE: volumic force in momentum equation
    """

    BC = 1
    FORCE = 2


@dataclass(kw_only=True)
class Actuator(ABC):
    """Actuator abstract base class

    Args:
        actuator_type (ACTUATOR_TYPE): boundary condition or force actuation
        expression (dolfin.Expression): mathematical expression of the actuator profile
    """

    actuator_type: ACTUATOR_TYPE
    expression: dolfin.Expression | None = None
    # only on (u,v) not p if type is force, for BC on syntax of DirichletBC

    @abstractmethod
    def load_expression(self, flowsolver) -> dolfin.Expression:
        """Load actuator expression projected on flowsolver function spaces. The reason
        behind this method is to be able to instantiate an Actuator independently from a
        FlowSolver object; then to be able to attach the first to the second and load the
        dolfin.Expression (that needs FlowSolver to be evaluated).

        Args:
            flowsolver (FlowSolver): FlowSolver that is using the actuator
        """
        pass

@dataclass(kw_only=True)
class ActuatorBC(Actuator):
    """_summary_

    Args:
        Actuator (_type_): _description_
    """

    boundary: None | dolfin.Expression = None


@dataclass(kw_only=True)
class ActuatorBCParabolicV(ActuatorBC):
    """Cylinder actuator: parabolic profile depending on first spatial
    coordinate only, located at the poles of the cylinder.
    This Actuator has type ACTUATOR_TYPE.BC, which means it is closely linked
    to the definition of boundaries (i.e. FlowSolver._make_boundaries() and
    FlowSolver._make_bcs())"""

    angular_size_deg: float
    actuator_type: ACTUATOR_TYPE = ACTUATOR_TYPE.BC

    def load_expression(self, flowsolver):
        L = (
            1
            / 2
            * flowsolver.params_flow.user_data["D"]
            * np.sin(1 / 2 * self.angular_size_deg * dolfin.pi / 180)
        )
        expression = dolfin.Expression(
            [
                "0",
                "(x[0]>=L || x[0] <=-L) ? 0 : u_ctrl * -1*(x[0]+L)*(x[0]-L) / (L*L)",
            ],
            element=flowsolver.V.ufl_element(),
            L=L,
            u_ctrl=0.0,
        )

        self.expression = expression


@dataclass(kw_only=True)
class ActuatorForceGaussianV(Actuator):
    """Cavity actuator: volumic force with Gaussian profile acting on
    the second component of the velocity, centered at custom position.
    This Actuator has type ACTUATOR_TYPE.FORCE, so it is taken into account
    automatically when building equations (in FlowSolver._make_varfs())."""

    sigma: float
    position: np.ndarray
    actuator_type: ACTUATOR_TYPE = ACTUATOR_TYPE.FORCE

    def load_expression(self, flowsolver):
        # expr_u = "0"
        # expr_v = "u_ctrl * eta*exp(-0.5*((x[0]-x10)*(x[0]-x10)+(x[1]-x20)*(x[1]-x20))/(sig*sig))"
        # user_parameters = {"sigma": self.sigma, "x10": self.position[0], "x20": self.position[1]}
        # expression = VectorExpression2D(expr_u, expr_v, degree=2, user_parameters=user_parameters, u_ctrl=1.0)
        expression = dolfin.Expression(
            [
                "0",
                "u_ctrl * eta*exp(-0.5*((x[0]-x10)*(x[0]-x10)+(x[1]-x20)*(x[1]-x20))/(sig*sig))",
            ],
            element=flowsolver.V.ufl_element(),
            eta=1,
            sig=self.sigma,
            x10=self.position[0],
            x20=self.position[1],
            u_ctrl=1.0,
        )

        BtB = dolfin.norm(expression, mesh=flowsolver.mesh)
        expression.eta = 1 / BtB
        expression.u_ctrl = 0.0
        self.expression = expression


# class VectorExpression2D:
#     def __init__(self, expr_u: str, expr_v: str, degree: int = 2, user_parameters: dict = None):
#         self.expr_u = expr_u
#         self.expr_v = expr_v
#         self.degree = degree
#         self.user_parameters = user_parameters or {}

#         self.expr = dolfin.Expression((expr_u, expr_v), degree=degree, **self.user_parameters)


if __name__ == "__main__":
    print("-" * 10)
    try:
        actuator = Actuator()
    except Exception as e:
        print(e)

    actuator_cylinder = ActuatorBCParabolicV(angular_size_deg=10)
    actuator_cavity = ActuatorForceGaussianV(
        sigma=0.00849, position=np.array([-0.1, 0.02])
    )

    print("-" * 10)
    print(actuator_cylinder)
    print("-" * 10)
    print(actuator_cavity)
    print("-" * 10)
    print("-" * 10)
