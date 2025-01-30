from abc import ABC, abstractmethod
from enum import IntEnum
from dataclasses import dataclass
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
    actuator_type: ACTUATOR_TYPE
    expression: dolfin.Expression | None = None
    # only on (u,v) not p if type is force, for BC on syntax of DirichletBC

    @abstractmethod
    def load_expression(self, flowsolver) -> dolfin.Expression:
        pass


@dataclass(kw_only=True)
class ActuatorBCParabolicV(Actuator):
    """Cylinder-like actuator"""

    actuator_type: ACTUATOR_TYPE = ACTUATOR_TYPE.BC
    angular_size_deg: float

    def load_expression(self, flowsolver):
        L = (
            1
            / 2
            * flowsolver.params_flow.d
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
        return expression


@dataclass(kw_only=True)
class ActuatorForceGaussianV(Actuator):
    """Cavity-like actuator"""

    actuator_type: ACTUATOR_TYPE = ACTUATOR_TYPE.FORCE
    sigma: float
    position: np.ndarray

    def load_expression(self, flowsolver):
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
            u_ctrl=0.0,
        )

        expression.eta = 1 / dolfin.norm(expression, mesh=flowsolver.mesh)

        self.expression = expression
        return expression


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
