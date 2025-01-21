from dataclasses import dataclass
import dolfin


@dataclass
class FlowField:
    """Flow field for incompressible 2D flow"""

    u: dolfin.Function
    p: dolfin.Function
    up: dolfin.Function

    @staticmethod
    def generate(up):
        u, p = up.split(deepcopy=True)
        ff = FlowField(u=u, p=p, up=up)
        return ff


@dataclass
class FlowFieldCollection:
    # Base flow (full field)
    STEADY: FlowField | None = None
    U0: dolfin.Function | None = None
    P0: dolfin.Function | None = None
    UP0: dolfin.Function | None = None
    # Initial condition (pert field)
    ic: FlowField | None = None
    # Current and past solutions (pert field)
    u_: dolfin.Function | None = None
    p_: dolfin.Function | None = None
    up_: dolfin.Function | None = None
    u_n: dolfin.Function | None = None
    u_nn: dolfin.Function | None = None
    p_n: dolfin.Function | None = None
    # Saved fields (full field)
    Usave: dolfin.Function | None = None
    Psave: dolfin.Function | None = None
    Usave_n: dolfin.Function | None = None
