from dataclasses import dataclass, field
import dolfin


@dataclass
class FlowField:
    """Flow field for incompressible 2D flow"""

    u: dolfin.Function
    p: dolfin.Function
    up: dolfin.Function
    misc: dict = field(default_factory=dict)
