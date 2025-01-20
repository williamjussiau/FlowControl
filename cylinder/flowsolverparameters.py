from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
import numpy as np


class ACTUATOR_TYPE(IntEnum):
    BC = 1
    FORCE = 2


class SENSOR_TYPE(IntEnum):
    U = 0
    V = 1
    P = 2
    OTHER = 5


@dataclass
class ParamFlow:
    """Parameters related to flow configuration
    Include by default:
    Re
    """

    Re: float


@dataclass
class ParamMesh:
    """Parameters related to flow configuration
    Include by default:
    meshpath
    meshname
    Could include:
    limits of domain
    """

    meshpath: Path
    # additional case-specific limits


@dataclass
class ParamControl:
    """Parameters related to control
    Include by default:
    sensor_location
    Could include:
    . case-specific sensor parameters
    . actuator parameters"""

    sensor_location: np.ndarray[int, float] = field(
        default_factory=lambda: np.empty(shape=(0, 2), dtype=float)
    )
    sensor_number: int = 0
    sensor_type: list[SENSOR_TYPE] = field(default_factory=list)
    sensor_parameters: dict = field(default_factory=dict)

    actuator_location: np.ndarray[int, float] = field(
        default_factory=lambda: np.empty(shape=(0, 2), dtype=float)
    )
    actuator_number: int = 0
    actuator_type: list[ACTUATOR_TYPE] = field(default_factory=list)
    actuator_parameters: dict = field(default_factory=dict)


@dataclass
class ParamTime:
    """Parameters related to time-stepping
    Include by default:
    num_steps
    dt
    Tstart
    Trestartfrom
    restart_order
    """

    def __init__(self, num_steps, dt, Tstart=0.0):
        self.num_steps = num_steps
        self.dt = dt
        self.Tstart = Tstart
        self.Tfinal = num_steps * dt


@dataclass
class ParamRestart:
    """For restart only?"""

    save_every_old: int = 0
    restart_order: int = 2
    dt_old: float = 0.0
    Trestartfrom: float = 0.0


@dataclass
class ParamSave:
    """Parameters related to saving
    Include by default:
    savedir0
    save_every
    save_every_old (for restarting)"""  # --> do Param_restart??

    path_out: Path
    save_every: int


@dataclass
class ParamSolver:
    """Parameters related to solver issues
    But not really since init_pert is initial condition"""  # --> do Param_IC?

    throw_error: bool = True
    ic_add_perturbation: float = 0.0
    shift: float = 0.0
    is_eq_nonlinear: bool = True


@dataclass
class ParamFlowSolver:
    """Class gathering all parameters"""

    param_flow: ParamFlow
    param_mesh: ParamMesh
    param_time: ParamTime
    param_solver: ParamSolver
    param_control: ParamControl
