from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import numpy as np


class ACTUATOR_TYPE(Enum):
    BC = 1
    FORCE = 2


@dataclass
class ParamFlow:
    """Parameters related to flow configuration
    Include by default:
    Re
    """

    Re: float
    shift: float = 0.0
    is_eq_nonlinear: bool = True


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

    sensor_location: np.array
    # additional case specific sensor/actuator param


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

    num_steps: int
    dt: float
    Tstart: float = 0.0
    Trestartfrom: float = 0.0
    restart_order: int = 2
    dt_old: float = 0.0


@dataclass
class ParamSave:
    """Parameters related to saving
    Include by default:
    savedir0
    save_every
    save_every_old (for restarting)"""  # --> do Param_restart??

    path_out: Path
    save_every: int
    save_every_old: int


@dataclass
class ParamSolver:
    """Parameters related to solver issues
    But not really since init_pert is initial condition"""  # --> do Param_IC?

    throw_error: bool = True
    is_eq_nonlinear: bool = True
    ic_add_perturbation: float = 0.0


@dataclass
class ParamFlowSolver:
    """Class gathering all parameters"""

    param_flow: ParamFlow
    param_mesh: ParamMesh
    param_time: ParamTime
    param_solver: ParamSolver
    param_control: ParamControl
