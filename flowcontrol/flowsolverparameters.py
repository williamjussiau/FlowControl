from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
import numpy as np


class ACTUATOR_TYPE(IntEnum):
    """Enumeration of actuator types

    Args:
        BC: boundary condition actuation
        FORCE: volumic force in momentum equation
    """

    BC = 1
    FORCE = 2


class SENSOR_TYPE(IntEnum):
    """Enumeration of sensor types

    Args:
        U: sensor measuring 1st velocity component
        V: sensor measuring 2nd velocity component
        P: sensor measuring pressure
        OTHER: sensor measuring something else (e.g. integral)
    """

    U = 0
    V = 1
    P = 2
    OTHER = 5


@dataclass
class ParamFlow:
    """Parameters related to flow configuration.

    Args:
        Re (float): Reynolds number
    """

    Re: float


@dataclass
class ParamMesh:
    """Parameters related to flow configuration.

    Args:
        meshpath (pathlib.Path): path to xdmf mesh file
    """

    meshpath: Path


@dataclass
class ParamControl:
    """Parameters related to control.

    Args:
        sensor_location (float): TODO
        sensor_number (float): TODO
        sensor_type (float): TODO
        actuator_location (float): TODO
        actuator_number (float): TODO
        actuator_type (float): TODO
        actuator_parameters (float): TODO
    """

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
    """Parameters related to time-stepping.

    Args:
        num_steps (int): number of steps
        dt (float): time step
        Tstart (float): starting simulation time
        Tfinal (float): final simulation time (computed automatically).
    """

    def __init__(self, num_steps, dt, Tstart=0.0):
        self.num_steps = num_steps
        self.dt = dt
        self.Tstart = Tstart
        self.Tfinal = num_steps * dt


@dataclass
class ParamRestart:
    """Parameters related to restarting a simulation.

    Args:
        save_every_old (int): previous save_every (see ParamSave)
        restart_order (int): equation order for restarting
        dt_old (float): previous time step
        Trestartfrom (float): starting time from the previous simulation
            (for finding the corresponding field files).
    """

    save_every_old: int = 0
    restart_order: int = 2
    dt_old: float = 0.0
    Trestartfrom: float = 0.0


@dataclass
class ParamSave:
    """Parameters related to saving fields and time series.

    Args:
        path_out (pathlib.Path): folder for saving files
        save_every (int): export files every _save_every_ iteration
    """

    path_out: Path
    save_every: int


@dataclass
class ParamSolver:
    """Parameters related to equations and solvers.

    Args:
        throw_error (bool): if False, does not catch error when solver fails.
            This may be useful when using FlowSolver as a backend for an optimization tool.
        ic_add_perturbation (float): amplitude of perturbation added to given initial condition.
        shift (float): shift equations by -_shift_*int(u * v * dx)
        is_eq_nonlinear (bool): if False, simulate equations linearized around base flow (i.e. the
            nonlinear term for the perturbation: (u.div)u, is neglected)
    """

    throw_error: bool = True
    ic_add_perturbation: float = 0.0
    shift: float = 0.0
    is_eq_nonlinear: bool = True
