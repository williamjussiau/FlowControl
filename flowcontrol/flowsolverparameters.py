from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import actuator
import sensor


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


@dataclass(init=False)
class ParamControl:
    """Parameters related to control.

    Args:
        sensor_list (list): TODO
        sensor_number (int): TODO
        actuator_list (list): TODO
        actuator_number (int): TODO
    """

    sensor_list: list[sensor.Sensor]
    sensor_number: int

    actuator_list: list[actuator.Actuator]
    actuator_number: int

    def __init__(self, sensor_list=[], actuator_list=[]):
        self.sensor_list = sensor_list
        self.sensor_number = len(sensor_list)
        self.actuator_list = actuator_list
        self.actuator_number = len(actuator_list)


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
