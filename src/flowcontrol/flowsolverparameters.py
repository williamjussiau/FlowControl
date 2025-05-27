from dataclasses import dataclass, field
from pathlib import Path

from flowcontrol.actuator import Actuator
from flowcontrol.sensor import Sensor


@dataclass(kw_only=True)
class ParamFlowSolver:
    """Base class for Param* classes. It includes a dedicated field
    for user parameters, in order to avoid dynamically adding
    parameters at runtime.

    Args:
        user_data (dict): user-defined data, e.g. domain limits..."""

    user_data: dict = field(default_factory=dict)


@dataclass
class ParamFlow(ParamFlowSolver):
    """Parameters related to flow configuration.

    Args:
        uinf (float): horizontal velocity at inlet
        Re (float): Reynolds number
    """

    Re: float
    uinf: float = 1.0


@dataclass
class ParamMesh(ParamFlowSolver):
    """Parameters related to flow configuration.

    Args:
        meshpath (pathlib.Path): path to xdmf mesh file
    """

    meshpath: Path


@dataclass(init=False)
class ParamControl(ParamFlowSolver):
    """Parameters related to control.

    Args:
        sensor_list (list): list of Sensor objects
        sensor_number (int): number of sensors (auto)
        actuator_list (list): list of Actuator objects
        actuator_number (int): number of actuators (auto)
    """

    sensor_list: list[Sensor]
    sensor_number: int

    actuator_list: list[Actuator]
    actuator_number: int

    def __init__(self, sensor_list=[], actuator_list=[], user_data={}):
        self.sensor_list = sensor_list
        self.sensor_number = len(sensor_list)
        self.actuator_list = actuator_list
        self.actuator_number = len(actuator_list)
        self.user_data = user_data


@dataclass(init=False)
class ParamTime(ParamFlowSolver):
    """Parameters related to time-stepping.

    Args:
        num_steps (int): number of steps
        dt (float): time step
        Tstart (float): starting simulation time
        Tfinal (float): final simulation time (computed automatically)
    """

    num_steps: int
    dt: float
    Tstart: float
    Tfinal: float

    def __init__(self, num_steps, dt, Tstart=0.0, user_data={}):
        self.num_steps = num_steps
        self.dt = dt
        self.Tstart = Tstart
        self.Tfinal = num_steps * dt
        self.user_data = user_data


@dataclass
class ParamRestart(ParamFlowSolver):
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
class ParamSave(ParamFlowSolver):
    """Parameters related to saving fields and time series.

    Args:
        path_out (pathlib.Path): folder for saving files
        save_every (int): export files every _save_every_ iteration
    """

    path_out: Path
    save_every: int


@dataclass
class ParamSolver(ParamFlowSolver):
    """Parameters related to equations and solvers.

    Args:
        throw_error (bool): if False, does not catch error when solver fails.
            This may be useful when using FlowSolver as a backend for an optimization tool.
        shift (float): shift equations by -_shift_*int(u * v * dx)
        is_eq_nonlinear (bool): if False, simulate equations linearized around base flow (i.e. the
            nonlinear term for the perturbation: (u.div)u, is neglected)
    """

    throw_error: bool = True
    shift: float = 0.0
    is_eq_nonlinear: bool = True


@dataclass
class ParamIC(ParamFlowSolver):
    """Parameters for Initial Condition (IC). By default, derivative of Gaussian bell
    (therefore divergence-free) with given position (xloc, yloc), radius and amplitude.

    Args:
        xloc (float): x position of center
        yloc (float): y position of center
        radius (float): radius of IC
        amplitude (float): amplitude of IC
    """

    xloc: float = 0.0
    yloc: float = 0.0
    radius: float = 0.0
    amplitude: float = 1.0
    amplitude: float = 1.0
