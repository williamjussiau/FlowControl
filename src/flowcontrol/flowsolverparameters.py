"""Parameter dataclasses for FlowSolver configuration.

Each Param* class covers one aspect of the simulation.  All inherit from
ParamFlowSolver, which provides a user_data dict for ad-hoc fields.

Classes
-------
ParamFlowSolver : base class (user_data dict)
ParamFlow       : Reynolds number and inlet velocity
ParamMesh       : mesh file path
ParamControl    : actuator and sensor lists (sensor_number/actuator_number auto-computed)
ParamTime       : time-stepping config (num_steps, dt, Tstart; Tfinal auto-computed)
ParamRestart    : legacy restart info (old dt, save_every, Trestartfrom)
ParamSave       : output directory, XDMF save frequency, energy logging frequency
ParamSolver     : solver options (linearity, time scheme, spectral shift, error handling)
ParamIC         : initial perturbation (Gaussian position, radius, amplitude)
"""
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


@dataclass
class ParamControl(ParamFlowSolver):
    """Parameters related to control.

    Args:
        sensor_list (list): list of Sensor objects
        sensor_number (int): number of sensors (auto)
        actuator_list (list): list of Actuator objects
        actuator_number (int): number of actuators (auto)
    """

    sensor_list: list[Sensor]
    sensor_number: int = field(init=False)

    actuator_list: list[Actuator]
    actuator_number: int = field(init=False)

    def __post_init__(self):
        self.sensor_number = len(self.sensor_list)
        self.actuator_number = len(self.actuator_list)


@dataclass
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
    Tfinal: float = field(init=False)

    def __post_init__(self):
        self.Tfinal = self.num_steps * self.dt


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
        save_every (int): export XDMF fields every N iterations (0 = never)
        energy_every (int): compute and log perturbation energy every N iterations
            (0 = never, 1 = every step).
    """

    path_out: Path
    save_every: int
    energy_every: int = 1


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
    time_scheme: str = "bdf"  # "bdf" → BDF1 ramp-up to BDF2  |  "cn" → Crank-Nicolson


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
