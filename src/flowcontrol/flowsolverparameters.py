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
    """Base class for all Param* dataclasses.

    Attributes
    ----------
    user_data :
        Arbitrary user-defined data (e.g. domain dimensions).  Provided to
        avoid dynamic attribute assignment at runtime.
    """

    user_data: dict = field(default_factory=dict)


@dataclass
class ParamFlow(ParamFlowSolver):
    """Flow configuration parameters.

    Parameters
    ----------
    Re :
        Reynolds number.
    uinf :
        Horizontal inlet velocity.  Defaults to 1.0.
    """

    Re: float
    uinf: float = 1.0


@dataclass
class ParamMesh(ParamFlowSolver):
    """Mesh configuration parameters.

    Parameters
    ----------
    meshpath :
        Path to the XDMF mesh file.
    """

    meshpath: Path


@dataclass
class ParamControl(ParamFlowSolver):
    """Control configuration parameters.

    Parameters
    ----------
    sensor_list :
        Ordered list of Sensor objects.
    actuator_list :
        Ordered list of Actuator objects.

    Attributes
    ----------
    sensor_number :
        Number of sensors (set automatically from ``sensor_list``).
    actuator_number :
        Number of actuators (set automatically from ``actuator_list``).
    """

    sensor_list: list[Sensor]
    sensor_number: int = field(init=False)

    actuator_list: list[Actuator]
    actuator_number: int = field(init=False)

    def __post_init__(self) -> None:
        self.sensor_number = len(self.sensor_list)
        self.actuator_number = len(self.actuator_list)


@dataclass
class ParamTime(ParamFlowSolver):
    """Time-stepping parameters.

    Parameters
    ----------
    num_steps :
        Total number of time steps.
    dt :
        Time step size.
    Tstart :
        Starting simulation time.

    Attributes
    ----------
    Tfinal :
        Final simulation time, computed as ``num_steps * dt``.
    """

    num_steps: int
    dt: float
    Tstart: float
    Tfinal: float = field(init=False)

    def __post_init__(self) -> None:
        self.Tfinal = self.num_steps * self.dt


@dataclass
class ParamRestart(ParamFlowSolver):
    """Parameters for restarting a simulation from a previous run.

    Parameters
    ----------
    save_every_old :
        ``save_every`` value used in the previous run (to locate checkpoint files).
    restart_order :
        Time-integration order to use at restart (1 or 2).
    dt_old :
        Time step used in the previous run.
    Trestartfrom :
        Simulation time from the previous run to restart from.
    """

    save_every_old: int = 0
    restart_order: int = 2
    dt_old: float = 0.0
    Trestartfrom: float = 0.0


@dataclass
class ParamSave(ParamFlowSolver):
    """Output and checkpointing parameters.

    Parameters
    ----------
    path_out :
        Directory for all output files.
    save_every :
        Write XDMF field snapshots every N iterations (0 = never).
    energy_every :
        Compute and log perturbation energy every N iterations
        (0 = never, 1 = every step).
    """

    path_out: Path
    save_every: int
    energy_every: int = 1


@dataclass
class ParamSolver(ParamFlowSolver):
    """Solver and equation parameters.

    Parameters
    ----------
    throw_error :
        If ``False``, swallow solver failures instead of raising.  Useful when
        FlowSolver is used as a backend in an optimization loop.
    shift :
        Adds ``-shift * ∫ u·v dx`` to the LHS (spectral shift for eigenvalue
        computation).
    is_eq_nonlinear :
        If ``False``, drop the nonlinear perturbation term ``(u·∇)u`` and
        simulate the linearized equations around the base flow.
    time_scheme :
        Time integration scheme: ``'bdf'`` (BDF1 → BDF2 ramp) or ``'cn'``
        (Crank-Nicolson).
    """

    throw_error: bool = True
    shift: float = 0.0
    is_eq_nonlinear: bool = True
    time_scheme: str = "bdf"  # "bdf" → BDF1 ramp-up to BDF2  |  "cn" → Crank-Nicolson


@dataclass
class ParamIC(ParamFlowSolver):
    """Initial condition parameters.

    Defines a divergence-free Gaussian perturbation (derivative of a Gaussian bell)
    centred at ``(xloc, yloc)`` with given ``radius`` and ``amplitude``.

    Parameters
    ----------
    xloc :
        x-coordinate of the perturbation centre.
    yloc :
        y-coordinate of the perturbation centre.
    radius :
        Radius of the Gaussian bell.
    amplitude :
        Peak amplitude of the perturbation.
    """

    xloc: float = 0.0
    yloc: float = 0.0
    radius: float = 0.0
    amplitude: float = 1.0
