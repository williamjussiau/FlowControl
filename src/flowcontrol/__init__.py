"""FlowControl — flow simulation and feedback control with FEniCS/dolfin.

Main classes
------------
FlowSolver        : abstract base for CFD simulations (flowsolver.py)
Controller        : continuous-time state-space controller with ZOH stepping (controller.py)
NSForms           : UFL variational forms for Navier-Stokes (nsforms.py)
FlowExporter      : field and timeseries I/O (exporter.py)
SteadyStateSolver : Newton/Picard steady-state iteration (steadystate.py)
Actuator          : actuator class hierarchy (actuator.py)
Sensor            : sensor class hierarchy (sensor.py)
"""

__version__ = "0.1.0"

from flowcontrol.actuator import (
    ACTUATOR_TYPE,
    Actuator,
    ActuatorBC,
    ActuatorBCParabolicV,
    ActuatorBCRotation,
    ActuatorBCUniformU,
    ActuatorForceGaussianV,
)
from flowcontrol.controller import Controller
from flowcontrol.exporter import FlowExporter
from flowcontrol.flowfield import BoundaryConditions, FlowField, FlowFieldCollection, SimPaths
from flowcontrol.flowsolver import FlowSolver
from flowcontrol.flowsolverparameters import (
    ParamControl,
    ParamFlow,
    ParamIC,
    ParamMesh,
    ParamRestart,
    ParamSave,
    ParamSolver,
    ParamTime,
)
from flowcontrol.nsforms import NSForms
from flowcontrol.sensor import (
    SENSOR_TYPE,
    Sensor,
    SensorHorizontalWallShear,
    SensorIntegral,
    SensorPoint,
)
from flowcontrol.steadystate import SteadyStateSolver

__all__ = [
    "__version__",
    # Main classes
    "FlowSolver",
    "Controller",
    "NSForms",
    "FlowExporter",
    "SteadyStateSolver",
    # Flow field
    "FlowField",
    "FlowFieldCollection",
    "BoundaryConditions",
    "SimPaths",
    # Parameters
    "ParamFlow",
    "ParamTime",
    "ParamSave",
    "ParamSolver",
    "ParamMesh",
    "ParamControl",
    "ParamIC",
    "ParamRestart",
    # Actuators
    "Actuator",
    "ActuatorBC",
    "ActuatorBCParabolicV",
    "ActuatorBCRotation",
    "ActuatorBCUniformU",
    "ActuatorForceGaussianV",
    "ACTUATOR_TYPE",
    # Sensors
    "Sensor",
    "SensorPoint",
    "SensorIntegral",
    "SensorHorizontalWallShear",
    "SENSOR_TYPE",
]
