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
