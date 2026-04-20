"""Example FlowSolver subclasses and run scripts for four flow configurations.

Subpackages
-----------
cavity     : open-cavity flow (Re~7500) — CavityFlowSolver, Gaussian force actuator
cylinder   : flow past a cylinder (Re~100) — CylinderFlowSolver, parabolic BC actuators
lidcavity  : lid-driven cavity (Re~8000) — LidCavityFlowSolver, uniform lid actuator
pinball    : fluidic pinball / 3 cylinders (Re~100) — PinballFlowSolver,
             suction (ActuatorBCParabolicV) or rotation (ActuatorBCRotation) actuation
"""
