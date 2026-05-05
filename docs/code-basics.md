# Code Basics

## Conventions in the Code

- Before every method name, the `_` prefix is used whenever the method is not intended to be used outside of the body of the class.
- `U, P` (capital) refer to the full fields $U(x,t), P(x,t)$, while `u, p` (small) refer to the perturbation fields $u'(x,t), p'(x,t)$ (see [Numerical Details](numerical-details.md)). For boundary conditions, `BC, bc` follow the same convention, and more generally, all names referring to flow fields follow this convention.

## Basic Use

### 1️⃣ Choose a Use-Case: Inherit the `FlowSolver` Abstract Class

The simulation revolves around the abstract class `FlowSolver` that implements core features such as loading the mesh, defining the function spaces & trial/test functions, variational formulations, numerical schemes and solvers, handling the time-stepping and exporting fields and timeseries. The class is abstract as it does not implement a simulation case *per se*, but only provides utility for doing so. It features two abstract methods that are redefined for each use-case:

- `_make_boundaries` provides a definition and naming of the boundaries of the mesh in a pandas DataFrame:

```python
@abstractmethod
def _make_boundaries(self) -> pd.DataFrame:
    pass
```

The expected DataFrame has the following simple structure:

```python
boundaries_as_df = pandas.DataFrame(
    index=boundaries_names_as_list: list[str], 
    data={"subdomain": subdomains_as_list: list[dolfin.SubDomain]}
)
```

> **Warning:** For each new use-case, the user is expected to provide a mesh in `xdmf` format (see [Third-party Tools](third-party-tools.md)), that is compatible with their definition of boundaries.

- `_make_bcs` provides a description of the boundary conditions on the boundaries defined above, in a dedicated class `BoundaryConditions` containing two lists:

```python
@abstractmethod
def _make_bcs(self) -> BoundaryConditions:
    pass
```

`BoundaryConditions` is a utility class that contains two list fields: `bcu` (velocity boundary conditions for the perturbation field) and `bcp` (pressure boundary conditions for the perturbation field).

```python
@dataclass
class BoundaryConditions:
    bcu: list[dolfin.DirichletBC]
    bcp: list[dolfin.DirichletBC]
```

We give two examples with the code (the flow past a cylinder, and the flow over an open cavity) that inherit from `FlowSolver`: they are respectively `CylinderFlowSolver` and `CavityFlowSolver`.

### 2️⃣ Attach Sensors and Actuators to an Instance of a `FlowSolver` Subclass

In order to perform sensing and actuation (with the objective to close the loop), two dedicated abstract classes are proposed: `Sensor` and `Actuator`. Both these classes implement behaviors common to all sensors or actuators. They are not aimed at being instantiated directly; they need to be inherited first.

The sensors and actuators are attached to a `FlowSolver` object as lists, through the `ParamControl` dataclass (as `ParamControl.sensor_list, ParamControl.actuator_list`). By attaching several sensors or actuators, it is possible to generate Multiple-Input, Multiple-Output configurations for control. The call to `Sensor`s and `Actuator`s is made automatically by `FlowSolver`.

For the cylinder case, we give an example below. We create two actuators forcing boundary conditions (on the top and bottom poles of the cylinder, respectively), and three point probes at different locations in the wake. They are gathered in a `ParamControl` object, which is passed as an argument to initialize a `CylinderFlowSolver`:

```python
# Actuators
actuator_bc_1 = ActuatorBCParabolicV(angular_size_deg=10)
actuator_bc_2 = ActuatorBCParabolicV(angular_size_deg=10)
# Sensors
sensor_feedback = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3, 0]))
sensor_perf_1 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3.1, 1]))
sensor_perf_2 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3.1, -1]))
# Gather actuators and sensors in ParamControl object
params_control = flowsolverparameters.ParamControl(
    sensor_list=[sensor_feedback, sensor_perf_1, sensor_perf_2],
    actuator_list=[actuator_bc_1, actuator_bc_2],
)
```

### 3️⃣ Run a (Closed-Loop) Simulation

Once a use-case is defined by implementing the corresponding class inheriting `FlowSolver`, the basic feedback syntax has the following philosophy:

1. The `FlowSolver` subclass is instantiated with user-defined parameters
2. The base flow (stationary solution) is computed first
3. The object is prepared for time-stepping (e.g., we define operators, solvers, numerical schemes)
4. (Optional) A `Controller` is synthesized or read from a file
5. Time loop: iterate the `FlowSolver.step(u)` method, providing the 1D vector input `u` (open-loop or closed-loop using the `Controller` output)

A draft is given below. See the folder `examples` for more exhaustive code:

```python
# Instantiate and initialize FlowSolver object
fs = CylinderFlowSolver(...)
fs.compute_steady_state(...)
fs.initialize_time_stepping(...)

# Instantiate Controller (e.g. load from .mat file)
Kss = Controller.from_file(...)

# Time loop
y_meas = fs.y_meas
for _ in range(fs.params_time.num_steps):
    u_ctrl = Kss.step(y=-y_meas[0], dt=fs.params_time.dt)
    y_meas = fs.step(u_ctrl=u_ctrl)
```

The simulation should run seamlessly while providing information on the computed fields and potentially exporting information (as `xdmf` and `csv`).
