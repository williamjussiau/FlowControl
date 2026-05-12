## Conventions in the code
* Before every method name, the `_` prefix is used whenever the method is not intended to be used outside of the body of the class.
* `U, P` (capital) refer to the full fields $U(x,t), P(x,t)$, while `u, p` (small) refer to the perturbation fields $u'(x,t), p'(x,t)$ (see [Numerical Details](numerical-details.md)). For boundary conditions, `BC, bc` follow the same convention, and more generally, all names refering to flow fields is following the convention.


## Advanced uses

### Initializations

---
#### Parameter classes
A handful of parameters are embedded in dataclasses prefixed with `Param*`, defined in the file `flowsolverparameters.py`: 
```py
ParamFlow
ParamMesh
ParamControl
ParamTime
ParamRestart
ParamSave
ParamSolver
ParamIC
``` 
All these dataclasses contain parameters used natively by `FlowSolver`: flow parameters ($Re$...), start time and time step for simulation, save frequency, paths... Some but not all fields have default values, but the instantiation of these parameters is usually straightforward.

In addition, they inherit from the base class `ParamFlowSolver` which embeds a dictionary of `user_data`. This dictionary is intended to be used for data unknown to `FlowSolver` (the latter never calls this dictionary), but by its subclasses. In other words, only the user is supposed to call the `user_data` field when inheriting from `FlowSolver`. For example, `ParamMesh.user_data` could contain the mesh extent or specific locations used to define boundaries/boundary conditions.

---
#### Mesh
A mesh should be provided by the user in `xdmf` format. It should be compatible with the definition of boundaries and boundary conditions in the `_make_boundaries, _make_bcs` methods overriden by the user. The path to the mesh is embedded in the dataclass `ParamMesh` as a `ParamMesh.meshpath: pathlib.Path`.


---
#### Initialization of a `FlowSolver`
In order to instantiate a `FlowSolver`, all parameters classes should be instantiated (see above) and passed as parameters.

When instantiating a `FlowSolver`, the timeline of internal methods called is the following:
```py
self.paths = self._define_paths()
self.mesh = self._make_mesh()
self.V, self.P, self.W = self._make_function_spaces()
self.boundaries = self._make_boundaries()  # @abstract
self._mark_boundaries()
self._load_actuators()
self._load_sensors()
self.bc = self._make_bcs()  # @abstract
```


---
#### Initialize time-stepping
Before launching a time simulation with a time loop, some operations are performed onto a `FlowSolver` object by calling `FlowSolver.initialize_time_stepping(Tstart, ic)`. This method can work in two distinct ways:
* If `Tstart=0`, then the parameter `ic` is taken into account. It corresponds to the initial condition of the perturbation field on the base flow. The default perturbation field is defined in the following method:
```py
_default_initial_perturbation(self, xloc: float = 0.0, yloc: float = 0.0, radius: float = 1.0) -> dolfin.Function
```
Its parameters may be tweaked from outside with the `ParamIC` dataclass, as follows:
```py
params_ic = flowsolverparameters.ParamIC(
    xloc=2.0, yloc=0.0, radius=0.5, amplitude=1.0
)
```

* If `Tstart!=0`, then the code will try to restart a simulation from a saved file corresponding to the prescribed `Tstart`. In this case, it is important that the parameter `ParamRestart` is set correctly, in order to find the snapshot corresponding to `Tstart` (because FEniCS is working with index-based snapshots instead of time-based snapshots, which means a computation is required to retrieve the index from the time instant). More details on the saving system and the restarting procedure are given below, in a dedicated section.



---
#### Base flow computation
Good practice is to do Picard iterations, then Newton

Default initial guess (used when none is provided):
```py
_default_steady_state_initial_guess(self) -> dolfin.UserExpression
```


---
#### Inlet flow profile
By default, the inlet flow profile is uniform, with velocity $(U_\infty, V_\infty) = (Uinf, 0)$ where Uinf is `ParamFlow.uinf`. This default profile may be modified in the method `make_BCs` with a `dolfin.Expression`. 

The perturbation velocity boundary condition on this boundary is always $(0, 0)$.


---
#### `FlowField` and `FlowFieldCollection`
Some helper classes were defined to embed flow fields more easily, especially because FEniCS might sometimes need the velocity and pressure fields separately, or merged into a single object. The dataclass `FlowField` provides this utility: it contains a velocity field `u`, a pressure field `p` and the merged field `up`, all as `dolfin.Function`s.

In order to split a `up` field into the corresponding `u, p` fields, the method `dolfin.Function.split` may be used. In order to reverse the operation and merge `u, p` into a single field `up`, the method `FlowSolver.merge` is advised.

In addition, an object `FlowSolver` holds a `FlowFieldCollection` dataclass to gather all fields into a single structure for easier access. `FlowFieldCollection` contains lots of types of fields, among which the base flow, the initial perturbation, the current and previous perturbations... By default, the collection can be accessed through the attribute `FlowSolver.fields`.






---
## Closing the loop: time-stepping, actuation, sensing and controllers

---
### Time-stepping a FlowSolver
The `FlowSolver` class is intended to be seen, among other things, as capable of performing time-stepping using a series of inputs `u_ctrl`, providing a series of outputs `y_meas`. The time simulation is done one time step `dt` at a time, by using the `FlowSolver.step(u_ctrl)` method. The `step` method may be iterated in a loop in order to perform long simulations.

The input `u_ctrl` may be defined by hand or stem from a controller in closed-loop (see below), some examples are found in the `examples/` folder.


---
### Actuators

1. Principle
`Actuator` is an abstract class that encapsulates a `dolfin.Expression` and other parameters. An actuator are passed as a parameter to a `FlowSolver` for instantiation, through an `actuator_list` in the `ParamControl` object. 

Actuator have an assigned type, defined as an integer enumeration: `ACTUATOR_TYPE(IntEnum)`. It may be one of the following:
* `ACTUATOR_TYPE.FORCE`: the actuator provides a volumic forcing. Its expression is automatically included in the momentum equation (in variational form).
* `ACTUATOR_TYPE.BC`: the actuator modifies the boundary conditions dynamically. It should be reflected by the user when overriding `_make_boundaries(), _make_bcs()`. An example can be found in `examples/cylinder/cylinderflowsolver.py`:
```py
def _make_bcs(self):
    ...
    bcu_actuation_up = dolfin.DirichletBC(
        self.W.sub(0),
        self.params_control.actuator_list[0].expression,
        self.get_subdomain["actuator_up"],
        )
    bcu_actuation_lo = dolfin.DirichletBC(
        self.W.sub(0),
        self.params_control.actuator_list[1].expression,
        self.get_subdomain["actuator_lo"],
        )
    ...
    return BoundaryConditions(bcu=bcu, bcp=[])
```

The expression of each actuator needs to be loaded after the `FlowSolver` is instantiated (the analytic `dolfin.Expression` is projected onto the FEM function spaces), which is handled automatically by the code. 




2. Examples of actuators
* `ActuatorBCParabolicV`: boundary condition actuator, 2nd component on velocity has parabolic profile

_Mathematical expression:_

$${v_{act}}({x}, t) =  -  \dfrac{(x_1-l)(x_1+l)}{l^2} u(t)$$, with $l = \frac{1}{2} D  \sin  \left( \frac{\delta}{2} \right)$ and $\delta$ is the tunable actuator opening in degrees.

_FEniCS syntax:_
```py
def load_expression(self, flowsolver):
    L = (
        1
        / 2
        * flowsolver.params_flow.user_data["D"]
        * np.sin(1 / 2 * self.angular_size_deg * dolfin.pi / 180)
    )
    expression = dolfin.Expression(
        [
            "0",
            "(x[0]>=L || x[0] <=-L) ? 0 : u_ctrl * -1*(x[0]+L)*(x[0]-L) / (L*L)",
        ],
        element=flowsolver.V.ufl_element(),
        L=L,
        u_ctrl=0.0,
    )

    self.expression = expression
```


* `ActuatorForceGaussianV`: force actuator, gaussian-shaped on the 2nd component of velocity

_Mathematical expression:_

$$B({x})u(t)=\left[ 0, \eta \exp\left( \frac{\left(x_1 - x_1^0\right)^2 + \left(x_2 - x_2^0\right)^2}{2\sigma_0^2}  \right)\right]^T u(t)$$ with $\eta$ such that $\int_\Omega B({x})^T B({x}) d\Omega = 1$.

_FEniCS syntax:_
```py
def load_expression(self, flowsolver):
    expression = dolfin.Expression(
        [
            "0",
            "u_ctrl * eta*exp(-0.5*((x[0]-x10)*(x[0]-x10)+(x[1]-x20)*(x[1]-x20))/(sig*sig))",
        ],
        element=flowsolver.V.ufl_element(),
        eta=1,
        sig=self.sigma,
        x10=self.position[0],
        x20=self.position[1],
        u_ctrl=1.0,
    )

    BtB = dolfin.norm(expression, mesh=flowsolver.mesh)
    expression.eta = 1 / BtB
    expression.u_ctrl = 0.0
    self.expression = expression
```



3. Define new actuators
One can readily define a new actuator by inheriting the base class `Actuator` and overriding the abstract method `_load_expression`:

```py
def _load_expression(self, V: dolfin.FunctionSpace, mesh: dolfin.Mesh) -> dolfin.Expression:
```

:warning: Do not forget
* Include a `u_ctrl` field in the `dolfin.Expression`. Its value may be changed in the body of the method (see `ActuatorForceGaussianV`), but it should be `0.0` when the method exits.
* Return the expression at the end of `_load_expression`.





---
### Sensors



1. Principle

`Sensor` is an abstract class that gathers a behavior common to all sensors: it exhibits an abstract method `eval(self, up: dolfin.Function) -> float` to evaluate the measurement on a mixed-field `(u,p)`. 

The `Sensor` abstract class is expecte to be inherited by specific kinds of sensors. For example, the classes `SensorPoint` (point probe) and `SensorHorizontalWallShear` (integration on a subdomain) are subclasses that implement the `Sensor.eval()` abstract method. 

The evaluation of sensors is handled automatically by the `FlowSolver` in the `step()` method.

Some sensors, for example those inheriting from `SensorIntegral` (e.g. `SensorHorizontalWallShear`) need to be loaded in some way (e.g. to define a subdomain of integration). They implement a `load()` method that is called by `FlowSolver` if the boolean `Sensor.require_loading` is set to `True`.

Just like actuators, sensors hold a `SENSOR_TYPE(IntEnum)`, but it serves a different purpose. The `SENSOR_TYPE.U, SENSOR_TYPE.V, SENSOR_TYPE.P, SENSOR_TYPE.OTHER` is merely a shortcut to evaluate point probes on the right component of the field.




2. Examples of sensors

* A simple `SensorPoint(Sensor)` has a straightforward definition of its `eval()` method: it evaluates the field at the given position (self.position) and on the given component (self.sensor_type):

_Mathematical expression:_ 

$y(t) = u_1(x_s, t)$ or $y(t) = u_2(x_s, t)$ or $y(t) = p(x_s, t)$ where $x_s$ is the sensor location.

_FEniCS syntax:_
```py
def eval(self, up: dolfin.Function) -> float:
    return peval(up, dolfin.Point(self.position))[self.sensor_type]
```

> **Note:** Direct point evaluation `up(x, y)` is not MPI-safe. Always use `peval` from `utils.mpi` for point probes.

* For a `SensorHorizontalWallShear(SensorIntegral)` (where `SensorIntegral` inherits directly from `Sensor`), the definition of the `eval()` method is more complex. First of all, the `load()` function defines a subdomain of integration with a given `index: int` and an associated `ds: dolfin.Measure`. The `eval()` method integrates (`assemble`) on the sensor subdomain (`self.ds(int(self.sensor_index))`) the quantity $\frac{\partial u_1}{\partial x_2}$.

_Mathematical expression:_

$y(t) = \int_{x \in S} \frac{\partial u_1}{\partial x_2} dx$

_FEniCS syntax:_
```py
def eval(self, up):
    return dolfin.assemble(up.dx(1)[0] * self.ds(int(self.sensor_index)))
```



3. Define new sensors

New sensors can be defined by inheriting existing classes.

:warning: The user is responsible for the compatibility of their `Sensor.eval()`code with parallel execution (MPI) of the code.



---
### Controller
The class `Controller` aims at implementing a LTI system used as a controller in a closed-loop. It inherits from `control.StateSpace` (LTI system) while encapsulating two additional attributes:
* The current plant state `x`, notably for performing the time simulation of the controller, 
* (Optional) A `file` from which the controller was read (e.g. if it was synthesized in Matlab and imported in Python). 

As such, it overrides methods from `control.StateSpace` for LTI systems: addition, multiplication, concatenation, etc., as well as inversion.

Additionally, the class `Controller` implements a `Controller.step(y, dt)` method, which advances the time simulation with a step `dt` using the input `y` from the current internal state `self.x`. The method itself is merely a wrapper around `control.StateSpace.forced_response` with dimension manipulations.



---
## Saving and restarting
The toolbox offers two ways of starting a simulation. A simulation can be:
* starting from a given initial condition (IC) from the time instant $t=0$ (by default). The IC is usually provided as a perturbation field over the base flow.
* restarting from a given time instant that needs to correspond to a previously saved file.

Below, we describe both methods, as well as the way FEniCS saves files.
 
### Saved files: location, format, frequency and content
#### Location
All files are saved in the folder specified in the `ParamSave.path_out`. It is assumed to exist. In general, it is suggested to set:
```py
  cwd = Path(__file__).parent
  params_save = flowsolverparameters.ParamSave(
      save_every=xxx, path_out=cwd / "data_output"
  )
```

#### Format
FEniCS is able to save fields in the `xdmf` [format](https://www.xdmf.org/index.php/Main_Page), a usual choice for high-performance computing. This format differentiates two types of data: meta-data, stored using [XML](https://en.wikipedia.org/wiki/XML) in the `.xdmf` file, and value data, stored using [HDF5](https://en.wikipedia.org/wiki/HDF5) in the `.h5` file. In other words, a single field being saved produces two files.

#### Frequency
In the developed toolbox, when performing a simulation, fields can be saved at a given frequence: `ParamSave.save_every` parameter. As its name suggests, it saves the current field only at every `ParamSave.save_every` time step of the simulation. 

When saving several time steps, a single xdmf file (and its h5 counterpart) is used. All time steps are referenced in the xdmf file with their corresponding time instant, and the field is saved in the h5 file. However, when reading a field from the xdmf/h5 files, it is not possible to reference it with its time instant: it must be referenced to by its index. Below, in the _Restarting_ section, it is explained how it is used for restarting from a particular time instant.


#### Content
Several full fields fields are saved simultaneously: $U(t)$ (current velocity `U`), $U(t-\delta t$) (previous velocity `Uprev`), $P(t)$ (current pressure `P`). Note that those fields are full fields, and not perturbation fields. When computed, the base flow is automatically saved in the subfolder `steady` of the path `ParamSave.data_output`.



### Start from initial condition (IC) at $t=0$
At every simulated time step (starting from $t=0$, with `ParamTime.Tstart=0`), we evaluate whether the step is a multiple of `ParamSave.save_every`. If it is, the current field is saved as a `xdmf/h5` file. The file structure is summarized below. 

<p align="center">
<img src="https://github.com/user-attachments/assets/14fc68ee-7572-4e21-9c3a-59063612bf9d" alt="Illustration of the saving process" width="600" align=center/>
</p>

It is important to note that fields are referenced in two different ways in the `xdmf` and `h5` files. 
* In the `xdmf` file, they are referenced with the corresponding time instant. A field has the following structure:
```xml
      <Grid Name="U_0" GridType="Uniform">
        <Topology NumberOfElements="12284" TopologyType="Triangle" NodesPerElement="3">
            ...
        </Topology>
        <Geometry GeometryType="XY">
            ...
        </Geometry>
        <Time Value="0.000000" />
        <Attribute ItemType="FiniteElementFunction" ElementFamily="CG" ElementDegree="2" ElementCell="triangle" Name="U" Center="Other" AttributeType="Vector">
            ...
        </Attribute>
      </Grid>
```
* In the `h5` file, they are referenced with their corresponding index (i.e. an integer). There is, however, a clear correspondence between the time instant and the index.

This is particularly important to explain the next section.



### Restart a simulation from an arbitrary time instant $t$

It is only possible to restart from a time instant that was actually saved (i.e. a multiple of `ParamSave.save_every * ParamTime.dt`).

#### Automatic restart (recommended)

At the end of a simulation, call `fs.write_timeseries()` — this writes a JSON sidecar file (`meta_restart*.json`) into `path_out` that records the time grid, time step, and number of saved checkpoints. On restart, simply set `ParamTime.Tstart` to the desired restart time and call `initialize_time_stepping(Tstart=...)`. The solver scans `path_out` for a matching sidecar automatically:

```py
fs2 = CylinderFlowSolver(**params, Tstart=T_restart)
fs2.load_steady_state()
fs2.initialize_time_stepping(Tstart=T_restart)
```

No `ParamRestart` is needed. An example is in `examples/cylinder/cylinderflowsolver.py`.

#### Legacy restart (manual)

For simulations run before the JSON sidecar mechanism was introduced, or when the sidecar file is unavailable, provide a `ParamRestart` object:
* set `ParamRestart.Trestartfrom` to the start time of the original simulation.
* set `ParamRestart.dt_old` and `ParamRestart.save_every_old` to the values used in the original run.
* set `ParamTime.Tstart` to the desired restart time (must be a saved checkpoint).

The solver uses these fields to compute the checkpoint index in the `h5` file.

The saving/restarting process is summarized below.

<p align="center">
<img src="https://github.com/user-attachments/assets/97b65fcd-f1b5-4484-ada6-77e3ab380e95" alt="Illustration of the restarting process from saved files" width="600">
</p>







