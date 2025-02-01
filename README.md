# :cyclone: FlowControl 
The FlowControl toolbox is an open-source toolbox addressing the simulation and control of 2D incompressible flows. It aims at providing a user-friendly way to simulate flows with actuators and sensors, and a possibility to readily define new use-cases.


<p align="center">
<img src="illustrations/cylinder_stabilization.gif" alt="Animated GIF featuring the stabilization of the flow past a cylinder at Re=100. The self-sustained, periodic oscillations of the flow (known as vortex shedding) gradually disappear as the controller actuates the flow. The feedback controller uses a sensor in the wake and actuates the flow on the poles of the cylinder. More details are given below." width="500"/>
</p>

## Introduction
The primary goal of the toolbox is the design and implementation of feedback control algorithms, but it may be used for a variety of other topics such as model reduction or identification, actuator and sensor placement study... 

The toolbox is shipped with two benchmarks for flow control and allows for easy implementation of new cases.

<p align="center">
<img src="illustrations/cavity_stabilization.gif" alt="Animated GIF featuring the stabilization of the flow over an open cavity at Re=7500. The self-sustained, quasi-periodic oscillations of the flow gradually disappear as the controller actuates the flow. The feedback controller uses a wall stress sensor on the wall after the cavity, and actuates the flow with a volume force upstream of the cavity. More details are given below." width="500"/>
</p>

The core of the toolbox is in Python and relies on [FEniCS 2019.1.0](https://fenicsproject.org/) as a backend.

<p align="center">
<img src="illustrations/fenics_banner.png" alt="FEniCS Project banner, featuring a flame meshed with colorful elements and the text fenics project next to it." width="300"/>
</p>



## What the toolbox offers
### Simulation
+ By default, the toolbox integrates in time the
**Incompressible Navier-Stokes equations**. For a 2D flow defined by its velocity ${v}({x}, t) = [v_1({x}, t), v_2({x}, t)]$ and pressure $p({x}, t)$ inside a domain ${x} = [x_1, x_2] \in\Omega$, the equations read as follows:

```math
\left\{
\begin{aligned} 
&  \frac{\partial {v}}{\partial t} + ({v} \cdot \nabla){v} = -\nabla p +  \frac{1}{Re}\nabla^2 {v}    \\
&  \nabla \cdot {v} = 0
\end{aligned}
\right.
```
+ The only numerical parameter of the non-dimensional equations, the Reynolds number defined as $Re = \frac{UL}{\nu}$, balances convective and viscous terms.



### Actuation and sensing
The toolbox allows the user to define actuators and sensors for forcing the flow. It also provides utility for controller design and implementation. See the examples given below.



### Two benchmarks
Two classic [oscillator flows](https://journals.aps.org/prfluids/pdf/10.1103/PhysRevFluids.1.040501) used for flow control are shipped with the current code.
| Use-case | Description | Feedback configuration |
| ------   | ----------- | ---------------------- |
| Cylinder | Flow past a cylinder at Re=100 |  SISO |
| Cavity | Flow over an open cavity at Re=7500  | SISO |

+ For the flow past a cylinder at Re=100, see e.g.:
  - [Barkley, D. (2006). Linear analysis of the cylinder wake mean flow. _Europhysics Letters_, 75(5), 750.](https://homepages.warwick.ac.uk/~masax/Research/Papers/Cylinder_EPL_75.pdf),
  - [Paris, R., Beneddine, S., & Dandois, J. (2021). Robust flow control and optimal sensor placement using deep reinforcement learning. _Journal of Fluid Mechanics_, 913, A25.](https://arxiv.org/pdf/2006.11005),
+ For the flow over an open cavity at Re=7500, see e.g.: 
  - [Barbagallo, A., Sipp, D., & Schmid, P. J. (2009). Closed-loop control of an open cavity flow using reduced-order models. _Journal of Fluid Mechanics_, 641, 1-50.](https://polytechnique.hal.science/hal-01021129/file/S0022112009991418a.pdf).



---
---
### Flow past a cylinder at Re=100
#### Illustration

<p align="center">
<img src="illustrations/cylinder_baseflow.png" alt="Stationary solution of the flow past a cylinder at Re=100. The base flow is symetric with respect to the axis x_2=0 and features a long recirculation bubble after the cylinder obstacle." width="400"/>
</p>
<p align="center">
Stationary solution
</p>



<p align="center">
<img src="illustrations/cylinder_attractor.png" alt="Snapshot of the attractor of the flow past a cylinder at Re=100. The vortex shedding exhibits symmetry around the x_2=0 axis and vortices are shed periodically." width="400"/>
</p>
<p align="center">
Periodic attractor (stable limit cycle)
</p>

#### Feedback configuration
The default feedback configuration (same as in [Jussiau, W., Leclercq, C., Demourant, F., & Apkarian, P. (2022). Learning linear feedback controllers for suppressing the vortex-shedding flow past a cylinder. _IEEE Control Systems Letters_, 6, 3212-3217.](https://hal.science/hal-03947469/document)) is as follows:
* Cross-stream velocity measurement $y(t)=v_2({x_s}, t)$ in the wake at ${x_s} = [3, 0]$,

* Boundary actuation at the poles of the cylinder, acting on the cross-stream velocity $v_2$. The velocity profile on the actuated boundary reads: 

$${v_{act}}({x}, t) =  -  \dfrac{(x_1-l)(x_1+l)}{l^2} u(t)$$

where $u(t)$ is the control input, $l = \frac{1}{2} D  \sin  \left( \frac{\delta}{2} \right)$, $\delta=10\degree$ are tunable actuator parameters.

<p align="center">
<img src="illustrations/cylinder_actuator.png" alt="Actuator profile at the top pole of the cylinder. The spatial parabolic profile amplitude is modified by the control input u(t)." width="300"/>
</p>

<p align="center">
<img src="illustrations/cylinder_domain.png" alt="Description of the feedback configuration of the cylinder. The cross-stream velocity sensor (black dot) in the wake (red wave) can be fed to the controller K (in blue) and conveyed back to the actuators at the poles of the cylinder (in green)." width="400"/>
</p>



---
---
### Flow over an open cavity at Re=7500
#### Illustration


<p align="center">
<img src="illustrations/cavity_baseflow.png" alt="Stationary solution of the flow over an open cavity at Re=7500" width="400"/>
</p>
<p align="center">
Stationary solution
</p>


<p align="center">
<img src="illustrations/cavity_attractor.png" alt="Snapshot of the attractor of the flow over an open cavity at Re=7500" width="400"/>
</p>
<p align="center">
Quasi-periodic attractor
</p>


Contrary to the cylinder, the attractor on the cavity at Re=7500 is quasi-periodic (featuring two incommensurable frequencies in its frequency spectrum):

<p align="center">
<img src="illustrations/cavity_timeresp_zoom.png" alt="Snapshot of the attractor of the flow over an open cavity at Re=7500" width="400"/>
</p>

#### Feedback configuration
The default feedback configuration (same as in [Leclercq et al. (2019). Linear iterative method for closed-loop control of quasiperiodic flows. _Journal of Fluid Mechanics_, 868, 26-65.](https://hal.science/hal-02296280/document)) is as follows:
*  Actuation is produced near the upstream edge of the cavity by a volume force $f({x}, t)=B({x}) u(t)$ in the momentum equation, acting on the cross-stream velocity, with:

$$B({x})=\left[ 0, \eta \exp\left( \frac{\left(x_1 - x_1^0\right)^2 + \left(x_2 - x_2^0\right)^2}{2\sigma_0^2}  \right), 0 \right]^T$$

By default, the center of the actuator is $(x_1^0, x_2^0) = (-0.1, 0.02)$, just before the cavity and slightly above the wall. The amplitude $\eta\approx 8.25$ is chosen such that $\int_\Omega B({x})^T B({x}) d\Omega = 1$. The spatial extent of the actuation is set by $\sigma_0 = 0.0849$, making the force reach $50\%$ of its peak value at a distance $0.1$ from its center.


*  The measurement is made through wall friction on the bottom wall just downstream of the cavity:

$$y(t) = \int_{x_1=1}^{1.1}   \left.  \frac{\partial v_1(x, t)}{\partial x_2} \right\rvert_{x_2=0} dx_1$$

<p align="center">
<img src="illustrations/cavity_domain.png" alt="Visual description of the feedback configuration of the cavity. The wall friction is measured on the bottom wall downstream of the cavity, then fed to a controller K, which produces an input signal u(t) modifying the amplitude of a volumic force before the cavity." width="400"/>
</p>



### Numerical details
+ For discretization in space, the Finite Element Method is used, using default continuous Galerkin elements of order 2 (for each component of the velocity) and 1 (for the scalar pressure).

+ For the time integration, a linear multistep semi-implicit method is used (the nonlinear term is extrapolated with a second-order Adams–Bashforth scheme, while the viscous term
is treated implicitly).

+ The equations are implemented using a perturbation formulation: 
    - the field $v(x,t)$ is decomposed as $v(x,t) = V(x) + v'(x, t)$,
    - $V(x)$ is computed first,
    - then, we can compute the time evolution of $v'(x,t)$.

+ To some extent, the toolbox aims at making the equations, numerical integration schemes and solvers replaceable by user-defined ones.


---
---
## Examples of use of the toolbox
The following articles were based on previous versions of the code:
* [Jussiau, W., Leclercq, C., Demourant, F., & Apkarian, P. (2022). Learning linear feedback controllers for suppressing the vortex-shedding flow past a cylinder. _IEEE Control Systems Letters_, 6, 3212-3217.](https://hal.science/hal-03947469/document)
* [Jussiau, W., Leclercq, C., Demourant, F., & Apkarian, P. (2024). Data-driven stabilization of an oscillating flow with linear time-invariant controllers. _Journal of Fluid Mechanics_, 999, A86.](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/47548BEA53D115E1F70FC1F772F641DB/S0022112024009042a.pdf/data-driven-stabilization-of-an-oscillating-flow-with-linear-time-invariant-controllers.pdf)



---
---
## Installation 🛠️
### conda
The ```conda```  environment required to run the code can be extracted from the file ```environment.yml```. Additional path tweaking may be required for all FEniCS (```dolfin``` module) and custom modules to be found.

### Docker :whale:
[coming soon]



## Code overview

### Define a new use-case: inherit FlowSolver abstract class
The simulation revolves around the abstract class ```FlowSolver``` that implements core features such as loading mesh, defining function spaces, trial/test functions, variational formulations, numerical schemes and solvers, handling the time-stepping and exporting information. The class is abstract as it does not implement a simulation case _per se_, but only provides utility for doing so. It features two abstract methods, that are redefined for each use-case:

1. ```_make_boundaries``` provides a definition and naming of the boundaries of the mesh in a pandas DataFrame.
``` py
    @abstractmethod
    def _make_boundaries(self) -> pd.DataFrame:
        pass
```
2. ```_make_bcs``` provides a description of the boundary conditions on the boundaries defined above, in a dictionary.
``` py
    @abstractmethod
    def _make_bcs(self) -> dict[str, Any]:
        pass
```

For the two aforementioned examples, these methods are reimplemented in the classes ```CylinderFlowSolver``` and ```CavityFlowSolver``` that inherit from ```FlowSolver```.



### Attach ```Sensor```s and ```Actuator```s to an instance of a ```FlowSolver``` subclass
In order to perform sensing and actuation (in order to close the loop), dedicated classes ```Sensor``` and ```Actuator``` are proposed. They are not aimed at being instantiated, but rather inherited.

* ```Sensor``` is an abstract class that provides a method ```eval(self, up: dolfin.Function) -> float```. Classes ```SensorPoint``` (point probe) and ```SensorIntegral``` (integration on a subdomain) are examples of subclasses that implement the ```eval``` method.
* Likewise, ```Actuator``` is an abstract class that encapsulates a ```dolfin.Expression``` among other elements, and embeds it in the variational formulations.

The sensors and actuators are attached to a FlowSolver object as a list, embedded in the ```ParamControl``` object. The call to ```Sensor```s and ```Actuator```s is made automatically by ```FlowSolver```.
By attaching several sensors or actuators, it is possible to use Multiple-Input, Multiple-Output controllers in the loop.

In the example below (for the cylinder use-case), we are creating an actuator acting on boundary conditions and three point probes at different locations. They are gathered in a ```ParamControl``` object, which is passed as an argument when creating a ```CylinderFlowSolver```. 
``` py
# Actuator
actuator_bc = ActuatorBCParabolicV(angular_size_deg=10)
# Sensors
sensor_feedback = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3, 0]))
sensor_perf_1 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3.1, 1]))
sensor_perf_2 = SensorPoint(sensor_type=SENSOR_TYPE.V, position=np.array([3.1, -1.3]))
# Gather actuators and sensors in ParamControl object
params_control = ParamControl(
    sensor_list=[sensor_feedback, sensor_perf_1, sensor_perf_2],
    actuator_list=[actuator_bc],
)
```

### Running a closed-loop simulation
Once a use-case has been defined by implementing the corresponding class, the basic feedback syntax has the following philosophy:
``` py
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

See examples for a more detailed description.



### Meshing tools
No meshing tools are shipped with this code, but [gmsh](https://gmsh.info/) (and [its Python API](https://pypi.org/project/gmsh/)) are suggested for generating meshes. The mesh should be exported to ```xdmf``` format, which can be reached thanks to [meshio](https://github.com/nschloe/meshio/tree/main).



### Visualization
[Paraview](https://www.paraview.org/) is suggested for visualizations, whether it be for ```csv``` timeseries or fields saved as ```xdmf```.



### Additional uses of the toolbox
The toolbox provides additional utility related to flow control:
* Compute dynamic operators A, B, C, D and mass matrix E,
* Restart a simulation from a previous one,
* Arbitrary number of sensors (for feedback or performance),
* Export time series (measurements from sensors, perturbation kinetic energy...) and fields for visualization,
* Parallel execution native to FEniCS,
* To some extent, easy modification of the equations, numerical schemes and solvers used for time simulation,
* Can be used as backend in an optimization tool (as in [Jussiau, W., Leclercq, C., Demourant, F., & Apkarian, P. (2022). Learning linear feedback controllers for suppressing the vortex-shedding flow past a cylinder. _IEEE Control Systems Letters_, 6, 3212-3217.](https://hal.science/hal-03947469/document)).



## Roadmap
The current roadmap is as follows:
* Complete the documentation :book:,
* Refactor and release additional control-related tools,
* Update the project to [FEniCSx](https://fenicsproject.org/documentation/),
* Sort and check all utility functions,
* General form for operator computation,
* Docker/venv/pip.



## Contact
:mailbox: william.jussiau@gmail.com

Also, I highly recommend [FEniCS documentation](https://olddocs.fenicsproject.org/dolfin/2019.1.0/), [FEniCS forum](https://fenicsproject.discourse.group/) (and potentially [the BitBucket repository](https://bitbucket.org/fenics-project/dolfin/src/master/)) for problems regarding FEniCS 2019.1.0 itself.


---


This README has been optimized for accessibility based on GitHub's blogpost "[Tips for Making your GitHub Profile Page Accessible](https://github.blog/2023-10-26-5-tips-for-making-your-github-profile-page-accessible)".

