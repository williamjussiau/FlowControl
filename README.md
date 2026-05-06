# :cyclone: FlowControl

The FlowControl toolbox is an open-source toolbox addressing the simulation and control of 2D incompressible flows at low Reynolds number. It aims at providing a user-friendly way to simulate flows with actuators and sensors, the possibility to readily define new use-cases, and support for operators & frequency response computations.

For in-depth documentation, see [docs/](docs/).

<p align="center">
<img src="./illustrations/cylinder_stabilization.gif" alt="Animated GIF featuring the stabilization of the flow past a cylinder at Re=100. The self-sustained, periodic oscillations of the flow (known as vortex shedding) gradually disappear as the controller actuates the flow. The feedback controller uses a sensor in the wake and actuates the flow on the poles of the cylinder." width="500"/>
</p>

The toolbox is shipped with four benchmarks for flow control and allows for easy implementation of new cases.

<p align="center">
<img src="./illustrations/cavity_stabilization.gif" alt="Animated GIF featuring the stabilization of the flow over an open cavity at Re=7500. The self-sustained, quasi-periodic oscillations of the flow gradually disappear as the controller actuates the flow. The feedback controller uses a wall stress sensor on the wall after the cavity, and actuates the flow with a volume force upstream of the cavity." width="500"/>
</p>

The core of the toolbox is in Python and relies on [FEniCS 2019.1.0](https://fenicsproject.org/) as a backend.

<p align="center">
<img src="./illustrations/fenics_banner.png" alt="FEniCS Project banner, featuring a flame meshed with colorful elements and the text fenics project next to it." width="300"/>
</p>


## Installation 🛠️

### conda

The conda environment required to run the code is defined in `environment.yml`. The proposed installation pipeline is:

```bash
conda env create -n fenics --file environment.yml
conda activate fenics
conda develop src
```

Additional path tweaking is sometimes required for FEniCS to be found through the `dolfin` module (see e.g. [this problem with PKG_CONFIG](https://fenicsproject.discourse.group/t/problem-with-fenics-and-macos-catalina/2106)).


### Docker :whale:

[Coming soon]


## What the Toolbox Offers

### Features

- **Actuators and Sensors**: define any number of actuators (boundary velocity, body force, cylinder rotation) and sensors (point measurements, wall shear stress)
- **Closed-loop control**: built-in `Controller` class; connect a state-space controller in a few lines
- **Linearized operators**: compute A, B, C, D and mass matrix E for control design
- **Checkpointing and restart**: save snapshots and resume from any checkpoint via JSON saves
- **Export**: timeseries CSV and XDMF fields for Paraview visualization
- **Parallel**: MPI-parallel execution via FEniCS native support
- **Extensible**: use as a backend in optimization or data-driven pipelines

### Four Benchmarks

Four classic [oscillator flows](https://journals.aps.org/prfluids/pdf/10.1103/PhysRevFluids.1.040501) used for flow control are shipped with the current code.

| Use-case | Description | Suggested Reynolds number |
| ------- | ----------- | ------------------------- |
| Cylinder | Flow past a cylinder | Re=100 |
| Lid-driven cavity | Flow in a lid-driven cavity | Re=8000 |
| Open cavity | Flow over an open cavity | Re=7500 |
| Fluidic pinball | Flow past 3 cylinders | Re=100 |


## Code Philosophy

To implement a new use-case, the workflow is the following:

1. Define a new use-case: inherit the `FlowSolver` abstract class
2. Attach `Sensor`s and `Actuator`s to an instance of a `FlowSolver` subclass
3. Run a simulation using an input signal $u(t)$, either open-loop or closed-loop via a `Controller`

See [Code Basics](docs/code-basics.md) for more information on how to perform these steps.


## Simulation

By default, the toolbox integrates in time the **Incompressible Navier-Stokes equations**. For a 2D flow defined by its velocity ${v}({x}, t) = [v_1({x}, t), v_2({x}, t)]$ and pressure $p({x}, t)$ inside a domain ${x} = [x_1, x_2] \in\Omega$, the equations read as follows:

```math
\left\{
\begin{aligned} 
&  \frac{\partial {v}}{\partial t} + ({v} \cdot \nabla){v} = -\nabla p +  \frac{1}{Re}\nabla^2 {v}   \
&  \nabla \cdot {v} = 0
\end{aligned}\right.
```

The only numerical parameter of the non-dimensional equations, the Reynolds number defined as $Re = \frac{UL}{\nu}$, balances convective and viscous terms.


## Publications

The following articles were based on previous versions of the code:

- [Jussiau, W., Leclercq, C., Demourant, F., & Apkarian, P. (2022). Learning linear feedback controllers for suppressing the vortex-shedding flow past a cylinder. *IEEE Control Systems Letters*, 6, 3212-3217.](https://hal.science/hal-03947469/document)
- [Jussiau, W., Leclercq, C., Demourant, F., & Apkarian, P. (2024). Data-driven stabilization of an oscillating flow with linear time-invariant controllers. *Journal of Fluid Mechanics*, 999, A86.](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/47548BEA53D115E1F70FC1F772F641DB/S0022112024009042a.pdf/data-driven-stabilization-of-an-oscillating-flow-with-linear-time-invariant-controllers.pdf)
- [Jussiau, W., Demourant, F., Leclercq, C., & Apkarian, P. (2025). Control of a Class of High-Dimensional Nonlinear Oscillators: Application to Flow Stabilization. *IEEE Transactions on Control Systems Technology*.](https://ieeexplore.ieee.org/abstract/document/10884641/)


## Roadmap

- Complete the documentation :book:
- Review and release operator and frequency-response computation tools
- Refactor utility functions
- Update the project to [FEniCSx](https://fenicsproject.org/documentation/)


## Contact

:mailbox: william.jussiau@gmail.com

---

This README has been optimized for accessibility based on GitHub's blogpost "[Tips for Making your GitHub Profile Page Accessible](https://github.blog/2023-10-26-5-tips-for-making-your-github-profile-page-accessible)".
