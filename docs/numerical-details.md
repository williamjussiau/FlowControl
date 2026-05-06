# Numerical Details

## Perturbation Formulation

The equations are implemented using a perturbation formulation:
- the field $U(x,t)$ is decomposed as $U(x,t) = U_0(x) + u'(x, t)$
- $U_0(x)$ is computed first
- then, we can compute the time evolution of $u'(x,t)$ to retrieve the full field $U(x,t)$

The same process is done with the pressure fields $P(x,t), P_0(x), p'(x,t)$.

In the code, `U, P` (capital) refer to the full field $U(x,t)$, while `u, p` (small) refer to the perturbation fields $u'(x,t), p'(x,t)$. For boundary conditions, `BC, bc` follow the same convention.

## Finite Element Method

For discretization in space, the Finite Element Method (FEM) is used, using default continuous Galerkin elements of order 2 (for each component of the velocity) and 1 (for the scalar pressure). In the code, the elements are respectively defined as:

```python
Ve = dolfin.VectorElement("CG", self.mesh.ufl_cell(), 2)
Pe = dolfin.FiniteElement("CG", self.mesh.ufl_cell(), 1)
```

These elements and their function spaces are created in the `_make_function_spaces()` method. The function spaces are respectively `V, P` (registered as object attributes), and `W` which is the concatenation of the two former.

## Discretization in Time

For the time integration, a linear multistep semi-implicit method is used (the nonlinear term is extrapolated with a second-order Adams–Bashforth scheme, while the viscous term is treated implicitly). The variational formulation of the equations for an unactuated flow reads as follows:

```python
F = dot((3 * u - 4 * u_n + u_nn) / (2 * dt), v) * dx
    + dot(dot(U0, nabla_grad(u)), v) * dx
    + dot(dot(u, nabla_grad(U0)), v) * dx
    + 1/Re * inner(nabla_grad(u), nabla_grad(v)) * dx
    + 2 * dot(dot(u_n, nabla_grad(u_n)), v) * dx
    + -1 * dot(dot(u_nn, nabla_grad(u_nn)), v) * dx
    - p * div(v) * dx
    - div(u) * q * dx
```

When one or several actuators are used, an additional term `-dot(f, v) * dx` is appended to the variational form, where `f` contains all the *volumic force* actuators contributions (i.e. `Actuator`s with the type `ACTUATOR_TYPE.FORCE`).

## How to Modify Numerical Schemes?

To some extent, the toolbox aims at making the equations, numerical integration schemes and solvers replaceable by user-defined ones. For example, one may override the following methods:

```python
_make_varf(self, order: int, **kwargs) -> dolfin.Form
_make_solver(self, **kwargs) -> dolfin.KrylovSolver | dolfin.LUSolver
```

from the abstract class `FlowSolver` to implement new schemes or solvers.

The FEM elements may be replaced with elements available within FEniCS by overriding:

```python
_make_function_spaces(self) -> tuple[dolfin.FunctionSpace, ...]
```

Attempting to use nodal-enriched (or *bubble*) elements is not straightforward and may break parts of the code.
