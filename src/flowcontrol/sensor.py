"""Sensor classes for pointwise and subdomain-integral flow measurements.

Classes
-------
SENSOR_TYPE               : enum — U (x-velocity), V (y-velocity), P (pressure), OTHER
Sensor                    : abstract base class; subclasses implement eval()
SensorPoint               : pointwise probe at a 2D position (MPI-safe)
SensorIntegral            : abstract base for sensors that integrate over a subdomain;
                            subclasses implement load() and linear_form()
SensorHorizontalWallShear : integral of dv/dx2 along a segment of the bottom wall
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from flowcontrol.flowsolver import FlowSolver

import dolfin
import numpy as np
from numpy.typing import NDArray

from utils.mpi import peval

SENSOR_INDEX_DEFAULT = 10000


class SENSOR_TYPE(IntEnum):
    """Component extracted by a sensor.

    ``U`` and ``V`` are the x- and y-velocity components; ``P`` is pressure;
    ``OTHER`` covers derived quantities such as integrals or derivatives.
    """

    U = 0
    V = 1
    P = 2
    OTHER = 3


@dataclass(kw_only=True)
class Sensor(ABC):
    """Abstract base class for all sensors.

    Attributes
    ----------
    sensor_type :
        Which field component this sensor extracts.
    require_loading :
        ``True`` if :meth:`load` must be called before the first :meth:`eval`
        (e.g. integral sensors that need a subdomain).
    """

    sensor_type: SENSOR_TYPE
    require_loading: bool

    @abstractmethod
    def eval(self, up: dolfin.Function) -> float:
        """Return the scalar measurement from the mixed field ``(u, p)``.

        Called once per time step; implementations must be MPI-compatible
        (see :func:`utils.mpi.peval` for point evaluations).

        Parameters
        ----------
        up :
            Mixed-space dolfin.Function holding the current ``(u, p)`` state.

        Returns
        -------
        float
            Scalar sensor reading.
        """
        pass


@dataclass(kw_only=True)
class SensorPoint(Sensor):
    """Pointwise probe that reads a single field component at a 2D location.

    Parameters
    ----------
    sensor_type :
        Which component to read (U, V, or P).
    position :
        2D probe location as ``[x, y]``.
    """

    position: NDArray[np.float64]
    require_loading: bool = False

    def eval(self, up: dolfin.Function) -> float:
        # warning: need to be compatible with parallel
        return peval(up, dolfin.Point(self.position))[self.sensor_type]
        # for example, do not:
        # return up(self.position[0], self.position[1])[self.sensor_type]


@dataclass(kw_only=True)
class SensorIntegral(Sensor):
    """Abstract base class for sensors that integrate a quantity over a subdomain.

    Subclasses must implement :meth:`load` (to build the subdomain and measure)
    and :meth:`linear_form` (to define the integrand).

    Attributes
    ----------
    sensor_index :
        Integer tag used to mark the integration subdomain.  Must be unique
        across all sensors attached to the same FlowSolver.
    ds :
        Integration measure (set by :meth:`load`).
    subdomain :
        Marked subdomain (set by :meth:`load`).
    """

    ds: Optional[dolfin.Measure] = None
    subdomain: Optional[dolfin.SubDomain] = None
    sensor_index: int = SENSOR_INDEX_DEFAULT
    require_loading: bool = True

    @abstractmethod
    def load(self, flowsolver: FlowSolver) -> None:
        """Build the integration subdomain and measure from the live FlowSolver.

        Must be called once after the FlowSolver is initialised, before the
        first call to :meth:`eval`.  Implementations should populate
        ``self.subdomain`` and ``self.ds``.

        Parameters
        ----------
        flowsolver :
            The live FlowSolver providing the mesh and boundary markers.
        """
        pass

    @abstractmethod
    def linear_form(self, v: Any) -> Any:
        """Return the UFL form that defines this sensor's measurement.

        The form must be linear in ``v``.  Two usage modes:

        - ``v`` is a ``dolfin.Function``: ``dolfin.assemble(linear_form(v))``
          returns the scalar measurement.
        - ``v`` is a ``dolfin.TestFunction``: ``dolfin.assemble(linear_form(v))``
          returns the corresponding row of the C matrix.

        Requires :meth:`load` to have been called first.

        Parameters
        ----------
        v :
            A dolfin Function or TestFunction on the mixed space W.

        Returns
        -------
        ufl.Form
            UFL form linear in ``v``.
        """
        pass

    def eval(self, up: dolfin.Function) -> float:
        """Assemble linear_form over the loaded subdomain and return the scalar result."""
        return dolfin.assemble(self.linear_form(up))


@dataclass(kw_only=True)
class SensorHorizontalWallShear(SensorIntegral):
    """Sensor that integrates the wall shear stress ``∂v/∂x₂`` along a bottom-wall segment.

    Parameters
    ----------
    sensor_type :
        Component to measure (typically ``SENSOR_TYPE.OTHER``).
    x_sensor_left :
        Left x-limit of the integration segment.
    x_sensor_right :
        Right x-limit of the integration segment.
    y_sensor :
        Wall height (y-coordinate of the boundary segment).
    """

    x_sensor_left: float = 1.0
    x_sensor_right: float = 1.1
    y_sensor: float = 0.0

    def linear_form(self, v: Any) -> Any:
        """Return the UFL form integrating ∂v[0]/∂x[1] over the wall segment.

        Integrates the wall-normal derivative of the streamwise velocity
        component (v[0] = u_x) along the marked horizontal boundary segment.
        """
        return v[0].dx(1) * self.ds(self.sensor_index)

    def load(self, flowsolver: FlowSolver) -> None:
        """Mark the wall segment and build the boundary measure ``self.ds``.

        Parameters
        ----------
        flowsolver :
            The live FlowSolver providing the mesh.
        """
        sensor_subdomain = dolfin.CompiledSubDomain(
            "on_boundary && near(x[1], y_sensor, MESH_TOL) && x[0]>=x_sensor_left && x[0]<=x_sensor_right",
            MESH_TOL=dolfin.DOLFIN_EPS,
            x_sensor_left=self.x_sensor_left,
            x_sensor_right=self.x_sensor_right,
            y_sensor=self.y_sensor,
        )

        sensor_mark = dolfin.MeshFunction(
            "size_t", flowsolver.mesh, flowsolver.mesh.topology().dim() - 1
        )

        sensor_subdomain.mark(sensor_mark, self.sensor_index)
        self.subdomain = sensor_subdomain
        self.ds = dolfin.Measure(
            "ds", domain=flowsolver.mesh, subdomain_data=sensor_mark
        )
        # to define 2D subdomain, use dolfin.Measure("dx") instead


if __name__ == "__main__":
    sensor_feedback_cylinder = SensorPoint(
        sensor_type=SENSOR_TYPE.V, position=np.array([3, 0])
    )

    sensor_feedback_cavity = SensorHorizontalWallShear(
        sensor_index=100,
        x_sensor_left=1.0,
        x_sensor_right=1.1,
        y_sensor=1.0,
        sensor_type=SENSOR_TYPE.OTHER,
    )
