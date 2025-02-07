from abc import ABC, abstractmethod
from enum import IntEnum
from dataclasses import dataclass
import dolfin
import numpy as np

SENSOR_INDEX_DEFAULT = 100


class SENSOR_TYPE(IntEnum):
    """Type of data returned by the Sensor

    Args:
        U: velocity, 1st component
        V: velocity, 2nd component
        P: pressure
        OTHER: mixed, e.g. integral or derivative
    """

    U = 0
    V = 1
    P = 2
    OTHER = 5


@dataclass(kw_only=True)
class Sensor(ABC):
    """Sensor abstract base class, providing the abstract method eval().

    Args:
        sensor_type (SENSOR_TYPE): see SENSOR_TYPE
        require_loading (bool): flag - some sensors require loading (e.g.
            for defining their integration subdomain) after the FlowSolver
            they are attached to is initialized.
    """

    sensor_type: SENSOR_TYPE
    require_loading: bool

    @abstractmethod
    def eval(self, up):
        # is it eval(up) or eval(u,v,p)? Absolutely AVOID split/merge in eval
        """Evaluate measurement value from sensor on (mixed) field (u,p)
        This function is going to be called a high number
        of times, so it needs to be optimized.
        The user is responsible for the parallelism compatibility
        (see for example flu.MpiUtils.peval)"""
        pass


@dataclass(kw_only=True)
class SensorPoint(Sensor):
    """Pointwise probe. It extracts information from the given field
    at a given 2D point _self.position_.

    Args:
        position (np.ndarray): position of probe
        require_loading (bool) = False: no loading required
    """

    position: np.ndarray
    require_loading: bool = False

    def eval(self, up):
        # warning: might need to be compatible with parallel
        # flu.MpiUtils.peval(up, position)
        return up(self.position[0], self.position[1])[self.sensor_type]


@dataclass(kw_only=True)
class SensorIntegral(Sensor):
    """Abstract base class for sensors performing integration on a subdomain,
     providing the abstract method _load. A SensorIntegral always require loading,
     which corresponds to initializing a _dolfin.SubDomain_ and a _dolfin.Measure_.

    Args:
        sensor_index (int): sensor index for marking the integration subdomain. If
            instantiating several sensors, they should not be equal
        ds (dolfin.Measure): curve element enabling dolfin integration
        subdomain (dolfin.SubDomain): subdomain to integrate (1D or 2D)
        require_loading (bool) = True: SensorIntegral always require loading
    """

    ds: dolfin.Measure | None = None
    subdomain: dolfin.SubDomain | None = None
    sensor_index: int | None = None
    require_loading: bool = True

    @abstractmethod
    def _load(self) -> None:
        """Defne and mark subdomain, define integration element ds."""
        pass


@dataclass
class SensorHorizontalWallShear(SensorIntegral):
    """Cavity sensor, integrating the wall shear stress (dv/dx2) on a
    portion of the channel bottom wall.

    Args:
        x_sensor_left (float): left x-limit of the sensor
        x_sensor_right (float): right x-limit of the sensor
        y_sensor (float): height of the sensor
    """

    x_sensor_left: float = 1.0
    x_sensor_right: float = 1.1
    y_sensor: float = 0.0

    def eval(self, up):
        return dolfin.assemble(up.dx(1)[0] * self.ds(int(self.sensor_index)))

    def _load(self, flowsolver):
        sensor_subdm = dolfin.CompiledSubDomain(
            "on_boundary && near(x[1], y_sensor, MESH_TOL) && x[0]>=x_sensor_left && x[0]<=x_sensor_right",
            MESH_TOL=dolfin.DOLFIN_EPS,
            x_sensor_left=self.x_sensor_left,
            x_sensor_right=self.x_sensor_right,
            y_sensor=self.y_sensor,
        )

        sensor_mark = dolfin.MeshFunction(
            "size_t", flowsolver.mesh, flowsolver.mesh.topology().dim() - 1
        )
        if self.sensor_index is None:
            self.sensor_index = SENSOR_INDEX_DEFAULT

        sensor_subdm.mark(sensor_mark, self.sensor_index)
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
