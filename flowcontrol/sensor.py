from abc import ABC, abstractmethod
from enum import IntEnum
from dataclasses import dataclass
import dolfin
import numpy as np


class SENSOR_TYPE(IntEnum):
    """Enumeration of sensor types

    Args: TODO
        # POINT: pointwise measurement
        # INTEGRAL: integral of something
        # OTHER: other measurement
    """

    U = 0
    V = 1
    P = 2
    OTHER = 5


# class SENSORPOINT_TYPE(IntEnum):
#     U = 0
#     V = 1
#     P = 2


@dataclass(kw_only=True)
class Sensor(ABC):
    sensor_type: SENSOR_TYPE
    position: np.ndarray

    @abstractmethod
    def eval(self, up):
        # is it eval(up) or eval(u,v,p)? Absolutely AVOID split/merge in eval
        """Evaluate measurement value from sensor on (mixed) field up
        This function is going to be called a high number
        of times, so it needs to be optimized.
        The user is responsible for the parallelism compatibility
        (see for example flu.MpiUtils.peval)"""
        pass


class SensorPoint(Sensor):
    def eval(self, up):
        # warning: might need to be compatible with parallel
        # flu.MpiUtils.peval(up, position)
        return up(self.position[0], self.position[1])[self.sensor_type]


@dataclass
class SensorIntegral(Sensor, ABC):
    size: np.ndarray
    ds: dolfin.Measure | None = None

    # @abstractmethod
    # def eval(self, up):
    #     SENSOR_IDX = 1  # TODO
    #     return dolfin.assemble(up.dx(1)[0] * self.ds(int(SENSOR_IDX)))

    @abstractmethod
    def _make_subdomain(self):
        # xs0 = 1.0
        # xs1 = 1.1
        # MESH_TOL = dolfin.DOLFIN_EPS
        # sensor_subdm = dolfin.CompiledSubDomain(
        #     "on_boundary && near(x[1], 0, MESH_TOL) && x[0]>=xs0 && x[0]<=xs1",
        #     MESH_TOL=MESH_TOL,
        #     xs0=xs0,
        #     xs1=xs1,
        # )
        return 1
        # return sensor_subdm

    def _setup_subdomain(self):
        # subdomain = self._make_subdomain()
        # sensor_mark = dolfin.MeshFunction(
        #     "size_t", self.mesh, self.mesh.topology().dim() - 1
        # )
        # SENSOR_IDX = 100
        # sensor_subdm.mark(sensor_mark, SENSOR_IDX)
        # ds_sensor = dolfin.Measure("ds", domain=self.mesh, subdomain_data=sensor_mark)
        # self.ds_sensor = ds_sensor

        # df_sensor = pd.DataFrame(
        #     data=dict(subdomain=sensor_subdm, idx=SENSOR_IDX), index=["sensor"]
        # )
        # self.boundaries = pd.concat((self.boundaries, df_sensor))
        # self.sensor_ok = True  # TODO rm
        return 1


if __name__ == "__main__":
    pass
    # sensor1 = SensorPoint(
    #     position=np.array([1, 2]), parameters=None, sensor_type=SENSOR_TYPE.U
    # )

    # sensor2 = SensorIntegral(
    #     position=np.array([1, 2]),
    #     parameters=None,
    #     sensor_type=SENSOR_TYPE.U,
    #     size=1,
    #     ds=1,
    # )
