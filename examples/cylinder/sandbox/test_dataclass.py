from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class Param_try:
    Re: float
    meshpath: Path
    meshname: str
    timeseries: np.array


if __name__ == "__main__":
    params_flow = Param_try(Re=100, meshpath=Path(), meshname="o1.xdmf")
    print(params_flow)
    print(params_flow.Re)
    params_flow.newpar = 13.8
