import control
import numpy as np
from pathlib import Path
import utils_flowsolver as flu
import youla_utils as yu


class SimulatedStateSpace(control.StateSpace):
    x: np.array
    file: Path = None

    def __init__(self, A, B, C, D, x0=0, file=None):
        super().__init__(A, B, C, D)
        self.file = file
        self.x = x0
        if x0 == 0:
            self.x = np.zeros((self.nstates,))

    @classmethod
    def from_file(cls, x0=0, file=None):
        stateSpaceMatrices = flu.read_matfile(file)
        return cls(
            stateSpaceMatrices["A"],
            stateSpaceMatrices["B"],
            stateSpaceMatrices["C"],
            stateSpaceMatrices["D"],
            x0=x0,
            file=file,
        )

    @classmethod
    def from_matrices(cls, A, B, C, D, x0=0):
        return cls(A, B, C, D, x0=x0)

    def step(self):
        self.x = self.x + 1

    def __add__(self, other):
        print("calling add")
        # check instances
        # if other is SimulatedStatepace, concatenate x
        # set file None (if different files)
        return super().__add__(other)

    def __radd__(self, other):
        print("calling radd")
        return super().__radd__(other)

    def __mul__(self, other):
        print("calling mul")
        return super().__mul__(other)

    def __rmul__(self, other):
        print("calling rmul")
        return super().__rmul__(other)

    def inv(self):
        print("invert sys")
        return yu.ss_inv(self)


if __name__ == "__main__":
    cwd = Path(__file__).parent
    sspath = cwd / "data_input" / "sysid_o16_d=3_ssest.mat"

    kd = flu.read_matfile(sspath)
    K1 = SimulatedStateSpace.from_matrices(A=kd["A"], B=kd["B"], C=kd["C"], D=kd["D"])
    K2 = SimulatedStateSpace.from_file(
        file=cwd / "data_input" / "sysid_o16_d=3_ssest.mat"
    )
    K3 = SimulatedStateSpace(A=kd["A"], B=kd["B"], C=kd["C"], D=kd["D"])

    SHOW_IDX = 3
    for K in [K1, K2, K3]:
        print("Loop ---")
        print(K.B[SHOW_IDX])
        print(K.x)
        print(K.file)

    KK = K1 + K2
    KK = K1 + 2
    KK = 2 + K1
    print("***")
    KK = K1 * K2
    KK = 2 * K1
    KK = K1 * 2
    print("***")
    KK = K1.inv()
