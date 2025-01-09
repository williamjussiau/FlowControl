import control
import numpy as np
from pathlib import Path
import utils_flowsolver as flu
import youla_utils as yu


class Controller(control.StateSpace):
    x: np.array
    file: Path = None

    def __init__(self, A, B, C, D, x0=0, file=None):
        super().__init__(A, B, C, D)
        self.file = file
        self.x = x0
        if np.asarray(x0).all() == 0:
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
    def from_matrices(cls, A, B, C, D, x0=0, file=None):
        return cls(A, B, C, D, x0=x0, file=file)

    def step(self):
        self.x = self.x + 1

    def __add__(self, other):
        return self._overload(other, super().__add__)

    def __radd__(self, other):
        return self._overload(other, super().__radd__)

    def __mul__(self, other):
        return self._overload(other, super().__mul__)

    def __rmul__(self, other):
        return self._overload(other, super().__rmul__)

    def inv(self):
        invK = yu.ss_inv(self)
        return Controller(invK.A, invK.B, invK.C, invK.D)

    @staticmethod
    def concatenate_states(x1, x2):
        return np.concatenate((x1, x2), axis=0)

    def _overload(self, other, super_operator):
        K = super_operator(other)
        K = Controller(A=K.A, B=K.B, C=K.C, D=K.D)
        if isinstance(other, Controller):
            K.x = Controller.concatenate_states(self.x, other.x)
            K.file = self.file if (self.file == other.file) else None
        return K


if __name__ == "__main__":
    cwd = Path(__file__).parent
    sspath = cwd / "data_input" / "sysid_o16_d=3_ssest.mat"

    kd = flu.read_matfile(sspath)
    K1 = Controller.from_matrices(A=kd["A"], B=kd["B"], C=kd["C"], D=kd["D"])
    K2 = Controller.from_file(file=cwd / "data_input" / "sysid_o16_d=3_ssest.mat")
    K3 = Controller(A=kd["A"], B=kd["B"], C=kd["C"], D=kd["D"])

    # SHOW_IDX = 3
    # for K in [K1, K2, K3]:
    #     print("Loop ---")
    #     print(K.B[SHOW_IDX])
    #     print(K.x)
    #     print(K.file)

    print("+++")
    print(f"Sum test: {type(K1 + K2)}")
    print(f"Sum test: {type(K1 + 1)}")
    print(f"Sum test: {type(1 + K1)}")

    print("***")
    print(f"Multiplication test: {type(K1 * K2)}")
    print(f"Multiplication test: {type(2 * K1)}")
    print(f"Multiplication test: {type(K1 * 2)}")
    print(f"Inversion test: {type(K1.inv())}")
