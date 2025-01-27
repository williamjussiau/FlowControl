from __future__ import annotations
import control
import numpy as np
from pathlib import Path
import utils_flowsolver as flu
import youla_utils as yu


class Controller(control.StateSpace):
    """Continuous-time linear state-space system, intended to be used as a controller.

    This class inherits from control.StateSpace and provides two additional
    attributes:
      * the _file_ from which a Controller was read,
      * the Controller internal state _x_.
    """

    x: np.ndarray
    file: Path = None

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
        file: Path = None,
        x0: np.ndarray | None = None,
    ):
        """Initialize Controller with either (A,B,C,D) matrices
        or file, and initialize its state x. This method allows
        the construction of two static initializers.

        Args:
            A (np.ndarray): System evolution matrix.
            B (np.ndarray): System input matrix.
            C (np.ndarray): System output matrix.
            D (np.ndarray): System feedthrough.
            file (Path, optional): File from which the Controller is read. Defaults to None.
            x0 (np.ndarray | None, optional): Internal state. Defaults to None.
        """
        super().__init__(A, B, C, D)
        self.file = file
        self.x = x0
        if x0 is None:
            self.x = np.zeros((self.nstates,))

    @classmethod
    def from_file(cls, file: Path = None, x0: np.ndarray | None = None) -> Controller:
        """Initialize Controller from file only, with given inital state x0.

        Args:
            file (Path, optional): File from which the Controller is read. Defaults to None.
            x0 (np.ndarray | None, optional): Internal state. Defaults to None.

        Returns:
            Controller
        """
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
    def from_matrices(
        cls,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
        file: Path = None,
        x0: np.ndarray | None = None,
    ) -> Controller:
        """Initialize Controller from matrices (A,B,C,D), with given initial state x0.

        Args:
            A (np.ndarray): System evolution matrix.
            B (np.ndarray): System input matrix.
            C (np.ndarray): System output matrix.
            D (np.ndarray): System feedthrough.
            file (Path, optional): File from which the Controller is read. Defaults to None.
            x0 (np.ndarray | None, optional): Internal state. Defaults to None.

        Returns:
            Controller
        """
        return cls(A, B, C, D, x0=x0, file=file)

    def step(self, y: float, dt: float) -> float:
        """Simulate Controller from its current state self.x on the
        time interval [0, dt] with input y, to produce a control
        output u.

        Args:
            y (float): Controller input (e.g. Plant output).
            dt (float): Time interval for simulation.

        Returns:
            float: control output u.
        """
        # TODO make MIMO-compatible
        y_rep = np.repeat(np.atleast_2d(y), repeats=2, axis=0).T
        Tsim = [0, dt]
        _, yout, xout = control.forced_response(
            self, U=y_rep, T=Tsim, X0=self.x, interpolate=False, return_x=True
        )
        u = np.ravel(yout)[0]
        self.x = xout[:, 1]
        return u

    def __add__(self, other: Controller) -> Controller:
        return self._overload(other, super().__add__)

    def __radd__(self, other: Controller) -> Controller:
        return self._overload(other, super().__radd__)

    def __mul__(self, other: Controller) -> Controller:
        return self._overload(other, super().__mul__)

    def __rmul__(self, other: Controller) -> Controller:
        return self._overload(other, super().__rmul__)

    def inv(self: Controller) -> Controller:
        """Attempt to invert Controller provided that self.D not 0.
        Args:
            self (Controller): Controller to invert.

        Returns:
            Controller
        """
        invK = yu.ss_inv(self)
        return Controller(invK.A, invK.B, invK.C, invK.D)

    def _concatenate_states_with(self, other: Controller) -> np.ndarray:
        """Return a concatenatation of states from two Controller instances.

        Args:
            other (Controller): other Controller to concatenate states with.

        Returns:
            np.ndarray: concatenated internal states.
        """
        return np.concatenate((self.x, other.x), axis=0)

    def _overload(self, other: Controller, super_operator) -> Controller:
        """Shortcut for overriding all binary operations
        between Controller instances: use the super() operation
        and adapt the internal state and the file accordingly.

        Args:
            other (Controller): other StateSpace or Controller for binary op
            super_operator (_type_): binary operation to override

        Returns:
            Controller
        """
        K = super_operator(other)
        # Cast as Controller instead of StateSpace
        K = Controller(A=K.A, B=K.B, C=K.C, D=K.D)
        if isinstance(other, Controller):
            K.x = self._concatenate_states_with(other)
            K.file = self.file if (self.file == other.file) else None
        return K


if __name__ == "__main__":
    cwd = Path(__file__).parent
    sspath = (
        cwd / ".." / "examples" / "cylinder" / "data_input" / "sysid_o16_d=3_ssest.mat"
    )

    kd = flu.read_matfile(sspath)
    K1 = Controller.from_matrices(A=kd["A"], B=kd["B"], C=kd["C"], D=kd["D"])
    K2 = Controller.from_file(
        file=cwd
        / ".."
        / "examples"
        / "cylinder"
        / "data_input"
        / "sysid_o16_d=3_ssest.mat"
    )
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

    print("K1 : ", K1.file)
    print("K2 : ", K2.file)
    print("K3 : ", K3.file)
