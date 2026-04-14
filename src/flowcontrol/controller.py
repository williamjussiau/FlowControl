from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import control
import numpy as np
from numpy.typing import NDArray

import utils.utils_flowsolver as flu
import utils.youla_utils as yu


class Controller(control.StateSpace):
    """Continuous-time linear state-space system, intended to be used as a controller.

    This class inherits from control.StateSpace and provides two additional
    attributes:
      * the _file_ from which a Controller was read,
      * the Controller internal state _x_.
    """

    def __init__(
        self,
        A: NDArray[np.float64],
        B: NDArray[np.float64],
        C: NDArray[np.float64],
        D: NDArray[np.float64],
        file: Optional[Path] = None,
        x0: Optional[NDArray[np.float64]] = None,
    ):
        """Initialize Controller with either (A,B,C,D) matrices
        or file, and initialize its state x. This method allows
        the construction of two static initializers.

        Args:
            A (NDArray[np.float64]): System evolution matrix.
            B (NDArray[np.float64]): System input matrix.
            C (NDArray[np.float64]): System output matrix.
            D (NDArray[np.float64]): System feedthrough.
            file (Path, optional): File from which the Controller is read. Defaults to None.
            x0 (Optional[NDArray[np.float64]], optional): Internal state. Defaults to None.
        """
        super().__init__(A, B, C, D)
        self.file = file
        self.x = x0 if x0 is not None else np.zeros((self.nstates,))

    @classmethod
    def from_file(
        cls, file: Path, x0: Optional[NDArray[np.float64]] = None
    ) -> Controller:
        """Initialize Controller from file only, with given inital state x0.

        Args:
            file (Path, optional): File from which the Controller is read. Defaults to None.
            x0 (Optional[NDArray[np.float64]], optional): Internal state. Defaults to None.

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
        A: NDArray[np.float64],
        B: NDArray[np.float64],
        C: NDArray[np.float64],
        D: NDArray[np.float64],
        file: Optional[Path] = None,
        x0: Optional[NDArray[np.float64]] = None,
    ) -> Controller:
        """Initialize Controller from matrices (A,B,C,D), with given initial state x0.

        Args:
            A (NDArray[np.float64]): System evolution matrix.
            B (NDArray[np.float64]): System input matrix.
            C (NDArray[np.float64]): System output matrix.
            D (NDArray[np.float64]): System feedthrough.
            file (Path, optional): File from which the Controller is read. Defaults to None.
            x0 (Optional[NDArray[np.float64]], optional): Internal state. Defaults to None.

        Returns:
            Controller
        """
        return cls(A, B, C, D, x0=x0, file=file)

    def _discretize(self, dt: float) -> None:
        """Discretize Controller with ZOH at the given time step and cache the result.

        Args:
            dt (float): Discretization time step.
        """
        sysd = control.c2d(self, dt, method="zoh")
        self._Ad = sysd.A
        self._Bd = sysd.B
        self._Cd = sysd.C
        self._Dd = sysd.D
        self._dt = dt

    def step(self, y: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
        """Advance Controller by one time step with input y using ZOH discretization.
        The discrete matrices are computed once and cached for a given dt. MIMO-compatible.

        Args:
            y (NDArray[np.float64]): Controller input (e.g. Plant output).
            dt (float): Time interval for simulation.

        Returns:
            NDArray[np.float64]: control output u.
        """
        if not hasattr(self, "_dt") or self._dt != dt:
            self._discretize(dt)
        y = np.atleast_1d(y)
        u = self._Cd @ self.x + self._Dd @ y
        self.x = self._Ad @ self.x + self._Bd @ y
        return u

    def __add__(self, other: Controller) -> Controller:
        return self._overload(other, super().__add__)

    def __radd__(self, other: Controller) -> Controller:
        return self._overload(other, super().__radd__)

    def __mul__(self, other: Controller) -> Controller:
        return self._overload(other, super().__mul__)

    def __rmul__(self, other: Controller) -> Controller:
        return self._overload(other, super().__rmul__)

    def inv(self) -> Controller:
        """Attempt to invert Controller provided that self.D not 0.
        Args:
            self (Controller): Controller to invert.

        Returns:
            Controller
        """
        invK = yu.ss_inv(self)
        return Controller(invK.A, invK.B, invK.C, invK.D)

    def _concatenate_states_with(self, other: Controller) -> NDArray[np.float64]:
        """Return a concatenatation of states from two Controller instances.

        Args:
            other (Controller): other Controller to concatenate states with.

        Returns:
            NDArray[np.float64]: concatenated internal states.
        """
        return np.concatenate((self.x, other.x), axis=0)

    def _overload(self, other: Controller, binary_op: Callable) -> Controller:
        """Shortcut for overriding all binary operations
        between Controller instances: use the super() operation
        and adapt the internal state and the file accordingly.

        Args:
            other (Controller): other StateSpace or Controller for binary op
            binary_op (_type_): binary operation to override

        Returns:
            Controller
        """
        K = binary_op(other)
        # Cast as Controller instead of StateSpace
        K = Controller(A=K.A, B=K.B, C=K.C, D=K.D)
        if isinstance(other, Controller):
            K.x = self._concatenate_states_with(other)
        # K.file stays None — derived controllers have no single file origin
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

    # Tests SISO/MIMO
    # Test SISO
    dt = 0.1
    num_steps = 5

    print("***** Test SISO *****")
    Ksiso = Controller.from_matrices(
        A=np.array([[1, 1, 1], [0.2, -1, 0], [0.0, 1.0, 1.0]]),
        B=np.array([[0], [1], [0.5]]),
        C=np.array([0.5, 0.2, 0]),
        D=0,
        x0=np.array([1, 2, 3]),
    )

    yy = np.asarray([1.2])
    for _ in range(num_steps):
        print("---")
        uu = Ksiso.step(yy, dt)
        print(f"output {uu}")
        print(f"states {Ksiso.x}")

    # Test MIMO
    print("***** Test MIMO *****")
    Kmimo = Controller.from_matrices(
        A=Ksiso.A,
        B=np.array([[0, 1], [1, 0], [0.5, 0.5]]),
        C=0.5 * np.eye(3),
        D=0,
        x0=np.array([1, 2, 3]),
    )

    yy = np.asarray([1.2, -1.3])
    for _ in range(num_steps):
        print("---")
        uu = Kmimo.step(yy, dt)  # type: ignore
        print(f"output {uu}")
        print(f"states {Kmimo.x}")
