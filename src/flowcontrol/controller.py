"""Discrete-time linear state-space controller with ZOH one-step integration.

Classes
-------
Controller : continuous-time state-space system (subclass of control.StateSpace)
             with cached ZOH discretization, MIMO-compatible step(), and
             arithmetic operators that preserve the Controller type.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import control
import numpy as np
from numpy.typing import NDArray

from utils.lticontrol import read_matfile, ss_inv


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
        """Initialise a Controller from (A, B, C, D) matrices and an optional initial state.

        Parameters
        ----------
        A :
            System evolution matrix.
        B :
            Input matrix.
        C :
            Output matrix.
        D :
            Feedthrough matrix.
        file :
            Path from which the controller was loaded, if any.
        x0 :
            Initial internal state.  Defaults to zeros when ``None``.
        """
        super().__init__(A, B, C, D)
        self.file = file
        self.x = x0 if x0 is not None else np.zeros((self.nstates,))

    @classmethod
    def from_file(
        cls, file: Path, x0: Optional[NDArray[np.float64]] = None
    ) -> Controller:
        """Load a Controller from a ``.mat`` file.

        Parameters
        ----------
        file :
            Path to the ``.mat`` file containing A, B, C, D matrices.
        x0 :
            Initial internal state.  Defaults to zeros when ``None``.

        Returns
        -------
        Controller
        """
        stateSpaceMatrices = read_matfile(file)
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
        """Construct a Controller directly from (A, B, C, D) matrices.

        Parameters
        ----------
        A :
            System evolution matrix.
        B :
            Input matrix.
        C :
            Output matrix.
        D :
            Feedthrough matrix.
        file :
            Source file, if any, for bookkeeping.
        x0 :
            Initial internal state.  Defaults to zeros when ``None``.

        Returns
        -------
        Controller
        """
        return cls(A, B, C, D, x0=x0, file=file)

    def _discretize(self, dt: float) -> None:
        """Discretize the controller with ZOH at ``dt`` and cache the result.

        Parameters
        ----------
        dt :
            Discretization time step.
        """
        sysd = control.c2d(self, dt, method="zoh")
        self._Ad = sysd.A
        self._Bd = sysd.B
        self._Cd = sysd.C
        self._Dd = sysd.D
        self._dt = dt

    def step(self, y: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
        """Advance the controller by one time step using ZOH discretization.

        Discrete matrices are computed once and cached for a given ``dt``.
        MIMO-compatible.

        Parameters
        ----------
        y :
            Controller input (e.g. plant output measurement vector).
        dt :
            Simulation time step.

        Returns
        -------
        NDArray[np.float64]
            Control output ``u``.
        """
        if not hasattr(self, "_dt") or self._dt != dt:
            self._discretize(dt)
        y = np.atleast_1d(y)
        u = self._Cd @ self.x + self._Dd @ y
        self.x = self._Ad @ self.x + self._Bd @ y
        return u

    def reset(self) -> None:
        """Reset the internal state to zero."""
        self.x = np.zeros((self.nstates,))

    def __add__(self, other: Controller) -> Controller:
        return self._overload(other, super().__add__)

    def __radd__(self, other: Controller) -> Controller:
        return self._overload(other, super().__radd__)

    def __mul__(self, other: Controller) -> Controller:
        return self._overload(other, super().__mul__)

    def __rmul__(self, other: Controller) -> Controller:
        return self._overload(other, super().__rmul__)

    def inv(self) -> Controller:
        """Return the inverse of this controller (requires D to be invertible).

        Returns
        -------
        Controller
        """
        invK = ss_inv(self)
        return Controller(invK.A, invK.B, invK.C, invK.D)

    def _concatenate_states_with(self, other: Controller) -> NDArray[np.float64]:
        """Concatenate the internal states of this controller and ``other``.

        Parameters
        ----------
        other :
            The second Controller whose state is appended.

        Returns
        -------
        NDArray[np.float64]
            Concatenated state vector ``[self.x, other.x]``.
        """
        return np.concatenate((self.x, other.x), axis=0)

    def _overload(self, other: Controller, binary_op: Callable) -> Controller:
        """Apply a binary operation and cast the result back to Controller.

        Parameters
        ----------
        other :
            The right-hand operand (StateSpace or Controller).
        binary_op :
            The parent-class binary operation to delegate to.

        Returns
        -------
        Controller
            Result with concatenated internal state when ``other`` is a Controller.
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

    kd = read_matfile(sspath)
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
