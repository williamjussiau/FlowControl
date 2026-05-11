import logging
from pathlib import Path

from examples.cavity.cavityflowsolver import CavityFlowSolver
from examples.cylinder.cylinderflowsolver import CylinderFlowSolver
from examples.lidcavity.lidcavityflowsolver import LidCavityFlowSolver
from examples.pinball.pinballflowsolver import PinballFlowSolver
from flowcontrol.operatorgetter import OperatorGetter
from utils.io import export_square_operators

logger = logging.getLogger(__name__)
cwd = Path(__file__).parent


def compute_operators_flowsolver(flowsolver, export):
    logger.info("Now computing operators...")
    opget = OperatorGetter(flowsolver)
    A, E, B, C = opget.get_all()

    if export:
        export_square_operators(
            path=cwd / "data_output",
            operators=[A, E],
            operators_names=["A", "E"],
        )

    return A, E, B, C


if __name__ == "__main__":
    COMPUTE_OPERATORS_CYLINDER = True
    COMPUTE_OPERATORS_CAVITY = False
    COMPUTE_OPERATORS_LIDCAVITY = False
    COMPUTE_OPERATORS_PINBALL = False

    if COMPUTE_OPERATORS_CYLINDER:
        fs = CylinderFlowSolver.make_default(
            Re=100, path_out=cwd / "cylinder" / "data_output"
        )
        fs.compute_steady_state(
            method="picard", max_iter=3, tol=1e-7, u_ctrl=[0.0, 0.0]
        )
        fs.compute_steady_state(
            method="newton", max_iter=25, u_ctrl=[0.0, 0.0], initial_guess=fs.fields.UP0
        )
        compute_operators_flowsolver(fs, export=True)

    if COMPUTE_OPERATORS_CAVITY:
        fs = CavityFlowSolver.make_default(
            Re=7500, path_out=cwd / "cavity" / "data_output"
        )
        fs.compute_steady_state(method="picard", max_iter=10, tol=1e-7, u_ctrl=[0.0])
        fs.compute_steady_state(
            method="newton", max_iter=10, u_ctrl=[0.0], initial_guess=fs.fields.UP0
        )
        compute_operators_flowsolver(fs, export=True)

    if COMPUTE_OPERATORS_LIDCAVITY:
        fs = LidCavityFlowSolver.make_default(
            Re=8000, path_out=cwd / "lidcavity" / "data_output"
        )
        fs.load_steady_state()
        compute_operators_flowsolver(fs, export=True)

    if COMPUTE_OPERATORS_PINBALL:
        fs = PinballFlowSolver.make_default(
            Re=50, path_out=cwd / "pinball" / "data_output"
        )
        fs.compute_steady_state(method="newton", max_iter=25, u_ctrl=[0.0, 0.0, 0.0])
        compute_operators_flowsolver(fs, export=True)
