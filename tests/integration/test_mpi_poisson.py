"""MPI smoke test: run demo_poisson.py under 4 MPI processes."""

import subprocess
import sys
from pathlib import Path

import pytest

DEMO = (
    Path(__file__).parent.parent.parent
    / "src"
    / "examples"
    / "mpitest"
    / "demo_poisson.py"
)


@pytest.mark.mpi
def test_demo_poisson_mpi(tmp_path):
    result = subprocess.run(
        ["mpirun", "-np", "4", sys.executable, str(DEMO)],
        capture_output=True,
        text=True,
        cwd=tmp_path,  # pvd/pvtu output files land here, not in the source tree
    )
    assert result.returncode == 0, (
        f"mpirun exited with code {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
