"""MPI smoke test: run demo_poisson.py under 2 MPI processes."""

import functools
import os
import subprocess
import sys
from pathlib import Path

import pytest

DEMO = Path(__file__).parent.parent.parent / "src" / "examples" / "mpitest" / "demo_poisson.py"


@functools.lru_cache(maxsize=None)
def _oversubscribe_flag() -> list[str]:
    """Return ['--oversubscribe'] if this mpirun supports it (OpenMPI), else [].

    MPICH/Hydra does not recognise --oversubscribe but allows oversubscription
    by default, so no flag is needed there.
    """
    probe = subprocess.run(
        ["mpirun", "--oversubscribe", "-np", "1", "true"],
        capture_output=True,
    )
    return ["--oversubscribe"] if probe.returncode == 0 else []


@pytest.mark.mpi
def test_demo_poisson_mpi(tmp_path):
    env = os.environ.copy()
    env["OMPI_MCA_rmaps_base_oversubscribe"] = "1"  # belt-and-suspenders for OpenMPI
    env["HYDRA_LAUNCHER"] = "fork"  # MPICH

    result = subprocess.run(
        ["mpirun", *_oversubscribe_flag(), "-np", "2", sys.executable, str(DEMO)],
        capture_output=True,
        text=True,
        cwd=tmp_path,  # pvd/pvtu output files land here, not in the source tree
        env=env,
    )
    assert result.returncode == 0, (
        f"mpirun exited with code {result.returncode}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
