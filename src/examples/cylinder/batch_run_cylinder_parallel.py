from examples.cylinder.compute_steady_state import Re
from examples.cylinder.batch_run_cylinder import run_lidcavity_with_ic
from pathlib import Path
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

def run_single_simulation(args):
    """Wrapper function for parallel execution"""
    xloc, yloc, radius, amplitude, Re, save_dir, num_steps = args

    print(f"Starting simulation xloc={xloc:.3f}, yloc={yloc:.3f} in process {mp.current_process().name}")

    try:
        run_lidcavity_with_ic(Re, xloc, yloc, radius, amplitude, save_dir, num_steps)
        print(f"✓ Completed simulation xloc={xloc:.3f}, yloc={yloc:.3f}")
        return True
    except Exception as e:
        print(f"Error in simulation xloc={xloc:.3f}, yloc={yloc:.3f}: {e}")
        return False

if __name__ == "__main__":
    base_dir = Path("/Users/jaking/Desktop/PhD/cylinder")
    parent_dir = base_dir / f"Re{Re}"
    parent_dir.mkdir(parents=True, exist_ok=True)

    x_vals = np.linspace(4.0, 5.0, 3)
    y_vals = np.linspace(0.0, 0.1, 3)
    radius = 0.5
    amplitude = 0.1
    num_steps = 40000
    # x_vals = [2.0]
    # y_vals = [0.0]

    # Prepare all simulation parameters
    simulation_args = []
    count = 1
    for xloc in x_vals:
        for yloc in y_vals:
            save_dir = parent_dir / f"run{count}"
            save_dir.mkdir(parents=True, exist_ok=True)

            # Store parameters for this simulation
            simulation_args.append((xloc, yloc, radius, amplitude, Re, save_dir, num_steps))
            count += 1

    # Determine number of processes (adjust based on your system)
    n_processes = min(mp.cpu_count() - 2, len(simulation_args))  # Leave 2 cores free
    print(f"Running {len(simulation_args)} simulations using {n_processes} processes")

    # Run simulations in parallel
    with mp.Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(run_single_simulation, simulation_args),
            total=len(simulation_args),
            desc="Running cavity simulations"
        ))

    # Report results
    successful = sum(results)
    total = len(results)
    print(f"\n{'='*50}")
    print(f"Parallel execution completed!")
    print(f"Successful simulations: {successful}/{total}")
    print(f"Failed simulations: {total - successful}")
    print(f"{'='*50}")