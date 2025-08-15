from examples.cylinder.compute_steady_state import Re
from examples.cylinder.batch_run_cylinder_forced import run_forced_simulation
from pathlib import Path
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

def run_single_simulation(args):
    """Wrapper function for parallel execution"""
    forcing_amplitude, forcing_frequency, Re, save_dir, num_steps, autonomous_dir = args

    print(f"Starting simulation freq={forcing_frequency:.3f}, amp={forcing_amplitude:.3f} in process {mp.current_process().name}")

    try:
        run_forced_simulation(Re, save_dir, num_steps, autonomous_dir, forcing_amplitude, forcing_frequency)
        print(f"✓ Completed simulation freq={forcing_frequency:.3f}, amp={forcing_amplitude:.3f}")
        return True
    except Exception as e:
        print(f"Error in simulation freq={forcing_frequency:.3f}, amp={forcing_amplitude:.3f}: {e}")
        return False

if __name__ == "__main__":
    base_dir = Path("/Users/jaking/Desktop/PhD/cylinder")
    parent_dir = base_dir / f"Re{Re}_forced_sweep"
    parent_dir.mkdir(parents=True, exist_ok=True)

    # Path to your autonomous simulation results
    autonomous_dir = base_dir / f"Re{Re}_autonomous" / "run1"

    forcing_frequencies = [0.5, 0.77, 1.0, 1.3] # In rad/s
    forcing_amplitudes = [0.2, 0.5, 0.8]
    # forcing_frequencies = np.linspace(0, np.pi, 8)
    # forcing_amplitudes = [0.5]
    num_steps = 20000

    # Prepare all simulation parameters
    simulation_args = []
    count = 1
    for f_amp in forcing_amplitudes:
        for freq in forcing_frequencies:
            save_dir = parent_dir / f"run{count}"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Write parameter log for each run
            log_path = save_dir / "run_parameters.txt"
            with open(log_path, "w") as f:
                f.write(f"Run directory: {save_dir}\n")
                f.write(f"Reynolds number: {Re}\n")
                f.write(f"Forcing frequency: {freq}\n")
                f.write(f"Forcing amplitude: {f_amp}\n")
                f.write(f"Number of steps: {num_steps}\n")
                f.write(f"Run index: {count}\n")
                f.write(f"Autonomous dir: {autonomous_dir}\n")
            
            # Store parameters for this simulation
            simulation_args.append((f_amp, freq, Re, save_dir, num_steps, autonomous_dir))
            count += 1

    # Determine number of processes (adjust based on your system)
    n_processes = min(mp.cpu_count() - 2, len(simulation_args))  # Leave 2 cores free
    print(f"Running {len(simulation_args)} simulations using {n_processes} processes")
    print(f"Total simulations: {len(forcing_frequencies)} frequencies × {len(forcing_amplitudes)} amplitudes = {len(simulation_args)}")

    # Run simulations in parallel
    with mp.Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(run_single_simulation, simulation_args),
            total=len(simulation_args),
            desc="Running forced cylinder simulations"
        ))

    # Report results
    successful = sum(results)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"Parallel execution completed!")
    print(f"Successful simulations: {successful}/{total}")
    print(f"Failed simulations: {total - successful}")
    
    if total - successful > 0:
        print(f"\nFailed simulation details:")
        for i, (success, args) in enumerate(zip(results, simulation_args)):
            if not success:
                f_amp, freq, _, save_dir, _, _ = args
                print(f"  - Frequency {freq:.3f}, amplitude {f_amp:.3f} (run{i+1})")
    
    print(f"{'='*60}")
    
    # Summary by frequency
    print(f"\nResults by frequency:")
    for freq in forcing_frequencies:
        freq_results = []
        for i, (success, args) in enumerate(zip(results, simulation_args)):
            if args[1] == freq:  # args[1] is frequency
                freq_results.append(success)
        
        successful_freq = sum(freq_results)
        total_freq = len(freq_results)
        print(f"  Frequency {freq:.3f}: {successful_freq}/{total_freq} successful")