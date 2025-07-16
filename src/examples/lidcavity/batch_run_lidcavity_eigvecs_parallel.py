from examples.lidcavity.compute_steady_state_increasing_Re import Re_final as Re
from examples.lidcavity.batch_run_lidcavity_eigvecs import run_lidcavity_with_eigenvector_ic
from pathlib import Path
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

def run_single_simulation(args):
    """Wrapper function for parallel execution"""
    phase_angle, eigenvector_amplitude, Re, save_dir, num_steps = args
    
    print(f"Starting simulation eig_idx={phase_angle}, amp={eigenvector_amplitude:.4f} in process {mp.current_process().name}")
    
    try:
        run_lidcavity_with_eigenvector_ic(Re, phase_angle, eigenvector_amplitude, save_dir, num_steps)
        print(f"✓ Completed simulation eig_idx={phase_angle}, amp={eigenvector_amplitude:.4f}")
        return True
    except Exception as e:
        print(f"Error in simulation eig_idx={phase_angle}, amp={eigenvector_amplitude:.4f}: {e}")
        return False

if __name__ == "__main__":
    # Adapt to wherever you want to save the results
    base_dir = Path("/Users/james/Desktop/PhD/lid_driven_cavity")
    parent_dir = base_dir / f"Re{Re}_test"
    parent_dir.mkdir(parents=True, exist_ok=True)

    # Define parameter ranges for eigenvector simulations
    phase_angles = np.linspace(0,2*np.pi, 9)
    amplitudes = [0.01]  # Different perturbation amplitudes
    num_steps = 200
    
    # Prepare all simulation parameters
    simulation_args = []
    count = 1
    
    for phase_angle in phase_angles:
        for amp in amplitudes:
            save_dir = parent_dir / f"run{count}"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Store parameters for this simulation
            simulation_args.append((phase_angle, amp, Re, save_dir, num_steps))
            count += 1
    
    # Determine number of processes (adjust based on your system)
    n_processes = min(mp.cpu_count() - 2, len(simulation_args))  # Leave 2 cores free
    print(f"Running {len(simulation_args)} simulations using {n_processes} processes")
    print(f"Total simulations: {len(phase_angles)} eigenvectors × {len(amplitudes)} amplitudes = {len(simulation_args)}")
    
    # Run simulations in parallel
    with mp.Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(run_single_simulation, simulation_args),
            total=len(simulation_args),
            desc="Running eigenvector simulations"
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
                phase_angle, amp, _, save_dir, _ = args
                print(f"  - Eigenvector {phase_angle}, amplitude {amp:.3f} (run{i+1})")
    
    print(f"{'='*60}")
    
    # Summary by eigenvector
    print(f"\nResults by eigenvector:")
    for phase_angle in phase_angles:
        eig_results = []
        for i, (success, args) in enumerate(zip(results, simulation_args)):
            if args[0] == phase_angle:
                eig_results.append(success)
        
        successful_eig = sum(eig_results)
        total_eig = len(eig_results)
        print(f"  Angle {phase_angle}: {successful_eig}/{total_eig} successful")