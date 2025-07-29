from examples.lidcavity.compute_steady_state_increasing_Re import Re_final as Re
from examples.lidcavity.batch_run_lidcavity_eigvecs import run_lidcavity_with_eigenvector_ic
from pathlib import Path
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

def run_single_simulation(args):
    phase_angle, eigenvector_amplitude, forcing_frequency, forcing_amplitude, Re, save_dir, num_steps = args
    print(f"Starting simulation freq={forcing_frequency}, amp={forcing_amplitude:.4f} in process {mp.current_process().name}")
    try:
        run_lidcavity_with_eigenvector_ic(Re, phase_angle, eigenvector_amplitude, forcing_frequency, forcing_amplitude, save_dir, num_steps)
        print(f"✓ Completed simulation freq={forcing_frequency}, amp={forcing_amplitude:.4f}")
        return True
    except Exception as e:
        print(f"Error in simulation freq={forcing_frequency}, amp={forcing_amplitude:.4f}: {e}")
        return False

if __name__ == "__main__":
    base_dir = Path("/Users/jaking/Desktop/PhD/lid_driven_cavity")
    parent_dir = base_dir / f"Re{Re}_forced_0p01"
    parent_dir.mkdir(parents=True, exist_ok=True)

    phase_angles = np.linspace(0, 2*np.pi, 1)
    # eigenvector_amplitudes = [0.001]  # Post hopf
    eigenvector_amplitudes = [0.05]  # Pre hopf
    forcing_frequencies = np.linspace(0,6,7)
    # forcing_frequencies = [0]
    forcing_amplitudes = [0.001]
    # forcing_amplitudes = [0.0]
    num_steps = 80000

    simulation_args = []
    count = 1
    for phase_angle in phase_angles:
        for eig_amp in eigenvector_amplitudes:
            for freq in forcing_frequencies:
                for f_amp in forcing_amplitudes:
                    save_dir = parent_dir / f"run{count}"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    
                    log_path = save_dir / "run_parameters.txt"
                    with open(log_path, "w") as f:
                        f.write(f"Run directory: {save_dir}\n")
                        f.write(f"Reynolds number: {Re}\n")
                        f.write(f"Phase angle: {phase_angle}\n")
                        f.write(f"Eigenvector amplitude: {eig_amp}\n")
                        f.write(f"Forcing frequency: {freq}\n")
                        f.write(f"Forcing amplitude: {f_amp}\n")
                        f.write(f"Number of steps: {num_steps}\n")
                        f.write(f"Run index: {count}\n")

                    simulation_args.append((phase_angle, eig_amp, freq, f_amp, Re, save_dir, num_steps))
                    count += 1

    n_processes = min(mp.cpu_count() - 2, len(simulation_args))
    print(f"Running {len(simulation_args)} simulations using {n_processes} processes")

    with mp.Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(run_single_simulation, simulation_args),
            total=len(simulation_args),
            desc="Running forced eigenvector simulations"
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