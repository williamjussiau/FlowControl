#!/bin/bash
#SBATCH --job-name=cl-ms-fenics    # -J nom-job      => nom du job
#SBATCH --ntasks=24          # -n 24           => nombre de taches (obligatoire)
#SBATCH --time 0-24:00         # -t 0-2:00       => duree (JJ-HH:MM) (obligatoire)
#SBATCH --qos=c5_prod_giga # co_long_std # c5_long_giga  => QOS choisie (obligatoire)
#SBATCH --output=slurm_out/slurm.%j.out # -o slurm.%j.out => Sortie standard
#SBATCH --error=slurm_out/slurm.%j.err  # -e slurm.%j.err => Sortie Erreur
#SBATCH --exclusive

cd $WORKDIR

#module load impi/17
#module load openmpi/1.10.7
module purge
module load anaconda #/2020.11
conda activate fenics_pj 
export OMP_NUM_THREADS=1

# mpirun utilise les variables d'environnement inititalisees par Slurm
mpirun python $WORKDIR/fenics-python/cylinder/run_closedloop_ms.py



