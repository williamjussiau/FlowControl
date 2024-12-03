#!/bin/bash
#SBATCH --job-name=opt-nm    # -J nom-job      => nom du job
#SBATCH --ntasks=2          # -n 24           => nombre de taches (obligatoire)
#SBATCH --time 0-1:00         # -t 0-2:00       => duree (JJ-HH:MM) (obligatoire)
#SBATCH --qos=co_long_std       #                 => QOS choisie (obligatoire)
#SBATCH --output=slurm.%j.out # -o slurm.%j.out => Sortie standard
#SBATCH --error=slurm.%j.err  # -e slurm.%j.err => Sortie Erreur

# exclusive ?
cd $WORKDIR

# module purge
module load anaconda
conda activate fenics_pj 
export OMP_NUM_THREADS=1

# mpirun utilise les variables d'environnement inititalisees par Slurm
mpirun python $HOME/fenics-python/test/test_scipy_optimize/test_nm_par.py
