#!/bin/bash
#SBATCH --job-name=eig-slepc  # -J nom-job      => nom du job
#SBATCH --ntasks=1          # -n 24           => nombre de taches (obligatoire)
#SBATCH --time 0-24:00         # -t 0-2:00       => duree (JJ-HH:MM) (obligatoire)
#SBATCH --qos=c5_prod_giga # co_long_bigmem #  c5_long_giga       #     => QOS choisie (obligatoire)
#SBATCH --output=slurm.%j.out # -o slurm.%j.out => Sortie standard
#SBATCH --error=slurm.%j.err  # -e slurm.%j.err => Sortie Erreur
#SBATCH --exclusive

cd $WORKDIR

module purge
#module load impi/17
module load anaconda/2020.11
conda activate eig_env
export OMP_NUM_THREADS=1

# mpirun utilise les variables d'environnement inititalisees par Slurm
mpirun python $WORKDIR/fenics-python/utils/eig/eig_utils.py
