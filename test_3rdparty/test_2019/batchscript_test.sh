#!/bin/bash
#SBATCH --job-name=f19-mpi    # -J nom-job      => nom du job
#SBATCH --ntasks=8           # -n 24           => nombre de taches (obligatoire)
#SBATCH --time 0-1:00         # -t 0-2:00       => duree (JJ-HH:MM) (obligatoire)
#SBATCH --qos=c5_test_giga        #                 => QOS choisie (obligatoire)
#SBATCH --output=slurm.%j.out # -o slurm.%j.out => Sortie standard
#SBATCH --error=slurm.%j.err  # -e slurm.%j.err => Sortie Erreur

# exclusive ?
cd $WORKDIR

module purge
module load anaconda
conda activate fenics_pj 

# export OMP_NUM_THREADS = 1

# mpirun utilise les variables d'environnement inititalisees par Slurm
# python /stck/wjussiau/fenics-python/ns/sipp/write_current_time.py
mpirun python $HOME/fenics-python/test_2019/test_python.py
# python /stck/wjussiau/fenics-python/ns/sipp/write_current_time.py
