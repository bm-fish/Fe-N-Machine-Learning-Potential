#!/bin/sh
#SBATCH -J 334_1kp4
#SBATCH -p L4cpu
#SBATCH -N 4
#SBATCH -n 512
#SBATCH -c 1
###SBATCH -t 41:00:00

set -e
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/deepmd-kit3b

#source /home/issp/materiapps/intel/lammps/lammpsvars-20220623update1-1.sh
module list
#lmp
#mpirun -np 128 lmp -in input #> stdout.log 2>&1
srun lmp_mpi -in input > stdout.log 2>&1
#cat log.lammps | sed -n "/Step/,/Loop time/p" | head -n-1 > thermo.out
