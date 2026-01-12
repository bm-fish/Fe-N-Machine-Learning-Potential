#!/bin/sh                                                                                                                                                                                                                                                                                                                    
#PBS -q F1accs
#PBS -l select=1:ncpus=16:ompthreads=1
#PBS -N test
#PBS -l walltime=02:40:00

#module purge
#module load PrgEnv-intel/8.3.3
#module load intel-mpi/2021.5

#make DEPS=1 -j 128 all
source /home/k0543/k054302/miniconda3/etc/profile.d/conda.sh
conda activate ~/deepmd-kit3b
export DP_INFER_BATCH_SIZE=32768

#CUDA_VISIBLE_DEVICES=0,1
echo $CUDA_VISIBLE_DEVICES
#horovodrun -np 2 dp train --mpi-log=workers input.json > ./stdlog 2>&1
dp --tf train input.json > ./stdlog 2>&1
#dp train input.json > ./stdlog 2>&1
#horovodrun -np 1 dp train --mpi-log=workers input.json > ./stdlog 2>&1
#mpirun -l -launcher=fork -hosts=localhost -np 1 dp train --mpi-log=workers input.json > ./stdlog 2>&1 
#horovodrun -np 1 dp train --mpi-log=workers input.json > ./stdlog 2>&1
# dp freeze -o graph.pb
# dp test -m graph.pb -s ../../database/10_fen_mp12120_dpmd_npy/ -n 20 -d results

# Freeze the model and output to a .pb file
dp --tf freeze -o "deepmd_potential.pb"