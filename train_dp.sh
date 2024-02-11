#!/usr/bin/env bash
#SBATCH --time=10:00:00
#SBATCH --account=def-panos
#SBATCH --nodes 1
#SBATCH --gres=gpu:p100:2
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
module load apptainer

export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"

echo "Extracting dataset"
tar -C $SLURM_TMPDIR --strip-components=2 -S -xvf /scratch/sjsch/vimeo90k_triplet_uncomp.tar

srun apptainer \
    exec -B /scratch/sjsch -B $SLURM_TMPDIR --nv -W $SLURM_TMPDIR /scratch/sjsch/ubuntu.sif \
    python3 train.py \
    --epoch 300 \
    --batch_size 64 \
    --world_size "$((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES))"
