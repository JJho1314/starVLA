#!/bin/bash
#SBATCH --job-name=wm4a-bs128-hold
#SBATCH --partition=acd_u
#SBATCH --nodelist=ACD1-11
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/data/user/jhe724/workspace/starVLA/playground/Checkpoints/holder_bs128_%j.out

mkdir -p /data/user/jhe724/workspace/starVLA/playground/Checkpoints
echo "ALLOC HOLDER. JobID=$SLURM_JOB_ID Node=$SLURMD_NODENAME GPUs=$CUDA_VISIBLE_DEVICES"
echo "Date=$(date)"
nvidia-smi --query-gpu=index,memory.total --format=csv
echo "ssh into the node and run: tmux new -s train 'bash examples/LIBERO/train_files/run_libero_train_wm4a_bs128.sh'"
sleep 85800   # 23h50m, leaves 10m before walltime hard kill
