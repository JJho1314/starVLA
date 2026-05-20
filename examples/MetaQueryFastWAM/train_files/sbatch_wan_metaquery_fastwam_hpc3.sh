#!/bin/bash
#SBATCH --job-name=metaq-wanfastwam
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=3-00:00:00
#SBATCH --output=/data/user/jhe724/workspace/starVLA/playground/Checkpoints/metaq_wanfastwam_%j.out
#SBATCH --error=/data/user/jhe724/workspace/starVLA/playground/Checkpoints/metaq_wanfastwam_%j.err

mkdir -p /data/user/jhe724/workspace/starVLA/playground/Checkpoints

echo "=== Job Info ==="
echo "JobID: $SLURM_JOB_ID"
echo "Node:  $SLURMD_NODENAME"
echo "Date:  $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv | head

source /share/anaconda3/etc/profile.d/conda.sh
conda activate /data/user/jhe724/.conda/envs/starVLA
export PATH=/data/user/jhe724/.conda/envs/starVLA/bin:$PATH
cd /data/user/jhe724/workspace/starVLA

# W&B (HPC3 self-hosted instance)
export WANDB_MODE=online
export WANDB_BASE_URL="http://10.12.1.245:8080"
export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY must be set (export it in your env, e.g. ~/.bashrc)}"
export WANDB_ENTITY="jjho1314"
export WANDB_PROJECT="starVLA_Libero"

bash examples/MetaQueryFastWAM/train_files/run_train_wan_metaquery_fastwam_hpc3.sh

echo "=== Job Done $(date) ==="
