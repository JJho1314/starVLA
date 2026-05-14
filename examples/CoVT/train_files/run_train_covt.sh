#!/usr/bin/env bash
set -e

# NCCL knobs identical to the rest of starVLA train scripts.
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000

###############################################################################
# === Please review these paths before launching ===
Framework_name=QwenCoVT
base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct
sam_ckpt=playground/Pretrained_models/sam_vit_h_4b8939.pth
config_yaml=./examples/CoVT/train_files/starvla_covt_qwen3vl.yaml
run_root_dir=./playground/Checkpoints
run_id=$(date +%m%d)_covt_qwen3vl_demo
###############################################################################


output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp "$0" "${output_dir}/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starcovt.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.covt.anchor_cfg.sam.checkpoint ${sam_ckpt} \
  --run_id ${run_id} \
  --run_root_dir ${run_root_dir}
