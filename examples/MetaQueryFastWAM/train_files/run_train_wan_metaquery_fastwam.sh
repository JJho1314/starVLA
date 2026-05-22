#!/usr/bin/env bash
set -e

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export MASTER_PORT=$((29500 + RANDOM % 1000))

###############################################################################
# WanMetaQueryFastWAM: Wan2.2-TI2V + Qwen3-VL-guided cross-attn context.
# Reuses the existing VLA training entry (train_starvla.py): the framework's
# forward/predict_action are inherited from Wan_FastWAM untouched.
Framework_name=WanMetaQueryFastWAM
# Only freeze VAE. The VLM is frozen inside framework.__init__ (with row-mask
# hook keeping the new embed/lm_head rows trainable). Including 'vlm' here
# would re-freeze those rows after __init__. See commit 08a2ddd.
freeze_module_list='backbone.vae'
base_wm=/data/user/jhe724/workspace/weights/Wan2.2-TI2V-5B-Diffusers
base_vlm=./playground/Pretrained_models/Qwen3-VL-4B-Instruct
config_yaml=./examples/MetaQueryFastWAM/train_files/starvla_wan_metaquery_fastwam_libero.yaml
libero_data_root=/data/user/jhe724/workspace/FastWAM/data/libero_mujoco3.3.2
data_mix=libero_all_fastwam
run_root_dir=./playground/Checkpoints
run_id=$(date +%Y%m%d)_libero_${Framework_name}
###############################################################################

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp "$0" "${output_dir}/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${NUM_GPUS:-8} \
  --main_process_port $MASTER_PORT \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.world_model.base_wm ${base_wm} \
  --framework.vlm.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 2 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Libero \
  --wandb_entity jjho1314 "$@"
