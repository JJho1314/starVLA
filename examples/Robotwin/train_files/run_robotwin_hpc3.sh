export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export MASTER_PORT=$((29500 + RANDOM % 1000))

###########################################################################################
# QwenOFT @ RoboTwin 2.0 (data-scaling: 50 tasks × 50 Clean + 500 Randomized)
#
# Reproduce paper config: global batch = 192 (paper: 6 nodes × 8 GPUs × per_device 4).
# On HPC3 single 8-GPU node we keep per_device_bs=4 and use grad_accum=6 to match.
# run_id is stable (no date) so SLURM time-out + re-sbatch resumes the same dir.
###########################################################################################
Framework_name=QwenOFT
freeze_module_list=''
base_vlm=/data/user/jhe724/workspace/weights/Qwen3-VL-4B-Instruct-Action
config_yaml=./examples/Robotwin/train_files/starvla_cotrain_robotwin_abs.yaml
run_root_dir=./playground/Checkpoints
data_mix=robotwin_all_50
run_id=robotwin_qwen3OFT_all_data50
###########################################################################################

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${NUM_GPUS:-8} \
  --main_process_port $MASTER_PORT \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.per_device_batch_size 4 \
  --datasets.vla_data.action_type abs_qpos \
  --datasets.vla_data.action_mode abs \
  --datasets.vla_data.data_mix ${data_mix} \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.gradient_accumulation_steps 6 \
  --trainer.max_train_steps 150000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 1000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Robotwin \
  --wandb_entity jjho1314 "$@"
