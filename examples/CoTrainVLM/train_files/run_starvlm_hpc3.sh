

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export MASTER_PORT=$((29500 + RANDOM % 1000))

###########################################################################################
# VLM-only training: Qwen3-VL-4B-Instruct-Action on LLaVA-OneVision-COCO sharegpt4v subset.
Framework_name=QwenFast
freeze_module_list=''
base_vlm=/data/user/jhe724/workspace/weights/Qwen3-VL-4B-Instruct
config_yaml=./examples/CoTrainVLM/train_files/starvla_vlmonly_qwen3vl.yaml
run_root_dir=./playground/Checkpoints
run_id=$(date +%Y%m%d)_starvlm_qwen3vl4b
vlm_data=sharegpt4v_coco
###########################################################################################


output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/


accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${NUM_GPUS:-8} \
  --main_process_port $MASTER_PORT \
  starVLA/training/train_starvlm.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vlm_data.dataset_use ${vlm_data} \
  --datasets.vlm_data.per_device_batch_size 2 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 50000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 1000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_VLM \
  --wandb_entity jjho1314 "$@"
