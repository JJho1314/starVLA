

# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
export NCCL_DEBUG=INFO
export NCCL_SOCKET_TIMEOUT_MS=360000
export MASTER_PORT=$((29500 + RANDOM % 1000))
###########################################################################################
# === Please modify the following paths according to your environment ===
# WM4A-OFT framework: Cosmos-Predict2 backbone + MLP L1 regression action head.
Framework_name=CosmoPredict2OFT
freeze_module_list=''
base_wm=/data/user/jhe724/workspace/weights/Cosmos-Predict2-2B-Video2World
config_yaml=./examples/LIBERO/train_files/starvla_wm4a_oft_libero.yaml
libero_data_root=/data/user/jhe724/workspace/data/libero_datasets/libero_lerobot
data_mix=libero_all
run_root_dir=./playground/Checkpoints
run_id=$(date +%Y%m%d)_libero4in1_${Framework_name}
# === End of environment variable configuration ===
###########################################################################################


# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${NUM_GPUS:-4} \
  --main_process_port $MASTER_PORT \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.world_model.base_wm ${base_wm} \
  --datasets.vla_data.data_root_dir ${libero_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 4 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Libero \
  --wandb_entity jjho1314 "$@" \
  # --is_debug True
