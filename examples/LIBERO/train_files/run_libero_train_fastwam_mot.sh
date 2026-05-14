

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export MASTER_PORT=$((29500 + RANDOM % 1000))
###########################################################################################
# WanFastWAM_MoT: Wan2.2-TI2V backbone + ActionDiT (MoT-Lite) + video FM aux loss.
Framework_name=WanFastWAM_MoT
freeze_module_list=''
base_wm=/data/user/jhe724/workspace/weights/TI2V_5B
config_yaml=./examples/LIBERO/train_files/starvla_wm4a_libero_fastwam_mot.yaml
libero_data_root=/data/user/jhe724/workspace/data/libero_datasets/libero_lerobot
data_mix=libero_all
run_root_dir=./playground/Checkpoints
run_id=$(date +%Y%m%d)_libero4in1_${Framework_name}
###########################################################################################


output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
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
  --wandb_entity jjho1314 "$@"
