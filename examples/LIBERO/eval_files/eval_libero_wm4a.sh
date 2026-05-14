#!/bin/bash
# === Paths (LFT-W02 local) ===
STARVLA_DIR=/data/LFT-W02_data/junjie/VLA_WM/starVLA

cd ${STARVLA_DIR}
# === Checkpoint ===
CKPT=${STARVLA_DIR}/playground/Checkpoints/1229_libero4in1_wm4a_cosmopredict2gr00t/final_model/pytorch_model.pt

###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME=/data/LFT-W02_data/junjie/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export LIBERO_Python=/data/LFT-W02_data/.conda/envs/libero/bin/python

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

host="127.0.0.1"
base_port=${PORT:-6694}
unnorm_key="franka"
your_ckpt=${CKPT}

# export DEBUG=true

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# model_root: playground/Checkpoints/<run_id>
model_root=$(echo "$your_ckpt" | awk -F'/checkpoints/' '{print $1}')
if [[ "$model_root" == "$your_ckpt" ]]; then
    # ckpt is in final_model/, not checkpoints/
    model_root=$(dirname $(dirname "$your_ckpt"))
fi
# === End of environment variable configuration ===
###########################################################################################

task_suite_name=${TASK_SUITE:-libero_goal}
num_trials_per_task=${NUM_TRIALS:-50}
video_out_path="${model_root}/results/${task_suite_name}/${folder_name}"

${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path"
