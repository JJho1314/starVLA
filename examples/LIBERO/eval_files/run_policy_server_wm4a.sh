#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
# === Paths (LFT-W02 local) ===
STARVLA_DIR=/data/LFT-W02_data/junjie/VLA_WM/starVLA
LIBERO_HOME=/data/LFT-W02_data/junjie/LIBERO
STARVLA_PYTHON=/data/LFT-W02_data/.conda/envs/starVLA/bin/python
LIBERO_PYTHON=/data/LFT-W02_data/.conda/envs/libero/bin/python

# === Checkpoint ===
CKPT=${STARVLA_DIR}/playground/Checkpoints/1229_libero4in1_wm4a_cosmopredict2gr00t/final_model/pytorch_model.pt

export star_vla_python=${STARVLA_PYTHON}
your_ckpt=${CKPT}
gpu_id=${GPU_ID:-0}
port=${PORT:-6694}
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################
