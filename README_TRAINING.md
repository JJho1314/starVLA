# starVLA 训练：数据路径与启动脚本

本文档汇总当前仓库中 **VLA / VLM / Co-train** 训练入口、**数据根目录约定**，以及 `examples/**/train_files` 下的 **bash 启动脚本**。实际路径请按本机磁盘与集群布局修改；YAML 中的默认值可被命令行 `--datasets.vla_data.data_root_dir` 等参数覆盖。

---

## 1. 训练入口（Python）

| 脚本 | 用途 |
|------|------|
| `starVLA/training/train_starvla.py` | 标准 VLA 训练（LeRobot 等） |
| `starVLA/training/train_starvlm.py` | 仅训练 VLM（多模态 LM） |
| `starVLA/training/train_starvla_cotrain.py` | VLA + VLM 联合训练 |

常用 Accelerate 启动方式见各 `run_*.sh`：`accelerate launch ... <入口脚本> --config_yaml <yaml> ...`。

---

## 2. 数据路径约定

### 2.1 VLA（LeRobot）— `datasets.vla_data.data_root_dir`

- **含义**：LeRobot 打包数据的**父目录**，其下为各子数据集目录（与 `data_mix` 中注册的名称对应）。
- **配置位置**：各示例 YAML 的 `datasets.vla_data.data_root_dir`，或在 shell 里通过 `--datasets.vla_data.data_root_dir` 传入（优先级以运行时为准）。

### 2.2 仓库内各示例 YAML 中的默认 `data_root_dir`（模板路径）

以下路径多为相对仓库根的 **`playground/Datasets/...`**，部署时需替换为你的真实数据目录或建软链接。

| 场景 / 示例目录 | YAML（节选） | 默认 `data_root_dir` |
|------------------|--------------|----------------------|
| LIBERO（cotrain） | `examples/LIBERO/train_files/starvla_cotrain_libero.yaml` | `playground/Datasets/LIBERO` |
| LIBERO（WM4A / FastWAM 等，本分支部分配置） | `examples/LIBERO/train_files/starvla_wm4a_libero.yaml` 等 | 含机器相关绝对路径，见下节 |
| CoTrainVLM 引用 LIBERO | `examples/CoTrainVLM/train_files/starvla_cotrain_libero.yaml` | `playground/Datasets/LEROBOT_LIBERO_DATA` |
| OXE / SimplerEnv | `examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml` | `playground/Datasets/OXE_LEROBOT_DATASET` |
| VLA-Arena | `examples/VLA-Arena/train_files/starvla_cotrain_vla_arena.yaml` | `playground/Datasets/VLA_ARENA_LEROBOT_DATA` |
| RoboTwin | `examples/Robotwin/train_files/starvla_cotrain_robotwin*.yaml` | `playground/Datasets/RoboTwin` |
| RoboTwin ARX | `examples/Robotwin/train_files/starvla_train_arx.yaml` | `playground/Datasets/arx-x5` |
| Calvin | `examples/calvin/train_files/starvla_train_calvin.yaml` | `playground/Datasets/calvin` |
| Robocasa 365 | `examples/Robocasa_365/train_files/starvla_qwenoft_robocasa365.yaml` | `playground/Datasets/robocasa365` |
| Robocasa tabletop / GR1 | `examples/Robocasa_tabletop/train_files/starvla_cotrain_robocasa_gr1.yaml` | `playground/Datasets/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim` |
| RoboChallenge table30v2 | `examples/RoboChallenge_table30v2/train_files/starvla_qwenoft_robochallenge_table30v2.yaml` | `playground/Datasets/RoboChallenge_table30v2` |
| DOMINO | `examples/DOMINO/train_files/starvla_train_domino.yaml` | `playground/Datasets/DOMINO` |
| Franka | `examples/Franka/train_files/starvla_cotrain_franka_*.yaml` | `./playground/Datasets` |
| Behavior | `examples/Behavior/starvla_cotrain_behavior.yaml` | `playground/Datasets/behavior-1k` |

### 2.3 本分支 LIBERO 相关 YAML 中出现的绝对路径（需与本机对齐）

部分 WM4A / WanFastWAM 配置写死了他人环境路径，使用前请改为你的路径或通过 CLI 覆盖：

- `/data/LFT-W02_data/junjie/data/libero_datasets`（例如 `starvla_wm4a_libero.yaml`）
- `/data/user/jhe724/workspace/data/libero_datasets/libero_lerobot`（若干 WM4A bs128 / OFT / FastWAM MoT 等 YAML）

Shell 示例：`examples/LIBERO/train_files/run_libero_train_wm4a.sh` 使用  
`libero_data_root=/data/LFT-W02_data/junjie/data/libero_datasets`。

### 2.4 VLM-only（`train_starvlm.py`）— LLaVA JSON 数据根

VLM 数据列表在 `starVLA/dataloader/qwenvl_llavajson/qwen_data_config.py`：

- `json_root`: `./playground/Datasets/LLaVA-OneVision-COCO/llava_jsons`
- `image_root`: `./playground/Datasets/LLaVA-OneVision-COCO/images`

注册名 `sharegpt4v_coco` 等指向上述目录下的标注与图片。

### 2.5 其它模板配置

- `starVLA/config/training/starvla_train_adapter.yaml`、`starvla_cotrain_libero.yaml`、`starvla_cotrain_oxe.yaml`：内含示例 `data_root_dir`，可作全局参考。

---

## 3. 训练脚本索引（`examples/**/train_files/*.sh`）

以下为仓库内 **启动训练的 bash 脚本**（路径相对于仓库根目录）。括号内为脚本里典型的入口或用途说明。

| 路径 | 说明 |
|------|------|
| `examples/LIBERO/train_files/run_libero_train.sh` | LIBERO，`train_starvla.py` |
| `examples/LIBERO/train_files/run_libero_train_wm4a.sh` | LIBERO + WM4A，`train_starvla.py` |
| `examples/LIBERO/train_files/run_libero_train_wm4a_oft.sh` | LIBERO + WM4A OFT |
| `examples/LIBERO/train_files/run_libero_train_wanfastwam.sh` | WanFastWAM 等 |
| `examples/LIBERO/train_files/run_libero_train_wanfastwam_aligned.sh` | WanFastWAM aligned |
| `examples/LIBERO/train_files/run_libero_train_fastwam_mot.sh` | FastWAM MoT |
| `examples/LIBERO/train_files/sbatch_holder_acd1_11.sh` | SLURM 占位/集群示例 |
| `examples/CoTrainVLM/train_files/run_libero_cotrain.sh` | `train_starvla_cotrain.py` |
| `examples/CoTrainVLM/train_files/run_train_starvlm.sh` / `run_starvlm_hpc3.sh` | VLM / HPC |
| `examples/CoTrainVLM/train_files/sbatch_starvlm_hpc3.sh` | VLM，SLURM |
| `examples/Robotwin/train_files/run_robotwin_train.sh` | RoboTwin |
| `examples/Robotwin/train_files/run_robotwin_train_batch.sh` | RoboTwin 批处理 |
| `examples/Robotwin/train_files/run_robotwin_hpc3.sh` | HPC3 单节点示例 |
| `examples/Robotwin/train_files/sbatch_robotwin_hpc3.sh` | SLURM + 数据目录软链说明 |
| `examples/calvin/train_files/run_calvin_train.sh` | Calvin |
| `examples/VLA-Arena/train_files/run_vla_arena_train.sh` | VLA-Arena |
| `examples/SimplerEnv/train_files/run_oxe_train.sh` | OXE |
| `examples/Robocasa_tabletop/train_files/run_robocasa.sh` | Robocasa tabletop |
| `examples/Robocasa_tabletop/train_files/submit_robocasa_training.sh` | 提交脚本 |
| `examples/Robocasa_365/train_files/run_robocasa365.sh` / `run_robocasa365_all.sh` | Robocasa 365 |
| `examples/RoboChallenge_table30v2/train_files/run_robochallenge_table30v2.sh` | Table30v2 |
| `examples/DOMINO/train_files/run_domino_train.sh` | DOMINO |
| `examples/Franka/train_files/run_franka_train_single.sh` / `run_franka_train_dual.sh` | Franka |
| `examples/Gemma4/submit_hpc3_libero.sh` | LIBERO，集群提交（若使用 Gemma4 流程） |

下载类辅助脚本（非训练主入口）：如 `examples/Robocasa_365/train_files/download_target_human.sh`、`examples/RoboChallenge_table30v2/train_files/download_table30v2.sh`。

---

## 4. 自检清单

1. **数据**：确认 `data_root_dir` 下目录结构与 `data_mix`、注册表一致。  
2. **权重**：各脚本中的 `base_vlm`、`base_wm` 等指向本地或缓存中的预训练权重。  
3. **输出**：`run_root_dir` / `run_id` 决定 checkpoint 目录（常见 `playground/Checkpoints` 或 `results/Checkpoints`）。  
4. **集群**：按需修改 NCCL 环境变量与 `accelerate` 的进程数、`deepspeed` 配置（`starVLA/config/deepseeds/deepspeed_zero2.yaml`）。
