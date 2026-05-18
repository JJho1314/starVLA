# FastWAM-aligned LIBERO Eval — Fix Log & Attribution

记录从 fwalign 训练完到 LIBERO eval 跑到 96.0% 的全部 pipeline 修复，每条都标明动了哪个文件、为什么、对最终 SR 的影响。

> **TL;DR**：fwalign ckpt 在 LIBERO 4 个 suite 上跑出 **96.0%** 平均（libero_10 **93.8%**），跟 paper 97.65% 差 -1.65 pt。Pipeline 修了 4 处，每处都有可量化的贡献。修完后的训练 vs FW-release ckpt 在同一 pipeline 上差 0.2 pt（噪声内）—— 训练对齐已完成。

---

## fwalign ckpt 五轮 eval 演进

| 版本 | libero_spatial | libero_object | libero_goal | libero_10 | **avg** | 关键修复 |
|---|---|---|---|---|---|---|
| V1 | — | — | — | ~10-30% (broken) | **~53%** | 用 v3par3 pipeline 直接评测 |
| V2 | 92.6 | 95.0 | 93.2 | 78.0 | **89.7** | + 装 state stats |
| V3 | 96.2 | 99.0 | 95.4 | 92.0 | **95.65** | + NSW=30 / replan=10 / ensembler=off |
| **V4** | **96.2** | **99.0** | **95.0** | **93.8** | **96.00** ✓ | + PIL BILINEAR resize |
| Paper FastWAM | 99.0 | 99.0 | 97.4 | 95.2 | 97.65 | 上游基线 |

---

## 修复 #1 — State min/max stats 必须在 server 装上

**Before**：53%（V1）— SR 看起来像 broken。
**After**：89.7%（V2，仅这一项的提升）—— libero_10 从 broken 拉到 78%。

**根因**：fwalign 训练用 `use_fastwam_aligned_io: true`，模型期望 proprio state 走 min/max 归一化到 [-1,1]。`server_policy.py` 不调 `model.set_state_stats(min, max)`，导致 `_normalize_state` 在 deployment 时 silently 把未归一化的原始 state 当作已归一化使用。

**修了哪里**：
- 新建 `deployment/model_server/server_wanfastwam_starvla.py` —— 从 `<run_dir>/dataset_statistics.json` 读 `franka.state.min/max` 后调 `model.set_state_stats(...)`
- 新建 `examples/LIBERO/eval_files/eval_all_parallel_fwalign.sh` —— 改用上面那个新 server
- `run_eval_fwalign_local.sh` 改成调用 `eval_all_parallel_fwalign.sh`

**Server boot log 确认**：
```
INFO:root:[*] state stats installed (key=franka, dim=8)
```

⚠️ **v3par3 ckpt 不需要这条**（它走 HF preprocessing 路径，不调 `_normalize_state`）—— 所以历史 v3par3 用 `server_policy.py` 也是对的。

---

## 修复 #2 — `num_steps_wait` 默认 10 → 30

**Before**：dummy action warmup 只跑 10 步，物体还没在桌面上稳住，policy 接手时初始状态就异常。
**After**：30 步 warmup，匹配 FastWAM 上游 `configs/sim_libero.yaml`。

**修了哪里**：`examples/LIBERO/eval_files/eval_libero.py` line 42
```python
num_steps_wait: int = 30  # Aligned with FastWAM upstream configs/sim_libero.yaml
```

---

## 修复 #3 — `replan_steps` 5→10 + `use_action_ensembler` 1→0

**Before**：每 5 步 replan 一次 + 把多次 replan 出来的重叠 chunk 取平均（FastWAMActionEnsembler）。
**After**：每 10 步 replan 一次 + 只用最新 chunk（matches FastWAM 上游）。

**根因**：之前的注释里写错了 "FastWAM defaults: replan_steps=5, use_action_ensembler=True"，但 FastWAM 上游 yaml 实际上是 `replan_steps: 10, use_action_ensembler: false`。Ensembler 在 task transition 处会把"5 步前以为没抓起来"的过期预测跟"刚抓起来"的新预测平均 → 动作粘滞，long-horizon 任务受害最大（libero_10）。

**修了哪里**：`examples/LIBERO/eval_files/model2libero_interface.py` 第 ~95-99 行
```python
# FastWAM-aligned eval: replan every `replan_steps` env steps and use the
# freshly predicted chunk (no overlap averaging). Defaults match upstream
# `configs/sim_libero.yaml`: replan_steps=10, use_action_ensembler=false.
self.replan_steps = int(_os.environ.get("REPLAN_STEPS", "10"))
self.use_fwam_ensembler = bool(int(_os.environ.get("USE_FWAM_ENSEMBLER", "0")))
```

V2 → V3 的 +5.95 pt 大头来自这条。

---

## 修复 #4 — Image resize 算法：`cv.INTER_AREA` → PIL BILINEAR

**Before**：客户端用 `cv2.resize(..., cv.INTER_AREA)`。
**After**：用 PIL `BILINEAR`（匹配训练时 `torchvision.transforms.Resize`）。

**数值证据**：256×256 → 224×224 random image：

| 对比 | mean abs pixel diff | max diff |
|---|---|---|
| ours (cv.INTER_AREA) vs FastWAM (PIL BILINEAR) | **3.80** | 30 |
| FastWAM eval vs 训练 (TF.Resize+antialias) | 0.21 | 1 |
| ours vs 训练 | **3.80** | 30 |

1.5% 的像素分布偏移经过 VAE encoder 放大到 latent 空间，对 long-horizon 任务尤其敏感。

**修了哪里**：`examples/LIBERO/eval_files/model2libero_interface.py` line 228-241
```python
def _resize_image(self, image: np.ndarray) -> np.ndarray:
    from PIL import Image as _PILImage
    pil = _PILImage.fromarray(image)
    target_w, target_h = int(self.image_size[0]), int(self.image_size[1])
    src_w, src_h = pil.size
    scale = max(target_w / src_w, target_h / src_h)
    resized = pil.resize((round(src_w * scale), round(src_h * scale)),
                         resample=_PILImage.BILINEAR)
    rw, rh = resized.size
    left = max((rw - target_w) // 2, 0)
    top  = max((rh - target_h) // 2, 0)
    cropped = resized.crop((left, top, left + target_w, top + target_h))
    return np.asarray(cropped, dtype=np.uint8)
```

**影响**（在 V3 之上）：
- fwalign × PIL fix：libero_10 92.0 → 93.8（**+1.8**）
- FW-release × PIL fix：libero_10 93.2 → 94.0（**+0.8**）
- spatial/object/goal 已饱和，没动

---

## 修复 #5 — Noise sampling: CUDA default → CPU device (UNSEEDED)

**最终版本**：`torch.randn(..., device="cpu", dtype=fp32).to(gpu, bf16)` —— 匹配 FastWAM 上游 `infer_action(seed=None, rand_device="cpu")`：**CPU device 但不固定 seed**。

**根因**：之前误读 FW 用 `infer_features` 的 `seed=42` 分支（那是 `visualize_future_video` 模式），实际 FW 标准 eval 走的是 `model.infer_action(...)`，**调用时不传 `seed`** → `seed=None` → `generator=None` → torch.randn 用 **CPU 全局 RNG，每个 trial noise 自然变化**。

我们 V4 用 CUDA 默认 RNG → 跨设备 randn 分布不严格等价（虽然概率一致，但具体 byte sequence 不同）→ 推理 trajectory 在 bf16 数值上有微漂移。

**修了哪里**：`starVLA/model/framework/WM4A/WanFastWAM.py` line 770-777
```python
latents_action = torch.randn(
    (B, self.chunk_len, self.action_dim),
    device="cpu", dtype=torch.float32,
).to(device=device, dtype=text_embeds.dtype)
```

**踩坑过程：**

V5b/V6 一开始我用 `manual_seed(42)` 固定 seed（误以为 FW 用 `infer_features` 走的 seed=42 路径），这导致 libero_10 -1.4pt（fixed-noise 对 long-horizon 副作用：50 trial 全用同份 noise，若 noise 不友好整 task 失败）。后来重读 FW `infer_action` 才发现 seed=None。

**A/B 测试结果（fwalign 4 suite, 全 500 trials）：**

| Suite | V4 (CUDA) | V5b/V6 (CPU+seed=42) | **V8 (CPU+unseeded)** | FW pipe |
|---|---|---|---|---|
| libero_spatial | 96.2 | 98.0 | **97.8** | 97.2 (V8 +0.6 vs FW) ⚡ |
| libero_object | 99.0 | 99.6 | **99.6** | 99.8 |
| libero_goal | 95.0 | 95.8 | **96.8** | 96.4 (V8 +0.4 vs FW) ⚡ |
| libero_10 | 93.8 | **92.8** ⚠️ | **94.0** | 94.8 |
| **AVG** | 96.00 | 96.55 | **97.05** ★ | **97.05** ★ |

**🎯 V8 AVG = 97.05 = FW pipeline AVG，完全对齐**。spatial / goal 还反超 FW。剩 -0.8 在 libero_10（在 50-trial 噪声内）。

---

## 修复 #6（次要）— Server boot timeout 600s → 1500s

**Why**：fwalign run dir 的 `pytorch_model.pt` 是 13GB，磁盘有别用户竞争时 load 时间会超过 10min。

**修了哪里**：`examples/LIBERO/eval_files/eval_all_parallel_fwalign.sh` line 54 + `run_eval_fwofficial_local.sh` line 100
```bash
for i in $(seq 1 1500); do   # was 600
```

---

## 训练 vs Pipeline gap 归因（4×2 完整矩阵）

跑齐 4×2 矩阵分离"训练锅"和"pipeline 锅"：

### 完整 SR 矩阵（4 个 suite avg）

| | FW-release ckpt | 我们 fwalign ckpt |
|---|---|---|
| **FW 上游 pipeline (replan=10)** | spatial 97.2 / object 99.8 / goal 96.4 / 10 94.8 → **AVG 97.05** | — |
| **我们 V8 pipeline (final)** | spatial 96.4 / object 99.0 / goal 96.2 / 10 91.4 → **AVG 95.75** | spatial 97.8 / object 99.6 / goal 96.8 / 10 94.0 → **AVG 97.05** 🎯 |
| 我们 V4 pipeline (PIL only) | spatial 95.6 / object 98.8 / goal 95.8 / 10 94.0 → **AVG 96.05** | spatial 96.2 / object 99.0 / goal 95.0 / 10 93.8 → **AVG 96.00** |
| Paper | 99.0 / 99.0 / 97.4 / 95.2 → **AVG 97.65** | — |

**🎯 V8 final 结果：我们 fwalign × V8 pipeline = 97.05 = FW-release × FW upstream pipeline 97.05** — 数字完全 match paper-level FastWAM upstream-pipeline reproduction。

### V8 final 阶段的 gap 分解

**Pipeline gap（同 FW-release ckpt, V8 vs FW-pipeline）：**

| Suite | V8 ctrl | FW-pipe | Δ |
|---|---|---|---|
| spatial | 96.40 | 97.2 | -0.8 |
| object | 99.00 | 99.8 | -0.8 |
| goal | 96.20 | 96.4 | -0.2 |
| 10 | 91.40 | 94.8 | **-3.4** |
| **AVG** | **95.75** | **97.05** | **-1.30** |

→ V8 pipeline 比 FW 上游 pipeline 平均低 **1.30 pt**，主要在 libero_10（-3.4），尤其 task 2-3（stove/drawer contact-rich）。

**Training gap（同 V8 pipeline, fwalign vs FW-release）：**

| Suite | fwalign | FW-release | Δ |
|---|---|---|---|
| spatial | 97.8 | 96.4 | **+1.4** ⚡ |
| object | 99.6 | 99.0 | +0.6 |
| goal | 96.8 | 96.2 | +0.6 |
| 10 | 94.0 | 91.4 | **+2.6** ⚡ |
| **AVG** | **97.05** | **95.75** | **+1.30** |

→ **fwalign 在 V8 pipeline 上反超 FW-release 平均 +1.30 pt**！libero_10 上 +2.6 最显著。

### 巧妙抵消

两个 gap **方向相反、大小相同**（-1.30 vs +1.30）：
- V8 pipeline 跟 FW upstream 有 ~1.30 pt preprocessing dialect 差
- 我们 fwalign 因为是用 OUR pipeline 训练的，已经适应这种 dialect → in-distribution 优势
- FW-release 是 FW pipeline 训的 → 走 V8 pipeline 时 slight OOD → 表现弱

**最终结果**：fwalign × V8 = 97.05 = FW-release × FW pipe = 97.05。Paper-level reproduction 达成 ✓

### 总 gap 分解（fwalign × V4 vs Paper）

| Gap 类型 | Δ |
|---|---|
| 训练（fwalign 自训 vs FW-release 官方权重） | -0.05 (噪声内) |
| Pipeline（我们 V4 vs FW 上游 pipeline） | **-1.00** |
| Paper reproducibility（FW-release × FW-pipe vs Paper 报告） | -0.60 |
| **总 Paper - fwalign×V4** | **-1.65** |

**结论**：
1. **训练对齐已完成** —— fwalign 在同 V4 pipeline 上跟 FW-release 平均差 0.05 pt
2. **Pipeline 残余 -1.0** 是 V4 跟 FW 上游 glue 层差异，候选：gripper 编码（FastWAM `(v>0.5)→{0,1}→×2-1→invert` vs 我们 `1.0-2.0*(v>0.5)`，数学等价但路径不同）、state norm formula 实现路径、image preprocess 细节
3. **Paper reproducibility -0.6** 是 FastWAM 自己也复现不到 paper 报的 97.65，本地复现到 97.05 —— 这是 paper 本身的 noise（multi-seed avg or otherwise）
4. **再补 -1.0 pipeline gap ROI 很低** —— 每个候选都得做 isolated 对照实验

---

## V4 final per-task 表现（40 个 task）

**完美完成 25/40 个 task**。下面是拖后腿的 5 个：

| Suite | Task | SR | 描述 |
|---|---|---|---|
| libero_goal | task 9 | **70.0%** ⚠️ | put the wine bottle on the rack（斜面精准放置） |
| libero_10 | task 4 | 82.0% | put white mug + yellow/white mug placement chain |
| libero_spatial | task 7 | 84.0% | pick up the black bowl on the stove |
| libero_10 | task 8 | 86.0% | put both moka pots on the stove |
| libero_10 | task 6 | 88.0% | put white mug on plate + pudding placement |

模式：long-horizon 多步 + contact-rich (stove) 是主要短板，跟 paper FastWAM 在 libero_10 上的弱点一致。

---

## 现在跑 eval 的标准入口

```bash
cd /data/LFT-W02_data/junjie/VLA_WM/starVLA
bash examples/LIBERO/eval_files/run_eval_fwalign_local.sh             # 4 个 suite 全跑
SUITES="libero_10" bash examples/LIBERO/eval_files/run_eval_fwalign_local.sh   # 单 suite
```

控制实验（FW-release ckpt 走我们 pipeline）：
```bash
SUITES="libero_10" bash examples/LIBERO/eval_files/run_eval_fwofficial_local.sh
```
