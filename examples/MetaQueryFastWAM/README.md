# WanMetaQueryFastWAM — VLM-guided video+action joint world model

A MetaQuery-style add-on for [WanFastWAM](../../starVLA/model/framework/WM4A/WanFastWAM.py).
The goal: have a strong MLLM (Qwen3-VL-4B) inject high-level visual /
language semantics into Wan2.2's video DiT through cross-attention, so
better video latents propagate via MoT joint attention to ActionDiT and
improve action prediction.

## Design (Plan B)

```
            ┌─ UMT5-XXL ─→ text_embeds         [B, L_text, 4096]
instruction │
images ─┬─→ Wan VAE ────→ video latent         (clean latents to DiT)
        │
        └─→ Qwen3-VL ─→ N <|img_query|> hidden ─→ connector ─→ vlm_ctx
                                                                  [B, N, 4096]

ctx = concat([text_embeds, vlm_ctx, proprio_emb], dim=1)
                                                 │
              ┌──────────────────────────────────┼──────────────────────────────────┐
              │                                  │                                  │
        Wan video DiT.cross-attn          ActionDiT.cross-attn                 (MoT self-attn
        sees text + VLM + proprio         sees text + VLM + proprio            already shared)
```

Only the **append to context** step is new. Everything else — MoT joint
self-attention, video FM loss, action FM loss, proprio injection, KV-cache
inference — is inherited from `Wan_FastWAM` unmodified.

## File map

| File | Purpose |
|------|---------|
| `starVLA/model/modules/projector/metaquery_connector.py` | N×in_dim → N×4096 Qwen3 bidirectional encoder + MLP + RMSNorm |
| `starVLA/model/framework/WM4A/WanMetaQueryFastWAM.py` | subclass of `Wan_FastWAM`; loads Qwen3-VL, monkey-patches `backbone.build_inputs` |
| `examples/MetaQueryFastWAM/train_files/starvla_wan_metaquery_fastwam_libero.yaml` | LIBERO config |
| `examples/MetaQueryFastWAM/train_files/run_train_wan_metaquery_fastwam.sh` | 8×GPU launcher |

The training entry (`train_starvla.py`) is reused — no new entry-point
needed, because `forward`/`predict_action`/`predict_action_joint` work as-is
on top of the patched `build_inputs`.

## Freeze plan (matches MetaQuery)

```
✗  Qwen3-VL backbone           (frozen, MetaQuery prior)
✓  new <|img_query|> embedding rows (row-mask hook on VLM embed / lm_head)
✓  MetaQueryConnector          (random init)
✓  Wan video DiT               (adapts to new context)
✓  ActionDiT                   (adapts to new context)
✓  proprio_encoder             (unchanged)
✗  Wan VAE                     (frozen; same as parent)
✗  UMT5-XXL                    (frozen via FastWAM precomputed cache)
```

To LoRA-tune the VLM instead, set `framework.vlm.trainable: true` and
wrap the VLM with PEFT separately (not done here to keep the framework
self-contained).

## Run

### Local (LFT-W02, 2× A6000-48G)

```bash
bash examples/MetaQueryFastWAM/train_files/run_train_wan_metaquery_fastwam.sh
# Uses examples/MetaQueryFastWAM/train_files/starvla_wan_metaquery_fastwam_libero_local.yaml
```

### HPC3 (8× H100 / SLURM)

Canonical yaml — paths point to `/data/user/jhe724/workspace/...`:
```
examples/MetaQueryFastWAM/train_files/starvla_wan_metaquery_fastwam_libero.yaml
```

Submit:
```bash
# Set your W&B API key once
export WANDB_API_KEY=<your key>

# Submit the job
sbatch examples/MetaQueryFastWAM/train_files/sbatch_wan_metaquery_fastwam_hpc3.sh
```

What the SLURM job does:
1. Reserves 1 node × 8 GPU × 64 CPU × 512 GB RAM × 3 days on `acd_u`
2. Activates `/data/user/jhe724/.conda/envs/starVLA`
3. Exports W&B env (HPC3 internal instance at `10.12.1.245:8080`)
4. Calls `run_train_wan_metaquery_fastwam_hpc3.sh` which launches
   `accelerate launch ... --num_processes 8 starVLA/training/train_starvla.py`
5. Logs to `playground/Checkpoints/metaq_wanfastwam_<jobid>.{out,err}`

Pre-flight on HPC3 (one-time):
```bash
# Verify weights / data / dataset stats exist
ls /data/user/jhe724/workspace/weights/Wan2.2-TI2V-5B-Diffusers/transformer
ls /data/user/jhe724/workspace/weights/Qwen3-VL-4B-Instruct/model.safetensors.index.json
ls /data/user/jhe724/workspace/FastWAM/data/libero_mujoco3.3.2
ls /data/user/jhe724/workspace/FastWAM/runs/libero_uncond_2cam224_1e-4/2026-05-09_16-19-41/dataset_stats.json
ls /data/user/jhe724/workspace/starVLA/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
```

The first run will:
1. download Qwen3-VL-4B-Instruct (if not local),
2. register `<|begin_of_img|>`, `<|end_of_img|>`, `<|img_query|>` tokens
   and resize its embedding,
3. initialise the connector (random),
4. monkey-patch `backbone.build_inputs` so every parent method that calls
   it transparently gets the augmented context.

## Knobs

- `framework.vlm.num_queries` — number of `<|img_query|>` slots between
  BOI and EOI. MetaQuery uses 64–256; 64 is a cheap baseline that matches
  Wan's UMT5 sequence length scale.
- `framework.vlm.connector_num_layers` — connector depth. MetaQuery used
  24 (paired with frozen LLaVA-OV); 8 is fine when video DiT will keep
  fine-tuning.
- `trainer.freeze_modules` — comma-separated module prefixes. Default
  freezes `backbone.vae` and the entire `vlm`. Drop `vlm` to fine-tune
  the VLM (only sensible with LoRA).
- `trainer.learning_rate.connector` — separate LR for the random-init
  connector (5e-4 default), higher than 1e-4 used elsewhere.

## What the VLM "sees"

For every sample the prompt is:
```
<|im_start|>user
<image>
{task instruction}
<|begin_of_img|><|img0|><|img1|>...<|img{N-1}|><|end_of_img|><|im_end|>
```
N is `framework.vlm.num_queries`. Each `<|imgK|>` is registered as its
own special token with its own embedding row (matches MetaQuery's design,
`metaquery/models/model.py:165-167`), so every query slot starts with a
distinct identity at the input layer.

We forward this once, slice all hidden states strictly between BOI and
EOI (MetaQuery's extraction), run the connector, and append to Wan's
UMT5 context. The conditioning frame is `images[i][0]` (the first frame
in the chunk; same frame Wan uses for TI2V conditioning).
