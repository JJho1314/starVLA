"""Local smoke test for the freeze_modules fix.

Verifies that with ``freeze_modules='backbone.vae'`` (NOT including 'vlm'),
gradients can actually reach the newly-added VLM token embedding rows after
``trainer.freeze_backbones`` runs.

Steps:
    1. Build WanMetaQueryFastWAM (same flow as training).
    2. Apply ``TrainerUtils.freeze_backbones`` with the patched freeze pattern.
    3. Inspect ``requires_grad`` on:
       - new VLM token embedding rows (must be True)
       - some other VLM body param (must be False — frozen by __init__)
       - a Wan VAE param (must be False — frozen by freeze_backbones)
       - ActionDiT param (must be True — trainable)
       - Connector param (must be True — trainable)
    4. Force a backward through embed_tokens.weight and check that
       gradient flows only to the new rows (mask hook should zero out
       gradients on the old vocab rows).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load same config the HPC3 training uses, then patch paths to LFT-W02 local.
CFG_YAML = ROOT / "examples/MetaQueryFastWAM/train_files/starvla_wan_metaquery_fastwam_libero.yaml"
cfg = OmegaConf.load(str(CFG_YAML))

# LFT-W02 path overrides (same swaps we did for ckpt config.yaml earlier)
cfg.framework.world_model.base_wm = "/data/LFT-W02_data/junjie/weights/Wan2.2-TI2V-5B-Diffusers"
cfg.framework.qwenvl.base_vlm = "/data/LFT-W02_data/junjie/weights/Wan2.2-TI2V-5B-Diffusers"
cfg.framework.world_model.text_embed_cache_path = "/data/LFT-W02_data/junjie/weights/fastwam_text_cache_safetensors_libero"
cfg.framework.vlm.base_vlm = "/data/LFT-W02_data/junjie/weights/Qwen3-VL-4B-Instruct"
# Avoid downloading ActionDiT pretrained — for this test we don't need optimal init
cfg.framework.action_dit.skip_pretrained_load = True

cfg.run_root_dir = "."
cfg.run_id = "freeze_hook_test"

# Mimic CLI overrides from sbatch
cfg.trainer.freeze_modules = "backbone.vae"   # ← the fix

print(f"[test] config loaded, freeze_modules='{cfg.trainer.freeze_modules}'")

from starVLA.model.framework.base_framework import build_framework
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils

print(f"[test] building framework (will load Wan + Qwen3-VL) …")
model = build_framework(cfg)
print(f"[test] framework class: {type(model).__name__}")
print(f"[test] new tokens added: {len(model.img_query_token_ids)} <|imgK|> + BOI + EOI = "
      f"{model._old_vocab_size}..{model.qwen_vl_interface.processor.tokenizer.__len__() if hasattr(model, 'qwen_vl_interface') else 'n/a'}")
print(f"[test] _old_vocab_size = {model._old_vocab_size}")

# === Apply trainer freeze logic (same as training pipeline) ===
print(f"\n[test] running TrainerUtils.freeze_backbones(model, '{cfg.trainer.freeze_modules}') …")
model = TrainerUtils.freeze_backbones(model, freeze_modules=cfg.trainer.freeze_modules)

# === Inspect requires_grad after freeze ===
print(f"\n[test] === requires_grad inspection ===")

# 1. VLM embed_tokens — must be True (so gradients can flow; row-mask hook restricts to new rows)
embed = model.vlm.get_input_embeddings()
print(f"  vlm embed_tokens.weight.requires_grad = {embed.weight.requires_grad}  (expect True)")
assert embed.weight.requires_grad, "BUG: vlm embed_tokens got frozen — freeze_modules is still touching vlm"

# 2. VLM body (a deep transformer layer) — must be False (frozen by __init__)
deep_layer = model.vlm.model.language_model.layers[10]
some_proj = deep_layer.self_attn.q_proj
print(f"  vlm layer[10].q_proj.weight.requires_grad = {some_proj.weight.requires_grad}  (expect False)")
assert not some_proj.weight.requires_grad, "BUG: vlm body params not frozen by __init__"

# 3. Wan VAE — must be False (frozen by trainer.freeze_backbones)
try:
    vae_p = next(model.backbone.vae.parameters())
    print(f"  backbone.vae param.requires_grad = {vae_p.requires_grad}  (expect False)")
    assert not vae_p.requires_grad, "BUG: backbone.vae not frozen"
except (AttributeError, StopIteration):
    print(f"  backbone.vae: no params (text_embed_cache mode — skipping)")

# 4. ActionDiT — must be True
dit_p = next(model.action_expert.parameters())
print(f"  action_expert.parameters[0].requires_grad = {dit_p.requires_grad}  (expect True)")
assert dit_p.requires_grad, "BUG: ActionDiT got frozen"

# 5. MetaQuery connector — must be True
conn_p = next(model.connector.parameters())
print(f"  connector.parameters[0].requires_grad = {conn_p.requires_grad}  (expect True)")
assert conn_p.requires_grad, "BUG: connector got frozen"

# === Force a backward to verify row-mask hook ===
print(f"\n[test] === gradient hook verification ===")
print(f"[test] simulating backward via 'embed.weight.sum().backward()' …")
loss = embed.weight.sum()
loss.backward()
grad = embed.weight.grad
print(f"  embed.weight.grad shape: {tuple(grad.shape)}")

old_n = model._old_vocab_size
new_n = grad.shape[0] - old_n
old_rows = grad[:old_n]
new_rows = grad[old_n:]
print(f"  old vocab rows (n={old_n}):  abs.sum = {old_rows.abs().sum().item():.6f}  (expect ≈ 0 due to row mask)")
print(f"  new token rows (n={new_n}):  abs.sum = {new_rows.abs().sum().item():.6f}  (expect > 0)")

assert old_rows.abs().sum().item() == 0.0, \
    f"BUG: old vocab rows received gradient (sum={old_rows.abs().sum().item():.6e}); row-mask hook broken"
assert new_rows.abs().sum().item() > 0, \
    f"BUG: new token rows got zero gradient; check unfreeze in __init__"

print(f"\n[test] ✓ all checks passed — fix is working")
print(f"[test] Summary:")
print(f"  - vlm.embed_tokens is unfrozen ✓")
print(f"  - vlm body is frozen (from __init__) ✓")
print(f"  - Wan VAE is frozen (from freeze_backbones) ✓")
print(f"  - ActionDiT + connector are trainable ✓")
print(f"  - row-mask hook isolates gradient to new {new_n} tokens ✓")
