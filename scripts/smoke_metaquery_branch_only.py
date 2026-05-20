"""Isolated smoke test for the NEW code in WanMetaQueryFastWAM.

Bypasses the Wan2.2 backbone (which requires diffusers≥0.33's
``AutoencoderKLWan``) and validates ONLY the bits we added in this PR:

    1. Qwen3-VL load + tokenizer resize with BOI/EOI + N <|imgK|> tokens
    2. New token embedding row-mask hook (gradient leakage check)
    3. Connector (Qwen3 bidirectional encoder, 8 layers, ~870M)
    4. ``_encode_vlm_context`` produces ``[B, 64, 4096]``
    5. The concat-with-text path in our monkey-patch is wire-correct
       (we simulate it manually with dummy T5 text_embeds)

Run with: ``CUDA_VISIBLE_DEVICES=0 python scripts/smoke_metaquery_branch_only.py``
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Workaround for stale `mistral_common` version on this machine: stub out the
# broken module BEFORE transformers' lazy auto-class lookup hits it.
_stub = types.ModuleType("transformers.tokenization_mistral_common")
for _name in (
    "MistralCommonTokenizer", "MistralCommonBackend",
    "MistralCommonProcessor", "MistralCommonImageProcessor",
):
    setattr(_stub, _name, type(_name, (), {}))
sys.modules["transformers.tokenization_mistral_common"] = _stub

import numpy as np
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from transformers import AutoTokenizer, AutoImageProcessor, AutoVideoProcessor

from starVLA.model.framework.WM4A.WanMetaQueryFastWAM import (
    BOI_TOKEN, EOI_TOKEN, IMG_QUERY_TOKEN_FMT,
)
from starVLA.model.modules.projector.metaquery_connector import MetaQueryConnector


VLM_PATH = "/data/LFT-W02_data/junjie/weights/Qwen3-VL-4B-Instruct"
NUM_QUERIES = 64
OUT_DIM = 4096           # Wan ActionDiT.text_dim
CONN_LAYERS = 8


def main():
    device = "cuda"
    print(f"[smoke] device = {torch.cuda.get_device_name(0)}")

    # ── 1) Load Qwen3-VL + processor ─────────────────────────────────────────
    print(f"[smoke] loading VLM: {VLM_PATH}")
    vlm = Qwen3VLForConditionalGeneration.from_pretrained(
        VLM_PATH, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(VLM_PATH)
    image_processor = AutoImageProcessor.from_pretrained(VLM_PATH)
    try:
        video_processor = AutoVideoProcessor.from_pretrained(VLM_PATH)
    except Exception:
        video_processor = None
    # Read chat template from the model dir (Qwen3-VL ships it as a separate JSON).
    import json as _json
    with open(os.path.join(VLM_PATH, "chat_template.json"), "r") as f:
        chat_template = _json.load(f)["chat_template"]
    processor = Qwen3VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=chat_template,
    )
    print(f"[smoke] VLM hidden = {vlm.config.text_config.hidden_size}, "
          f"vocab_before = {len(tokenizer)}")
    old_vocab = len(tokenizer)

    # ── 2) Add tokens & resize embedding (mirror framework code) ─────────────
    img_tokens = [IMG_QUERY_TOKEN_FMT.format(i=i) for i in range(NUM_QUERIES)]
    all_new = [BOI_TOKEN, EOI_TOKEN] + img_tokens
    tokenizer.add_special_tokens({"additional_special_tokens": all_new})
    vlm.resize_token_embeddings(len(tokenizer))
    print(f"[smoke] added {len(all_new)} tokens, new vocab = {len(tokenizer)}")

    boi_id = tokenizer.convert_tokens_to_ids(BOI_TOKEN)
    eoi_id = tokenizer.convert_tokens_to_ids(EOI_TOKEN)

    # ── 3) Freeze VLM, then unfreeze just embed/lm_head with row-mask hook ──
    for p in vlm.parameters():
        p.requires_grad_(False)
    embed = vlm.get_input_embeddings()
    lm_head = vlm.get_output_embeddings()
    embed.weight.requires_grad_(True)
    if lm_head is not None:
        lm_head.weight.requires_grad_(True)

    mask = torch.zeros(embed.weight.shape[0], dtype=torch.bool, device=device)
    mask[old_vocab:] = True

    def _hook(grad):
        return grad * mask.view(-1, 1)

    embed.weight.register_hook(_hook)
    if lm_head is not None and lm_head.weight is not embed.weight:
        lm_head.weight.register_hook(_hook)

    # ── 4) Connector (Qwen3-encoder, 8 layers, heads=40) ─────────────────────
    vlm_hidden = vlm.config.text_config.hidden_size
    assert vlm_hidden % 64 == 0
    num_heads = vlm_hidden // 64
    print(f"[smoke] building connector: in={vlm_hidden}, out={OUT_DIM}, "
          f"layers={CONN_LAYERS}, heads={num_heads}")
    connector = MetaQueryConnector(
        in_dim=vlm_hidden, out_dim=OUT_DIM,
        num_layers=CONN_LAYERS, num_heads=num_heads, num_kv_heads=num_heads,
    ).to(device=device, dtype=torch.bfloat16)
    n_conn = sum(p.numel() for p in connector.parameters()) / 1e6
    print(f"[smoke] connector params = {n_conn:.1f}M")

    # ── 5) Build VLM prompt and run forward ──────────────────────────────────
    suffix = BOI_TOKEN + "".join(img_tokens) + EOI_TOKEN
    pil = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    messages = [[{
        "role": "user",
        "content": [
            {"type": "image", "image": pil},
            {"type": "text", "text": f"pick up the red block\n{suffix}"},
        ],
    }]]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, padding=True,
        add_generation_prompt=False, return_dict=True, return_tensors="pt",
    ).to(device)
    print(f"[smoke] VLM input_ids shape = {inputs['input_ids'].shape}")
    print(f"[smoke] BOI in input_ids: {(inputs['input_ids'] == boi_id).sum().item()}")
    print(f"[smoke] EOI in input_ids: {(inputs['input_ids'] == eoi_id).sum().item()}")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = vlm(**inputs, output_hidden_states=True, return_dict=True)
    h = out.hidden_states[-1]
    print(f"[smoke] VLM hidden_states[-1] = {tuple(h.shape)}")

    # ── 6) BOI..EOI slicing (MetaQuery extraction) ───────────────────────────
    input_ids = inputs["input_ids"]
    B, T = input_ids.shape
    boi_pos = (input_ids == boi_id).int().argmax(dim=1)
    eoi_pos = (input_ids == eoi_id).int().argmax(dim=1)
    col = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    span_mask = (col > boi_pos.unsqueeze(1)) & (col < eoi_pos.unsqueeze(1))
    n_span = span_mask.sum().item()
    print(f"[smoke] span (BOI..EOI) count = {n_span}, expected B*N = {B * NUM_QUERIES}")
    assert n_span == B * NUM_QUERIES, "tokenizer split <|imgK|> tokens unexpectedly!"
    q_hidden = h[span_mask].view(B, NUM_QUERIES, h.size(-1))
    print(f"[smoke] q_hidden = {tuple(q_hidden.shape)}")

    # ── 7) Connector forward ─────────────────────────────────────────────────
    vlm_ctx = connector(q_hidden)
    print(f"[smoke] connector(out) = {tuple(vlm_ctx.shape)}")
    assert vlm_ctx.shape == (B, NUM_QUERIES, OUT_DIM)

    # ── 8) Concat with fake T5 text embeds (simulates monkey-patch result) ──
    L_text = 32
    text_embeds = torch.randn(B, L_text, OUT_DIM, device=device, dtype=torch.bfloat16)
    text_mask = torch.ones(B, L_text, dtype=torch.bool, device=device)
    ctx = torch.cat([text_embeds, vlm_ctx.to(text_embeds.dtype)], dim=1)
    ctx_mask = torch.cat([text_mask, torch.ones(B, NUM_QUERIES, dtype=torch.bool, device=device)], dim=1)
    print(f"[smoke] final ctx = {tuple(ctx.shape)}, ctx_mask = {tuple(ctx_mask.shape)}")

    # ── 9) Backward smoke (just to confirm grad flow + row-mask hook) ────────
    loss = vlm_ctx.float().pow(2).mean()
    loss.backward()
    embed_grad = embed.weight.grad
    masked_rows = embed_grad[:old_vocab].abs().sum().item() if embed_grad is not None else 0
    unmasked_rows = embed_grad[old_vocab:].abs().sum().item() if embed_grad is not None else 0
    print(f"[smoke] embed grad on OLD rows (should be 0): {masked_rows:.4e}")
    print(f"[smoke] embed grad on NEW rows (should be >0): {unmasked_rows:.4e}")
    assert masked_rows == 0, "row-mask hook leaked gradient to old vocab rows!"

    conn_grad = sum(p.grad.abs().sum().item() for p in connector.parameters() if p.grad is not None)
    print(f"[smoke] connector grad sum (should be >0): {conn_grad:.4e}")

    print(f"\n[smoke] peak mem = {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print("[smoke] ✓ all checks passed")


if __name__ == "__main__":
    main()
