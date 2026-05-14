"""SAM teacher wrapper for CoVT-style mask token decoding.

The 8 learnable ``<|sam_pad|>`` tokens are projected to SAM's
prompt embedding dim (256) and fed to the *frozen* SAM mask_decoder
together with the SAM image embedding. The output masks are then
trained against ground-truth masks via Hungarian matching.

Requires the ``segment_anything`` package and a SAM checkpoint
(default ViT-H ``sam_vit_h_4b8939.pth``).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class SamAnchor(nn.Module):
    """Frozen SAM teacher used as a mask decoder for CoVT seg tokens.

    Public API:
        encode_image(pil)  → image_embedding ``[1, 256, 64, 64]``
        decode_with_tokens(emb, pil, tokens)  → masks ``[N, H, W]``
    """

    def __init__(
        self,
        checkpoint: str,
        model_type: str = "vit_h",
        image_size: int = 256,
    ) -> None:
        super().__init__()
        try:
            from segment_anything import sam_model_registry, SamPredictor
            from segment_anything.utils.transforms import ResizeLongestSide
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "SamAnchor requires `segment_anything`. Install with "
                "`pip install git+https://github.com/facebookresearch/segment-anything.git`."
            ) from e

        self.image_size = image_size
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.eval()
        for p in sam.parameters():
            p.requires_grad_(False)
        self.sam = sam
        self.predictor = SamPredictor(sam)
        self.transform = ResizeLongestSide(sam.image_encoder.img_size)

    @torch.no_grad()
    def encode_image(self, pil: Image.Image) -> torch.Tensor:
        """Returns SAM image embedding ``[1, 256, 64, 64]``."""
        img = pil.convert("RGB").resize((self.image_size, self.image_size))
        self.predictor.set_image(np.array(img))
        return self.predictor.get_image_embedding()

    def decode_with_tokens(
        self,
        image_embedding: torch.Tensor,
        pil: Image.Image,
        token_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Decode masks from learnable token embeddings.

        Args:
            image_embedding: ``[1, 256, 64, 64]`` from :meth:`encode_image`.
            pil:        PIL image (used for postprocess sizing).
            token_embeds: ``[N, 256]`` token embeddings.

        Returns:
            ``[N, H, W]`` mask logits at the original image resolution
            (after resize to ``image_size``).
        """
        img = pil.convert("RGB").resize((self.image_size, self.image_size))
        np_img = np.array(img)
        h, w = np_img.shape[:2]
        input_size = self.transform.apply_image(np_img).shape[:2]

        preds = []
        for tok in token_embeds:
            text_embeds = tok.view(1, 1, -1)  # [1, 1, 256]
            sparse, dense = self.sam.prompt_encoder(
                points=None, boxes=None, masks=None, text_embeds=text_embeds
            )
            low_res, _ = self.sam.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            up = self.sam.postprocess_masks(low_res, input_size=input_size, original_size=(h, w))[0]
            preds.append(up.squeeze(0))  # [H, W]
        return torch.stack(preds, dim=0)
