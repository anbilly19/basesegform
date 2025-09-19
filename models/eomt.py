# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the timm library by Ross Wightman,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
from models.attention import Attention

import math

from models.scale_block import ScaleBlock


class EoMT(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes,
        num_q,
        num_blocks=4,
        masked_attn_enabled=True,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_q = num_q
        self.num_blocks = num_blocks
        self.masked_attn_enabled = masked_attn_enabled

        self.register_buffer("attn_mask_probs", torch.ones(num_blocks))
        self.attn = Attention(
            dim=self.encoder.embed_dim,
            num_heads=self.encoder.num_heads
        )
        self.q = nn.Embedding(num_q, self.encoder.embed_dim)

        self.class_head = nn.Linear(self.encoder.embed_dim, num_classes + 1)

        self.mask_head = nn.Sequential(
            nn.Linear(self.encoder.embed_dim, self.encoder.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.embed_dim, self.encoder.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.embed_dim, self.encoder.embed_dim),
        )

        patch_size = encoder.patch_size
        max_patch_size = max(patch_size[0], patch_size[1])
        num_upscale = max(1, int(math.log2(max_patch_size)) - 2)

        self.upscale = nn.Sequential(
            *[ScaleBlock(self.encoder.embed_dim) for _ in range(num_upscale)],
        )

    def _predict(self, x: torch.Tensor):
        print(f"X predict entry:{x.shape}")
        q = x[:, : self.num_q, :]

        class_logits = self.class_head(q)

        x = x[:, self.num_q + self.encoder.num_prefix_tokens :, :]
        x = x.transpose(1, 2).reshape(
            x.shape[0], -1, *self.encoder.grid_size
        )

        mask_logits = torch.einsum(
            "bqc, bchw -> bqhw", self.mask_head(q), self.upscale(x)
        )

        return mask_logits, class_logits


    def forward(self, x: torch.Tensor):
        x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std

        x = self.encoder.pre_block(x)
        orig_x_dims = len(x.shape)
        mask_logits_per_layer, class_logits_per_layer = [], []
        q = None

        for i, block in enumerate(self.encoder.blocks):
            if i == len(self.encoder.blocks) - self.num_blocks:
                print("X before concat",x.shape)
                if len(x.shape) > 3:
                    x = x.flatten(1,2)
                x = torch.cat(
                    (self.q.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1
                )
                print("X after concat",x.shape)
                

            if (
                # self.masked_attn_enabled and
                i >= len(self.encoder.blocks) - self.num_blocks
            ):
                mask_logits, class_logits = self._predict(self.encoder.norm(x))
                mask_logits_per_layer.append(mask_logits)
                class_logits_per_layer.append(class_logits)
                
                print("X pre attn",x.shape)
                
                # Cross attention between queries and embeddings
                x = self.attn(self.encoder.norm(x)) # does every variant have a similar norm layer?
                
                print("X post attn",x.shape)
                
                # Queries dislodged from embeddings
                q = x[:, : self.num_q, :]
                x = x[:, self.num_q :, :]
              
            
            # encoder blocks always process embeddings without queries
            if len(x.shape) != orig_x_dims:
                x = x.reshape(x.shape[0], self.encoder.grid_size[0], self.encoder.grid_size[1], -1).transpose(1,2)
            x = block(x)
            
            # queries re-added for next iter
            if q  is not None:
                if len(x.shape) > 3:
                    x = x.flatten(1,2)
                print("X before re-concat", x.shape)
                print("Q before re-concat", q.shape)
                x = torch.cat(
                        (q, x), dim=1
                    )
                print("X after re-concat", x.shape)
            
        mask_logits, class_logits = self._predict(self.encoder.norm(x))
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        return (
            mask_logits_per_layer,
            class_logits_per_layer,
        )
