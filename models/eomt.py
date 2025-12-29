# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the timm library by Ross Wightman,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
from models.cross_attention import CrossAttentionSingleInput
from models.attention import Attention
from models.layer_scale import LayerScale
from models.drop_path import DropPath
import math

from models.scale_block import ScaleBlock


class EoMT(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes,
        num_q,
        num_blocks=4,
        self_a=False,
        masked_attn_enabled=True,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_q = num_q
        self.num_blocks = num_blocks
        self.masked_attn_enabled = masked_attn_enabled
        self.register_buffer("attn_mask_probs", torch.ones(num_blocks))
        self.attn = self.get_attention_list(self_a)
        self.q = nn.Parameter(torch.randn(num_q, self.encoder.embed_dim))
        self.ls_list = self.get_layer_scale_list()
        self.dp = DropPath(0.0)

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

    def get_layer_scale_list(self):
        ls_list = nn.ModuleList()
        embed_dim = self.encoder.embed_dim
        len_blocks = sum(self.encoder.depths)
        i = 0
        for d in self.encoder.depths:
            for _ in range(d):
                if i >= len_blocks - self.num_blocks:
                    ls_list.append(LayerScale(embed_dim))
                i+=1
            embed_dim = embed_dim * self.encoder.attn_multiplier
        return ls_list
    
    def get_attention_list(self, self_a):
        attn_list = nn.ModuleList()
        embed_dim = self.encoder.embed_dim
        len_blocks = sum(self.encoder.depths)
        i = 0
        for d,h in zip(self.encoder.depths,self.encoder.num_heads):
            for _ in range(d):
                if i >= len_blocks - self.num_blocks:
                    if self_a:
                            
                        attn_list.append(
                            Attention(
                                dim=embed_dim,
                                num_heads=h
                            )
                        )
                    else:
                       attn_list.append( 
                           CrossAttentionSingleInput(
                                dim = embed_dim,
                                num_heads = h)
                        )
                i+=1
            embed_dim = embed_dim * self.encoder.attn_multiplier
        return attn_list
    
    def _predict(self, x: torch.Tensor):
        
        q = x[:, : self.num_q, :]

        class_logits = self.class_head(q)

        x = x[:, self.num_q + self.encoder.num_prefix_tokens :, :]
        x = x.transpose(1, 2).reshape(
            x.shape[0], -1, *self.encoder.grid_size
        )
        mask_features = self.mask_head(q)  # (B, Q, C)
        upscaled = self.upscale(x)  # (B, C, H, W)

        B, Q, _ = mask_features.shape
        mask_logits = torch.matmul(
            mask_features, 
            upscaled.flatten(2)  # (B, C, H*W)
        ).view(B, Q, *upscaled.shape[-2:])  # (B, Q, H, W)       

        return mask_logits, class_logits


    def forward(self, x: torch.Tensor):
        x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std

        x = self.encoder.pre_block(x)
        orig_x_dims = len(x.shape)
        mask_logits_per_layer, class_logits_per_layer = [], []
        q = None

        for i, block in enumerate(self.encoder.blocks):
            if i == len(self.encoder.blocks) - self.num_blocks:
               
                if len(x.shape) > 3:
                    x = x.flatten(1,2)
                x = torch.cat(
                    (self.q[None, :, :].expand(x.shape[0], -1, -1), x), dim=1
                )                
            if (
                i >= len(self.encoder.blocks) - self.num_blocks
            ):
                mask_logits, class_logits = self._predict(self.encoder.norm(x))
                mask_logits_per_layer.append(mask_logits)
                class_logits_per_layer.append(class_logits)
             
                # Cross attention between queries and embeddings
                new_x = self.attn[i - len(self.encoder.blocks)](self.encoder.norm(x))
                x = x + self.dp(self.ls_list[i - len(self.encoder.blocks)](new_x))
                # Queries dislodged from embeddings
                q = x[:, : self.num_q, :]
                x = x[:, self.num_q :, :]
               
                q = self.encoder.q_mlp(block,q)
            # encoder blocks always process embeddings without queries
            if len(x.shape) != orig_x_dims:
                x = x.reshape(x.shape[0], self.encoder.grid_size[0], self.encoder.grid_size[1], -1).transpose(1,2)
            x = self.encoder.run_block(block,x,i)
            
            # queries re-added for next iter
            if q is not None:
                if len(x.shape) > 3:
                    x = x.flatten(1,2)
                x = torch.cat(
                        (q, x), dim=1
                    )
            
        mask_logits, class_logits = self._predict(self.encoder.norm(x))
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        return (
            mask_logits_per_layer,
            class_logits_per_layer,
        )
