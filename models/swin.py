

from typing import Optional
import timm
import torch
import torch.nn as nn


class Swin(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        patch_size=4,
        backbone_name="swin_tiny_patch4_window7_224.ms_in1k", # patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24)
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=ckpt_path is None,
            img_size=img_size,
            patch_size=patch_size,
            num_classes=0,
        )

        self.embed_dim = self.backbone.embed_dim
        self.num_heads = 12
        self.patch_size = self.backbone.patch_embed.patch_size
        self.grid_size = self.backbone.patch_embed.grid_size
        self.num_prefix_tokens = 0
        self.blocks = self.backbone.layers
        self.norm = nn.LayerNorm(self.embed_dim)
        pixel_mean = torch.tensor(self.backbone.default_cfg["mean"]).reshape(
            1, -1, 1, 1
        )
        pixel_std = torch.tensor(self.backbone.default_cfg["std"]).reshape(1, -1, 1, 1)

        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)
    
    def pre_block(self, x: torch.Tensor):
        x = self.backbone.patch_embed(x)
        return x