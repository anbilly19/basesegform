import math
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler

class TwoStageWarmupCosineSchedule(LRScheduler):
    """
    A learning rate scheduler that inherits from PyTorch's _LRScheduler to
    implement a two-stage warmup followed by a cosine decay.

    This scheduler uses the parent class's step() method and only requires
    the get_lr() method to be defined.

    It handles two parameter groups differently:
    1. Backbone parameters (e.g., from a Vision Transformer), which have a
       delayed warmup.
    2. Other parameters, which start a linear warmup from the beginning.

    After the warmup phases, the learning rate for all parameters decays
    following a cosine annealing schedule.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        num_backbone_params: int,
        warmup_steps: tuple[int, int],
        total_steps: int,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer (Optimizer): The optimizer for which to schedule the learning rate.
            num_backbone_params (int): The number of parameter groups belonging to the
                                       backbone model. These will have a delayed warmup.
            warmup_steps (tuple[int, int]): A tuple containing the number of warmup
                                            steps for non-backbone and backbone parameters,
                                            respectively (non_vit_warmup, vit_warmup).
            total_steps (int): The total number of training steps.
            last_epoch (int): The index of the last epoch. Defaults to -1.
        """
        self.num_backbone_params = num_backbone_params
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        # Initialize the parent LRScheduler class
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        
        step = self.last_epoch
        lrs = []
        non_vit_warmup, vit_warmup = self.warmup_steps

        for i, base_lr in enumerate(self.base_lrs):
            # Check if the parameter group belongs to the backbone
            is_backbone = i >= self.num_backbone_params

            if is_backbone:
                # --- Non-Backbone Parameters ---
                if non_vit_warmup > 0 and step < non_vit_warmup:
                    # Linear warmup for non-backbone parameters
                    lr = base_lr * (step / non_vit_warmup)
                else:
                    # Cosine decay after warmup
                    adjusted_step = max(0, step - non_vit_warmup)
                    max_decay_steps = max(1, self.total_steps - non_vit_warmup)
                    cosine_decay = 0.5 * (1 + math.cos(math.pi * adjusted_step / max_decay_steps))
                    lr = base_lr * cosine_decay
            else:
                # --- Backbone Parameters ---
                if step < non_vit_warmup:
                    # LR is 0 during the initial non-backbone warmup phase
                    lr = 0
                elif step < non_vit_warmup + vit_warmup:
                    # Linear warmup for backbone parameters
                    adjusted_step = step - non_vit_warmup
                    lr = base_lr * (adjusted_step / vit_warmup)
                else:
                    # Cosine decay after warmup
                    adjusted_step = max(0, step - non_vit_warmup - vit_warmup)
                    max_decay_steps = max(1, self.total_steps - non_vit_warmup - vit_warmup)
                    cosine_decay = 0.5 * (1 + math.cos(math.pi * adjusted_step / max_decay_steps))
                    lr = base_lr * cosine_decay
            
            lrs.append(lr)
            
        return lrs

