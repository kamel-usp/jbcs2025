from typing import Any, Optional

import torch
import torch.nn as nn

from trainer.loss_strategy.loss_strategy import LossStrategy


class CrossEntropyLossStrategy(LossStrategy):
    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        self.class_weights = class_weights

    def compute(
        self, logits: torch.Tensor, labels: torch.Tensor, model_config: Any, **kwargs
    ) -> torch.Tensor:
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        return loss_fct(logits.view(-1, model_config.num_labels), labels.view(-1))
