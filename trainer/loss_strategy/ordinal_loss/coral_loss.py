from typing import Any

import torch
from coral_pytorch.losses import coral_loss
from coral_pytorch.dataset import levels_from_labelbatch

from trainer.loss_strategy.ordinal_loss.ordinal_loss import OrdinalLossStrategy


class CORALLossStrategy(OrdinalLossStrategy):
    def compute(
        self, logits: torch.Tensor, labels: torch.Tensor, model_config: Any, **kwargs
    ) -> torch.Tensor:
        levels = levels_from_labelbatch(labels, num_classes=self.num_classes)
        importance_weights = self.get_importance_weights()
        return coral_loss(logits, levels.to(logits.device), importance_weights=importance_weights.to(logits.device))
