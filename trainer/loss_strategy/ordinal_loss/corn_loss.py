from typing import Any

import torch
from coral_pytorch.losses import corn_loss

from trainer.loss_strategy.ordinal_loss.ordinal_loss import OrdinalLossStrategy


class CORNLossStrategy(OrdinalLossStrategy):
    def compute(
        self, logits: torch.Tensor, labels: torch.Tensor, model_config: Any, **kwargs
    ) -> torch.Tensor:
        labels_ordinal = self._convert_labels(labels)
        return corn_loss(logits, labels_ordinal, num_classes=self.num_classes)
