from typing import Any

import torch
from coral_pytorch.losses import coral_loss

from trainer.loss_strategy.ordinal_loss.ordinal_loss import OrdinalLossStrategy


class CORALLossStrategy(OrdinalLossStrategy):
    def compute(
        self, logits: torch.Tensor, labels: torch.Tensor, model_config: Any, **kwargs
    ) -> torch.Tensor:
        labels_ordinal = self._convert_labels(labels)
        return coral_loss(logits, labels_ordinal, num_classes=self.num_classes)
