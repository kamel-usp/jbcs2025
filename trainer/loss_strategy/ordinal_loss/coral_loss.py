from typing import Any

import torch
from coral_pytorch.losses import coral_loss
from coral_pytorch.dataset import levels_from_labelbatch

from trainer.loss_strategy.ordinal_loss.ordinal_loss import OrdinalLossStrategy


class CORALLossStrategy(OrdinalLossStrategy):
    def _threshold_weights_from_class_weights(self):
        """
        Average the class weights on the '≤ j' side and the '> j' side of every
        CORAL threshold and take their mean.  Returns (K-1,) torch.float32.
        """
        cw = torch.as_tensor(self.class_weights, dtype=torch.float32)
        K = cw.numel()

        # arithmetic-mean of weights to the *left* of each threshold
        left = torch.cumsum(cw, 0)[:-1] / torch.arange(
            1, K, dtype=torch.float32, device=self.class_weights.device
        )
        # arithmetic-mean of weights to the *right* of each threshold
        right = torch.cumsum(cw.flip(0), 0).flip(0)[1:] / torch.arange(
            K - 1, 0, -1, dtype=torch.float32, device=self.class_weights.device
        )

        return 0.5 * (left + right)

    def compute(
        self, logits: torch.Tensor, labels: torch.Tensor, model_config: Any, **kwargs
    ) -> torch.Tensor:
        levels = levels_from_labelbatch(labels, num_classes=self.num_classes)
        importance_weights = (
            self._threshold_weights_from_class_weights()
            if self.class_weights is not None
            else None
        )
        return coral_loss(logits, levels.to(logits.device), importance_weights=importance_weights)
