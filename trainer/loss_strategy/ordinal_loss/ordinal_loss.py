from typing import Optional

import torch
from torch import Tensor

from trainer.loss_strategy.loss_strategy import LossStrategy


class OrdinalLossStrategy(LossStrategy):
    """Base class for ordinal regression losses."""

    def __init__(
        self, num_classes: int = 6, class_weights: Optional[torch.Tensor] = None
    ):
        self.num_classes = num_classes
        self.class_weights = class_weights

    def _convert_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Convert labels to ordinal format if needed."""
        if labels.max() > self.num_classes:
            return (
                labels // 40
            )  # Convert [0, 40, 80, 120, 160, 200] to [0, 1, 2, 3, 4, 5]
        return labels

    @torch.no_grad()
    def get_importance_weights(
        self,
    ) -> Tensor:
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
