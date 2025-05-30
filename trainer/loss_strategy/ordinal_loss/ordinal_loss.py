import torch

from trainer.loss_strategy.loss_strategy import LossStrategy


class OrdinalLossStrategy(LossStrategy):
    """Base class for ordinal regression losses."""

    def __init__(self, num_classes: int = 6):
        self.num_classes = num_classes

    def _convert_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Convert labels to ordinal format if needed."""
        if labels.max() > self.num_classes - 1:
            return (
                labels // 40
            )  # Convert [0, 40, 80, 120, 160, 200] to [0, 1, 2, 3, 4, 5]
        return labels
