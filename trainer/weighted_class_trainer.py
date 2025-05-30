from typing import Optional

import torch
from transformers import Trainer

from models.fine_tuning_models.model_config.model_config import ModelConfig
from models.fine_tuning_models.model_types_enum import LossType
from trainer.loss_strategy.cross_entropy_strategy import CrossEntropyLossStrategy
from trainer.loss_strategy.loss_strategy import LossStrategy
from trainer.loss_strategy.ordinal_loss.coral_loss import CORALLossStrategy
from trainer.loss_strategy.ordinal_loss.corn_loss import CORNLossStrategy


class WeightedLossTrainer(Trainer):
    def __init__(
        self,
        *args,
        class_weights: Optional[torch.Tensor] = None,
        model_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Parse model configuration
        self.model_config = (
            ModelConfig.from_model_type(model_type) if model_type else None
        )

        # Initialize loss strategy
        self.loss_strategy = self._create_loss_strategy(class_weights)

    def _create_loss_strategy(
        self, class_weights: Optional[torch.Tensor]
    ) -> LossStrategy:
        """Factory method to create the appropriate loss strategy."""
        if not self.model_config:
            # Default to cross-entropy
            return CrossEntropyLossStrategy(
                class_weights.to(self.model.device)
                if class_weights is not None
                else None
            )

        loss_type = self.model_config.loss_type

        if loss_type == LossType.CORAL:
            return CORALLossStrategy()
        elif loss_type == LossType.CORN:
            return CORNLossStrategy()
        else:
            return CrossEntropyLossStrategy(
                class_weights.to(self.model.device)
                if class_weights is not None
                else None
            )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.get("labels")

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Compute loss using strategy
        loss = self.loss_strategy.compute(
            logits=logits, labels=labels, model_config=self.model.config
        )

        return (loss, outputs) if return_outputs else loss
