from typing import Any

import torch
import torch.nn.functional as F
from trainer.loss_strategy.ordinal_loss.ordinal_loss import OrdinalLossStrategy


class CORNLossStrategy(OrdinalLossStrategy):

    def _corn_loss(self, logits, y_train, num_classes, importance_weights=None):
        # source: https://github.com/Raschka-research-group/coral-pytorch/issues/39
        sets = []
    
        for i in range(num_classes-1):
            label_mask = y_train > i-1
            label_tensor = (y_train[label_mask] > i).to(torch.int64)
            sets.append((label_mask, label_tensor))

        num_examples = 0
        losses = 0.
    
        if importance_weights is None:
            importance_weights = torch.ones(len(sets))
    
        for task_index, s in enumerate(sets):
            train_examples = s[0]
            train_labels = s[1]

            if len(train_labels) < 1:
                continue

            num_examples += len(train_labels)
            pred = logits[train_examples, task_index]

            loss = -torch.sum(F.logsigmoid(pred)*train_labels
                              + (F.logsigmoid(pred) - pred)*(1-train_labels))
            
            #losses += loss
            losses += importance_weights[task_index] * loss

        return losses/num_examples
    
    def compute(
        self, logits: torch.Tensor, labels: torch.Tensor, model_config: Any, **kwargs
    ) -> torch.Tensor:
        importance_weights = self.get_importance_weights()
        return self._corn_loss(logits, labels, num_classes=self.num_classes, importance_weights=importance_weights)
