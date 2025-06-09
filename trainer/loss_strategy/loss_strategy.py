from abc import ABC, abstractmethod
import torch
from typing import Any 

class LossStrategy(ABC):
    """Abstract base class for loss computation strategies."""
    
    @abstractmethod
    def compute(self, logits: torch.Tensor, labels: torch.Tensor, 
                model_config: Any, **kwargs) -> torch.Tensor:
        """Compute the loss."""
        pass