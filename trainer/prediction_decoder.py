import torch
import numpy as np
from coral_pytorch.dataset import corn_label_from_logits, proba_to_label
from typing import Union, List

from models.fine_tuning_models.model_config.model_config import ModelConfig
from models.fine_tuning_models.model_types_enum import LossType



class PredictionDecoder:
    """Handles prediction decoding for different model types."""
    
    # Label mapping for your specific use case
    LABEL_VALUES = np.array([0, 40, 80, 120, 160, 200])
    
    @classmethod
    def decode(cls, logits: Union[torch.Tensor, np.ndarray], 
               model_type: str, num_classes: int = 6) -> np.ndarray:
        """
        Decode model predictions based on model type.
        
        Args:
            logits: Model outputs
            model_type: String identifier of the model type
            num_classes: Number of classes
            
        Returns:
            Decoded predictions in original label space
        """
        # Convert numpy to torch if needed
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        
        # Parse model configuration
        model_config = ModelConfig.from_model_type(model_type, num_classes)
        
        with torch.no_grad():
            if model_config.loss_type == LossType.CORN:
                # Use coral-pytorch's CORN prediction function
                predictions = corn_label_from_logits(logits)
            elif model_config.loss_type == LossType.CORAL:
                # Use coral-pytorch's CORAL prediction function
                probas = torch.sigmoid(logits)
                predictions = proba_to_label(probas)
            else:  # Cross-entropy
                # Standard argmax for classification
                predictions = torch.argmax(logits, dim=1)
            
            # Convert to numpy
            predictions_np = predictions.cpu().numpy()
            
            # Map to original label space
            # For all models, we assume predictions are in [0, 5] range
            # and need to be mapped to [0, 40, 80, 120, 160, 200]
            return cls.LABEL_VALUES[predictions_np.clip(0, num_classes - 1)]
    
    @classmethod
    def prepare_labels(cls, labels: Union[np.ndarray, List[int]], 
                      model_type: str) -> np.ndarray:
        """
        Prepare labels for model training/evaluation.
        
        Args:
            labels: Original labels
            model_type: String identifier of the model type
            
        Returns:
            Prepared labels for the specific model type
        """
        labels_array = np.array(labels) if isinstance(labels, list) else labels
        model_config = ModelConfig.from_model_type(model_type)
        
        if model_config.is_ordinal and labels_array.max() > 5:
            # Convert to ordinal format [0, 1, 2, 3, 4, 5]
            return labels_array // 40
        elif not model_config.is_ordinal and labels_array.max() > 5:
            # For standard classification, also convert to class indices
            return labels_array // 40
        
        return labels_array