from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, Dict

from omegaconf import DictConfig
from transformers import (
    PreTrainedModel,
)

from models.fine_tuning_models.model_config.model_config import ModelConfig
from models.fine_tuning_models.model_types_enum import ModelArchitecture


class BaseModelLoader(ABC):
    """Abstract base class for model loaders."""

    def __init__(self, cfg: DictConfig, logger: Logger, model_config: ModelConfig):
        self.cfg = cfg
        self.logger = logger
        self.model_config = model_config
        self.model_cfg = cfg.experiments.model

    @abstractmethod
    def load(self) -> PreTrainedModel:
        """Load and return the model."""
        pass

    def _get_base_model_kwargs(self) -> Dict[str, Any]:
        """Common kwargs for model initialization."""
        kwargs = {
            "num_labels": self.model_config.num_labels,
            "cache_dir": self.cfg.cache_dir,
            "trust_remote_code": True,
            "device_map": "cuda",
            "torch_dtype": "auto",
        }
        # Only add label mappings for classification models using cross entropy
        if self.model_config.architecture == ModelArchitecture.ENCODER and not self.model_config.is_ordinal:
            id2label = {0: 0, 1: 40, 2: 80, 3: 120, 4: 160, 5: 200}
            label2id = {0: 0, 40: 1, 80: 2, 120: 3, 160: 4, 200: 5}
            kwargs.update({
                "id2label": id2label,
                "label2id": label2id,
            })
        if self.model_config.is_ordinal:
            kwargs.update({"num_labels": self.model_config.num_labels -1})
        
        return kwargs

    def _get_decoder_model_kwargs(self) -> Dict[str, Any]:
        """Additional kwargs for decoder models."""
        kwargs = self._get_base_model_kwargs()
        kwargs.update(
            {
                "device_map": "cuda",
                "torch_dtype": "auto",
                "attn_implementation": "flash_attention_2",
            }
        )
        return kwargs
