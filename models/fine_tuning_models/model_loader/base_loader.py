from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, Dict

from omegaconf import DictConfig
from transformers import (
    PreTrainedModel,
)

from models.fine_tuning_models.model_config.model_config import ModelConfig


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
        id2label = {0: 0, 1: 40, 2: 80, 3: 120, 4: 160, 5: 200}
        label2id = {0: 0, 40: 1, 80: 2, 120: 3, 160: 4, 200: 5}
        return {
            "num_labels": self.model_config.num_labels,
            "id2label": id2label,
            "label2id": label2id,
            "cache_dir": self.cfg.cache_dir,
            "trust_remote_code": True,
        }

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
