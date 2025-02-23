from logging import Logger

from omegaconf import DictConfig
from models.api_models.open_ai_api import create_openai_client
from models.fine_tuning_models.classification_head import (
    load_slm_decoder_classification_lora,
    load_model_with_classification_head,
    load_phi3_classification_lora,
)
from models.fine_tuning_models.model_types_enum import ModelTypesEnum


class ModelFactory:
    @staticmethod
    def create_model(experiment_config: DictConfig, logger: Logger):
        model_cfg = experiment_config.experiments.model
        model = None
        if model_cfg.type == ModelTypesEnum.ENCODER_CLASSIFICATION.value:
            model = load_model_with_classification_head(experiment_config)
        if model_cfg.type == ModelTypesEnum.PHI35_CLASSIFICATION_LORA.value:
            model = load_phi3_classification_lora(experiment_config, logger)
        if model_cfg.type in [
            ModelTypesEnum.PHI4_CLASSIFICATION_LORA.value,
            ModelTypesEnum.LLAMA31_CLASSIFICATION_LORA.value,
        ]:
            model = load_slm_decoder_classification_lora(experiment_config, logger)
        if model_cfg.type in [
            ModelTypesEnum.CHATGPT_4O.value,
            ModelTypesEnum.MARITACA_SABIA.value,
            ModelTypesEnum.DEEPSEEK_R1.value,
        ]:
            model = create_openai_client(experiment_config, logger)
        if model is None:
            raise ValueError("You need to provide a valid Model Classification Type")

        return model
