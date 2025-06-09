from logging import Logger

from omegaconf import DictConfig
from models.api_models.open_ai_api import create_openai_client
from models.fine_tuning_models.model_config.model_config import ModelConfig
from models.fine_tuning_models.model_loader import EncoderModelLoader, DecoderModelLoraLoader
from models.fine_tuning_models.model_types_enum import ModelTypesEnum


class ModelFactory:
    # API model types
    API_MODELS = {
        ModelTypesEnum.CHATGPT_4O.value,
        ModelTypesEnum.MARITACA_SABIA.value,
        ModelTypesEnum.DEEPSEEK_R1.value,
    }
    
    @staticmethod
    def create_model(experiment_config: DictConfig, logger: Logger):
        model_type = experiment_config.experiments.model.type
        
        # Handle API models
        if model_type in ModelFactory.API_MODELS:
            return create_openai_client(experiment_config, logger)
        
        # Parse model configuration
        model_config = ModelConfig.from_model_type(model_type)
        
        # Select appropriate loader
        if model_config.is_lora:
            loader = DecoderModelLoraLoader(experiment_config, logger, model_config)
        else:
            loader = EncoderModelLoader(experiment_config, logger, model_config)
        
        return loader.load()
    
    @staticmethod
    def get_loss_type(model_type: str) -> str:
        """Extract loss type from model type."""
        model_config = ModelConfig.from_model_type(model_type)
        return model_config.loss_type.value