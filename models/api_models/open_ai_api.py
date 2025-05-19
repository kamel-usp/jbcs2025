import logging
from typing import Literal

import openai
from omegaconf import DictConfig
from pydantic import BaseModel

from utils.secrets.secret_manager import get_api_key


class Competencia(BaseModel):
    """
    Data model representing a competency with a justification and a fixed set of scores.
    """

    justificativa: str
    pontuacao: Literal[0, 40, 80, 120, 160, 200]


def create_openai_client(cfg: DictConfig, logger: logging.Logger) -> openai.AsyncOpenAI:
    """
    Factory function to create an asynchronous OpenAI client using the provided configuration.

    Args:
        cfg (DictConfig): Configuration object with API details.
        logger (logging.Logger): Logger instance for logging client initialization.

    Returns:
        openai.AsyncOpenAI: An instance of the asynchronous OpenAI client.
    """
    logger.info(f"Setting up model {cfg.experiments.model.name} through OpenAI Client.")
    model_type = cfg.experiments.model.type
    api_key = cfg.experiments.model.api_key or get_api_key(model_type)
    client = openai.AsyncOpenAI(
        api_key=api_key,
        base_url=cfg.experiments.model.api_url,
    )
    logger.info("OpenAI client initialized successfully.")
    return client