import logging
import os
import random
import sys
from datetime import datetime
from logging import Logger
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import hydra
import numpy as np
import torch
from fine_tuning import fine_tune_pipeline
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed

# Append the parent directory to sys.path using pathlib
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Register OmegaConf resolvers BEFORE hydra.main is called
from utils.secrets.secret_manager import register_resolvers  # NOQA

register_resolvers()

from metrics.metrics import save_evaluation_results_to_csv  # NOQA
from models.fine_tuning_models.model_types_enum import ModelTypesEnum  # NOQA
from models.api_models.api_inference import api_inference_pipeline  # NOQA

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.INFO)
transformers_logger.propagate = True
logger = logging.getLogger(__name__)


def get_experiment_id(experiment_config: DictConfig) -> str:
    model_name = experiment_config.experiments.model.name
    model_type = None
    if hasattr(experiment_config.experiments.model, "prompt_type"):
        model_type = experiment_config.experiments.model.prompt_type
    elif hasattr(experiment_config.experiments.model, "type"):
        model_type = experiment_config.experiments.model.type
        if "/" in model_name:
            model_name = model_name.split("/")[-1]
    experiment_id = (
        f"{model_name}-{model_type}-"
        f"C{experiment_config.experiments.dataset.grade_index + 1}"
    )
    return experiment_id


def fine_tune_process(cfg: DictConfig, logger: Logger):
    trainer, tokenized_dataset = fine_tune_pipeline(cfg, logger)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    evaluate_baseline = trainer.evaluate()
    evaluate_baseline["epoch"] = -1
    evaluate_baseline["reference"] = "validation_before_training"
    experiment_id = get_experiment_id(cfg)
    save_evaluation_results_to_csv(
        experiment_id,
        evaluate_baseline,
        current_time,
    )
    trainer.train()
    evaluate_after_training = trainer.evaluate()
    evaluate_after_training["reference"] = "validation_after_training"
    save_evaluation_results_to_csv(
        experiment_id,
        evaluate_after_training,
        current_time,
    )
    logger.info("Training completed successfully.")
    logger.info("Running on Test")
    evaluate_test = trainer.evaluate(tokenized_dataset["test"])
    evaluate_test["reference"] = "test_results"
    save_evaluation_results_to_csv(
        experiment_id,
        evaluate_test,
        current_time,
    )
    trainer.save_model(cfg.experiments.model.best_model_dir)


def api_calling_pipeline(cfg: DictConfig, logger: Logger):
    api_results = api_inference_pipeline(cfg, logger)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    experiment_id = get_experiment_id(cfg)
    save_evaluation_results_to_csv(
        experiment_id,
        api_results,
        current_time,
    )


@hydra.main(version_base="1.1", config_path="../configs/", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))

    set_seed(cfg.training_params.seed)
    torch.manual_seed(cfg.training_params.seed)
    np.random.seed(cfg.training_params.seed)
    random.seed(cfg.training_params.seed)

    # If using CUDA
    torch.cuda.manual_seed_all(cfg.training_params.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if cfg.experiments.model.type in [
        ModelTypesEnum.ENCODER_CLASSIFICATION.value,
        ModelTypesEnum.LLAMA31_CLASSIFICATION_LORA.value,
        ModelTypesEnum.PHI35_CLASSIFICATION_LORA.value,
        ModelTypesEnum.PHI4_CLASSIFICATION_LORA.value,
    ]:
        logger.info("Starting the Fine Tuning training process.")
        fine_tune_process(cfg, logger)
        logger.info("Fine Tuning Finished.")
    elif cfg.experiments.model.type in [
        ModelTypesEnum.CHATGPT_4O.value,
        ModelTypesEnum.MARITACA_SABIA.value,
        ModelTypesEnum.DEEPSEEK_R1.value,
    ]:
        logger.info("Starting Zero-Shot or Few-Shot Learning Process")
        api_calling_pipeline(cfg, logger)
        logger.info("API Calls Pipeline Finished.")


if __name__ == "__main__":
    main()
