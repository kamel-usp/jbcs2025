import logging
import os
import random
import shutil
import sys
from datetime import datetime
from logging import Logger
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import hydra
import numpy as np
import torch
from codecarbon import EmissionsTracker
from fine_tuning import fine_tune_pipeline
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed

from models.fine_tuning_models.model_config.model_config import ModelConfig
from models.fine_tuning_models.model_factory import ModelFactory
from models.fine_tuning_models.model_types_enum import ModelArchitecture

# Append the parent directory to sys.path using pathlib
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Register OmegaConf resolvers BEFORE hydra.main is called
from utils.secrets.secret_manager import register_resolvers  # NOQA
from utils.logger.additional_logging import log_gpu_info  # NOQA

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
    context_type = "essay_only"
    use_full_context = experiment_config.experiments.dataset.use_full_context
    model_type = None
    if hasattr(experiment_config.experiments.model, "prompt_type"):
        model_type = experiment_config.experiments.model.prompt_type
    elif hasattr(experiment_config.experiments.model, "type"):
        model_type = experiment_config.experiments.model.type
        if "/" in model_name:
            model_name = model_name.split("/")[-1]
    if use_full_context:
        context_type = "full_context"
    experiment_id = (
        f"{model_name}-{model_type}-"
        f"C{experiment_config.experiments.dataset.grade_index + 1}-"
        f"{context_type}"
    )
    if hasattr(experiment_config.experiments.model, "lora_r"):
        experiment_id += f"-r{experiment_config.experiments.model.lora_r}"
    return experiment_id


def fine_tune_process(cfg: DictConfig, logger: Logger):
    trainer, tokenized_dataset, tokenizer = fine_tune_pipeline(cfg, logger)
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
    logger.info(f"Test metrics: {evaluate_test}")
    evaluate_test["reference"] = "test_results"
    save_evaluation_results_to_csv(
        experiment_id,
        evaluate_test,
        current_time,
    )

    # Save model and tokenizer
    best_model_dir = cfg.experiments.model.best_model_dir
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    logger.info(f"Model and tokenizer saved to {best_model_dir}")

    return experiment_id


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
    original_output_dir = os.getcwd()
    tracker = EmissionsTracker(
        project_name="jbcs2025",
        experiment_id=get_experiment_id(cfg),
        output_dir=original_output_dir,
        output_file="emissions.csv",
        log_level="error",
        measure_power_secs=30,
    )
    tracker.start()

    set_seed(cfg.training_params.seed)
    torch.manual_seed(cfg.training_params.seed)
    np.random.seed(cfg.training_params.seed)
    random.seed(cfg.training_params.seed)

    # If using CUDA
    torch.cuda.manual_seed_all(cfg.training_params.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    log_gpu_info(logger)
    model_config = ModelConfig.from_model_type(cfg.experiments.model.type)

    experiment_id = None
    if model_config.architecture in [
        ModelArchitecture.ENCODER,
        ModelArchitecture.PHI35,
        ModelArchitecture.PHI4,
        ModelArchitecture.LLAMA31,
    ]:
        logger.info("Starting the Fine Tuning training process.")
        experiment_id = fine_tune_process(cfg, logger)
        logger.info("Fine Tuning Finished.")
    elif cfg.experiments.model.type in ModelFactory.API_MODELS:
        logger.info("Starting Zero-Shot or Few-Shot Learning Process")
        api_calling_pipeline(cfg, logger)
        logger.info("API Calls Pipeline Finished.")
        experiment_id = get_experiment_id(cfg)

    # Stop emissions tracking
    emissions = tracker.stop()
    logger.info(f"Total emissions: {emissions:.4f} kg CO2eq")

    # Rename output directory with experiment ID (similar to run_inference_experiment.py)
    if experiment_id:
        # Close all file handlers to release log files
        handlers = list(logger.handlers)
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)

        # Also close root logger's handlers
        root_logger = logging.getLogger()
        for handler in list(root_logger.handlers):
            handler.close()
            root_logger.removeHandler(handler)

        # Prepare new directory name
        parent_dir = os.path.dirname(original_output_dir)
        new_dir_name = os.path.join(parent_dir, f"{experiment_id}")

        # Check if target directory already exists
        if os.path.exists(new_dir_name):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_dir_name = f"{new_dir_name}_{timestamp}"

        try:
            # Create new directory and copy files
            os.makedirs(new_dir_name, exist_ok=True)

            # Copy files one by one
            for item in os.listdir(original_output_dir):
                src_path = os.path.join(original_output_dir, item)
                dst_path = os.path.join(new_dir_name, item)

                try:
                    if os.path.isfile(src_path):
                        shutil.copy2(src_path, dst_path)
                    elif os.path.isdir(src_path):
                        shutil.copytree(src_path, dst_path)
                except Exception as e:
                    logger.error(f"Failed to copy {item}: {e}")

            logger.info(f"Training outputs copied to: {new_dir_name}")
            logger.info(f"Original directory remains at: {original_output_dir}")

        except Exception as e:
            logger.error(f"Failed to create output directory: {e}")


if __name__ == "__main__":
    main()
