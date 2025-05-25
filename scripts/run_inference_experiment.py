import json
import logging
import os
import shutil
import sys
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import torch
from datasets import Dataset, load_dataset
from omegaconf import DictConfig, OmegaConf
from sklearn.utils.class_weight import compute_class_weight
from transformers import TrainingArguments, set_seed

# Append the parent directory to sys.path using pathlib
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Register OmegaConf resolvers BEFORE hydra.main is called
from utils.secrets.secret_manager import register_resolvers  # NOQA

register_resolvers()

from metrics.metrics import compute_metrics, save_evaluation_results_to_csv  # NOQA
from models.fine_tuning_models.model_factory import ModelFactory  # NOQA
from models.fine_tuning_models.model_types_enum import ModelTypesEnum  # NOQA
from models.api_models.api_inference import api_inference_pipeline  # NOQA
from preprocess import load_tokenizer, tokenize_dataset  # NOQA
from run_experiment import get_experiment_id  # NOQA
from trainer.weighted_class_trainer import WeightedLossTrainer  # NOQA

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def save_model_predictions_jsonl(
    dataset_test: Dataset,
    predictions: List[int],
    labels: List[int],
    grade_index: int,
    experiment_id: str,
    jsonl_filename: str = "inference_results.jsonl",
) -> None:
    """
    Save inference results to a JSONL file for fine-tuned models.

    Parameters:
        dataset_test: The test dataset
        predictions: Model predictions
        labels: Ground truth labels
        grade_index: Index of the grade being evaluated
        experiment_id: ID of the experiment
        jsonl_filename: Name of the output file
    """
    # Retrieve fields from the dataset
    ids = dataset_test["id"]
    id_prompts = dataset_test["id_prompt"]
    test_essays = dataset_test["essay_text"]
    reference = (
        dataset_test["reference"] if "reference" in dataset_test else ["" for _ in ids]
    )

    rows = []
    for idx, essay in enumerate(test_essays):
        row = {
            "id": ids[idx],
            "id_prompt": id_prompts[idx],
            "essay_text": essay,
            "label": int(labels[idx]),
            "prediction": int(predictions[idx]),
            "grade_index": grade_index,
            "reference": reference[idx],
        }
        rows.append(row)

    # Write each row as a JSON object on a new line
    with open(f"{experiment_id}_{jsonl_filename}", "w", encoding="utf-8") as jsonlfile:
        for row in rows:
            json_line = json.dumps(row, ensure_ascii=False)
            jsonlfile.write(json_line + "\n")

    logger.info(f"Inference results saved to {experiment_id}_{jsonl_filename}")


def load_model_from_hub(cfg: DictConfig, logger: Logger):
    """
    Load a pretrained model from Hugging Face Hub.
    """
    logger.info(f"Loading model from: {cfg.experiments.model.name}")

    model = ModelFactory.create_model(cfg, logger)

    return model


def hf_model_inference(cfg: DictConfig, logger: Logger) -> Dict[str, float]:
    """
    Run inference with a Hugging Face model loaded from the hub.
    """
    # Load dataset
    dataset = load_dataset(
        cfg.dataset.name,
        cfg.dataset.split,
        cache_dir=cfg.cache_dir,
    )

    # Load tokenizer
    tokenizer = load_tokenizer(
        cfg.experiments.model.type,
        cfg.experiments.tokenizer.name,
        cache_dir=cfg.cache_dir,
    )

    # Tokenize the dataset
    grade_index = cfg.experiments.dataset.grade_index
    tokenized_dataset = tokenize_dataset(
        dataset,
        tokenizer,
        text_column="essay_text",
        grade_index=grade_index,
        model_type=cfg.experiments.model.type,
        logger=logger,
    )

    # Load the model from Hub
    model = load_model_from_hub(cfg, logger)

    train_labels = tokenized_dataset["train"]["label"]
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=np.array(train_labels),
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    training_args = TrainingArguments(
        seed=cfg.training_params.seed,
        data_seed=cfg.training_params.seed,
        output_dir=os.getcwd(),
        per_device_eval_batch_size=cfg.experiments.training_params.eval_batch_size,
        bf16=cfg.training_params.bf16,
    )

    # Create trainer
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        class_weights=class_weights_tensor,
    )

    # Run inference
    logger.info("Running inference on test dataset")
    test_results = trainer.predict(tokenized_dataset["test"])
    logits = test_results.predictions
    labels = test_results.label_ids

    # Calculate metrics
    metrics = compute_metrics((logits, labels), cfg)

    # Save inference results
    save_model_predictions_jsonl(
        dataset_test=dataset["test"],
        predictions=np.argmax(logits, axis=1),
        labels=labels,
        grade_index=grade_index,
        experiment_id=get_experiment_id(cfg),
    )

    return metrics


def run_inference(cfg: DictConfig, logger: Logger):
    """
    Main inference function that determines whether to use HF model or API.
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if cfg.experiments.model.type in [
        ModelTypesEnum.ENCODER_CLASSIFICATION.value,
        ModelTypesEnum.LLAMA31_CLASSIFICATION_LORA.value,
        ModelTypesEnum.PHI35_CLASSIFICATION_LORA.value,
        ModelTypesEnum.PHI4_CLASSIFICATION_LORA.value,
    ]:
        logger.info("Running inference with fine-tuned HF model")
        metrics = hf_model_inference(cfg, logger)
    elif cfg.experiments.model.type in [
        ModelTypesEnum.CHATGPT_4O.value,
        ModelTypesEnum.MARITACA_SABIA.value,
        ModelTypesEnum.DEEPSEEK_R1.value,
    ]:
        logger.info("Running inference with API model")
        metrics = api_inference_pipeline(cfg, logger)
    else:
        raise ValueError(f"Unsupported model type: {cfg.experiments.model.type}")

    # Save metrics to CSV
    experiment_id = get_experiment_id(cfg)
    save_evaluation_results_to_csv(
        experiment_id,
        metrics,
        current_time,
    )

    # Log results
    logger.info(f"Inference results: {metrics}")

    return metrics


@hydra.main(version_base="1.1", config_path="../configs/", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point for inference.
    """
    original_output_dir = os.getcwd()
    logger.info("Starting inference experiment")
    logger.info(OmegaConf.to_yaml(cfg))

    # Set random seed for reproducibility
    set_seed(cfg.training_params.seed)
    torch.manual_seed(cfg.training_params.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training_params.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Run inference
    run_inference(cfg, logger)

    logger.info("Inference experiment completed")

    # We need to close all file handlers to release the log files
    # This is crucial for Windows where open files prevent directory renaming
    handlers = list(logger.handlers)
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)

    # Also close the root logger's handlers
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        handler.close()
        root_logger.removeHandler(handler)

    # On Windows, we need a different approach since the file system
    # sometimes keeps handles open even after closing loggers
    experiment_id = get_experiment_id(cfg)
    parent_dir = os.path.dirname(original_output_dir)
    new_dir_name = os.path.join(parent_dir, experiment_id)

    # Check if the target directory already exists
    if os.path.exists(new_dir_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_dir_name = f"{new_dir_name}_{timestamp}"

    try:
        # Instead of moving, copy files to the new location
        os.makedirs(new_dir_name, exist_ok=True)

        # Copy files one by one, skipping any that might be locked
        for item in os.listdir(original_output_dir):
            src_path = os.path.join(original_output_dir, item)
            dst_path = os.path.join(new_dir_name, item)

            try:
                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)
                elif os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path)
            except Exception as e:
                print(f"Failed to copy {item}: {e}")

        print(f"Files copied to: {new_dir_name}")

        # We don't try to delete the original directory since it might still be locked
        print(f"Original directory remains at: {original_output_dir}")

    except Exception as e:
        print(f"Failed to create output directory: {e}")


if __name__ == "__main__":
    main()
