import json
import logging
import sys
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Dict, List

import hydra
import torch
from datasets import load_dataset, Dataset
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed

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
    logger.info(f"Loading model from: {cfg.inference.model_id}")

    # Create the model using ModelFactory which should handle HF Hub loading
    model = ModelFactory.create_model(
        cfg, logger, pretrained_model_name_or_path=cfg.inference.model_id
    )

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
        cfg.experiments.model.name,
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

    # Create trainer
    trainer = WeightedLossTrainer(
        model=model,
        args=None,  # We're only doing inference, no training args needed
    )

    # Run inference
    logger.info("Running inference on test dataset")
    test_results = trainer.predict(tokenized_dataset["test"])
    predictions = test_results.predictions.argmax(-1)
    labels = test_results.label_ids

    # Calculate metrics
    metrics = compute_metrics((predictions, labels), cfg)

    # Save inference results
    save_model_predictions_jsonl(
        dataset_test=dataset["test"],
        predictions=predictions,
        labels=labels,
        grade_index=grade_index,
        experiment_id=cfg.experiments.inference_id,
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
    metrics["reference"] = "inference_results"
    save_evaluation_results_to_csv(
        cfg.experiments.inference_id,
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
    logger.info("Starting inference experiment")
    logger.info(OmegaConf.to_yaml(cfg))

    # Set random seed for reproducibility
    set_seed(cfg.inference.seed)
    torch.manual_seed(cfg.inference.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.inference.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Run inference
    run_inference(cfg, logger)

    logger.info("Inference experiment completed")


if __name__ == "__main__":
    main()
