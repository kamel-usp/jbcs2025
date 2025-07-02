import json
import logging
import os
import shutil
import sys
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from omegaconf import DictConfig, OmegaConf
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm
from transformers import TrainingArguments, set_seed

# Append the parent directory to sys.path using pathlib
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Register OmegaConf resolvers BEFORE hydra.main is called
from utils.gpu.estimate_gpu_size import estimate_vram_gib  # NOQA
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
    repo = cfg.experiments.model.name
    logger.info("Loading model from: %s", repo)
    inference_gib, training_gib = estimate_vram_gib(repo)
    logger.info(
        f"Model need ≈ {inference_gib:.2f} GiB to run inference and {training_gib:.2f} for training "
    )

    model = ModelFactory.create_model(cfg, logger)

    return model


def compute_bootstrap_confidence_intervals(
    predictions: np.ndarray,
    labels: np.ndarray,
    dataset_test: Dataset,
    metrics_to_compute: List[str],
    cfg: DictConfig,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = None,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Compute bootstrap confidence intervals for specified metrics.

    Parameters:
        predictions: Model predictions (can be logits or predicted classes)
        labels: Ground truth labels
        dataset_test: Test dataset containing reference information
        metrics_to_compute: List of metric names to compute CIs for
        cfg: Configuration object
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default 0.95 for 95% CI)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary mapping metric names to (mean, lower_bound, upper_bound)
    """
    if random_state is not None:
        np.random.seed(random_state)


    n_samples = len(predictions)
    bootstrap_metrics = {metric: [] for metric in metrics_to_compute}

    # Extract grade_index from config
    grade_index = cfg.experiments.dataset.grade_index

    # Parse reference field and grades to get grader A and B labels
    grader_a_labels = []
    grader_b_labels = []
    
    # Get reference and grades fields
    references = dataset_test["reference"]
    all_grades = dataset_test["grades"]  # This should contain the grades array for each essay
    
    for i in range(len(references)):
        ref = references[i]
        grades = all_grades[i]
        
        # The grades array contains 6 values: [grader_a scores for C1-C5, total_grader_a]
        # For grader_a: indices 0-4 are C1-C5 scores
        # For grader_b: we need to get this from somewhere else or use the same structure
        
        if ref == "grader_a":
            # Use grader_a's score for the specified grade_index
            grader_a_labels.append(grades[grade_index])
            # For grader_b, we need to use the actual label from the dataset
            # which should be the consensus or the other grader's score
            grader_b_labels.append(labels[i])
        elif ref == "grader_b":
            # Use the label as grader_b's score
            grader_b_labels.append(labels[i])
            # Use grader_a's score from grades array
            grader_a_labels.append(grades[grade_index])
        else:
            raise ValueError(
                f"Unexpected reference value: {ref}. Expected 'grader_a' or 'grader_b'."
            )

    grader_a_labels = np.array(grader_a_labels)
    grader_b_labels = np.array(grader_b_labels)

    # Perform bootstrap sampling
    for _ in tqdm(range(n_bootstrap), desc="Performing Bootstrap samples"):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_predictions = predictions[indices]
        boot_labels_a = grader_a_labels[indices]
        boot_labels_b = grader_b_labels[indices]

        # Compute metrics for grader A
        metrics_a = compute_metrics((boot_predictions, boot_labels_a), cfg)

        # Compute metrics for grader B
        metrics_b = compute_metrics((boot_predictions, boot_labels_b), cfg)

        # Average metrics from both graders
        for metric in metrics_to_compute:
            if metric in metrics_a and metric in metrics_b:
                avg_metric = (metrics_a[metric] + metrics_b[metric]) / 2.0
                bootstrap_metrics[metric].append(avg_metric)

    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_results = {}
    for metric, values in bootstrap_metrics.items():
        if values:  # Check if we have values for this metric
            values_array = np.array(values)
            mean_val = np.mean(values_array)
            lower_bound = np.percentile(values_array, lower_percentile)
            upper_bound = np.percentile(values_array, upper_percentile)
            ci_results[metric] = (mean_val, lower_bound, upper_bound)

    return ci_results


def save_bootstrap_ci_to_csv(
    experiment_id: str,
    ci_results: Dict[str, Tuple[float, float, float]],
    timestamp: str,
    csv_filename: str = "bootstrap_confidence_intervals.csv",
) -> None:
    """
    Save bootstrap confidence interval results to CSV.
    If experiment_id exists, overwrite that row; otherwise append.

    Parameters:
        experiment_id: Unique identifier for the experiment
        ci_results: Dictionary of metric names to (mean, lower, upper) tuples
        timestamp: Timestamp of the experiment
        csv_filename: Name of the output CSV file
    """
    # Prepare data for the new row
    new_row = {
        "experiment_id": experiment_id,
        "timestamp": timestamp,
    }

    # Add CI results
    for metric, (mean_val, lower, upper) in ci_results.items():
        new_row[f"{metric}_mean"] = mean_val
        new_row[f"{metric}_lower_95ci"] = lower
        new_row[f"{metric}_upper_95ci"] = upper
        new_row[f"{metric}_ci_width"] = upper - lower

    # Check if CSV exists
    if os.path.exists(csv_filename):
        # Read existing data
        df = pd.read_csv(csv_filename)

        # Check if experiment_id already exists
        if experiment_id in df["experiment_id"].values:
            # Overwrite the existing row
            df = df[df["experiment_id"] != experiment_id]
            logger.info(
                f"Overwriting existing CI results for experiment: {experiment_id}"
            )

        # Append the new row
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        # Create new dataframe
        df = pd.DataFrame([new_row])

    # Save to CSV
    df.to_csv(csv_filename, index=False)
    logger.info(f"Bootstrap CI results saved to {csv_filename}")


def hf_model_inference(
    cfg: DictConfig, logger: Logger
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, Dataset]:
    """
    Run inference with a Hugging Face model loaded from the hub.
    Returns metrics, raw predictions/labels, and test dataset for CI computation.
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
        use_full_context=cfg.experiments.dataset.use_full_context,
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

    return metrics, logits, labels, dataset["test"]


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
        metrics, predictions, labels, dataset_test = hf_model_inference(cfg, logger)
    elif cfg.experiments.model.type in [
        ModelTypesEnum.CHATGPT_4O.value,
        ModelTypesEnum.MARITACA_SABIA.value,
        ModelTypesEnum.DEEPSEEK_R1.value,
    ]:
        logger.info("Running inference with API model")
        metrics, predictions, labels, dataset_test = api_inference_pipeline(cfg, logger)
    else:
        raise ValueError(f"Unsupported model type: {cfg.experiments.model.type}")

    # Save metrics to CSV
    experiment_id = get_experiment_id(cfg)
    save_evaluation_results_to_csv(
        experiment_id,
        metrics,
        current_time,
    )

    # Compute bootstrap confidence intervals if specified
    if hasattr(cfg, "bootstrap") and cfg.bootstrap.enabled:
        metrics_for_ci = cfg.bootstrap.get("metrics", ["QWK"])
        n_bootstrap = cfg.bootstrap.get("n_bootstrap", 1000)

        logger.info(
            f"Computing bootstrap confidence intervals for metrics: {metrics_for_ci}"
        )
        ci_results = compute_bootstrap_confidence_intervals(
            predictions=predictions,
            labels=labels,
            dataset_test=dataset_test,
            metrics_to_compute=metrics_for_ci,
            cfg=cfg,
            n_bootstrap=n_bootstrap,
            random_state=cfg.training_params.seed,
        )

        # Save CI results
        save_bootstrap_ci_to_csv(
            experiment_id=experiment_id,
            ci_results=ci_results,
            timestamp=current_time,
        )

        # Log CI results
        logger.info("Bootstrap Confidence Intervals (95%):")
        for metric, (mean_val, lower, upper) in ci_results.items():
            logger.info(f"  {metric}: {mean_val:.4f} [{lower:.4f}, {upper:.4f}]")

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
