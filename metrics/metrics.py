import csv
import logging
import os
from collections import OrderedDict
from typing import Tuple, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    root_mean_squared_error,
)

from models.fine_tuning_models.model_types_enum import ModelTypesEnum
from trainer.prediction_decoder import PredictionDecoder 

ALL_LABELS = [0, 40, 80, 120, 160, 200]


def enem_accuracy_score(true_values, predicted_values):
    assert len(true_values) == len(predicted_values), (
        "Mismatched length between true and predicted values."
    )

    non_divergent_count = sum(
        [1 for t, p in zip(true_values, predicted_values) if abs(t - p) <= 80]
    )

    return non_divergent_count / len(true_values)


def _is_api_model(model_type: str) -> bool:
    """Check if the model type is an API model."""
    api_models = {
        ModelTypesEnum.CHATGPT_4O.value,
        ModelTypesEnum.MARITACA_SABIA.value,
        ModelTypesEnum.DEEPSEEK_R1.value,
    }
    return model_type in api_models


def _is_classification_model(model_type: str) -> bool:
    """Check if the model type is a classification model (including ordinal)."""
    # Check if it contains any classification-related keywords
    model_type_upper = model_type.upper()
    return any(keyword in model_type_upper for keyword in ["CLASSIFICATION", "ORDINAL"])


def _process_predictions(eval_pred, model_type: str) -> Tuple[List[int], List[int]]:
    """Process predictions based on model type."""
    if _is_api_model(model_type):
        # API models directly return predictions and labels
        all_predictions, all_true_labels = eval_pred
        return all_predictions, all_true_labels
    
    elif _is_classification_model(model_type):
        # Classification and ordinal models return logits
        logits, all_true_labels = eval_pred
        
        # Use PredictionDecoder to handle both standard and ordinal predictions
        all_predictions = PredictionDecoder.decode(logits, model_type)
        
        # Ensure true labels are in the correct format (original scale)
        if isinstance(all_true_labels[0], (int, np.integer)) and all_true_labels.max() <= 5:
            all_true_labels = all_true_labels * 40
        
        return all_predictions.tolist(), all_true_labels.tolist()
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def compute_metrics(eval_pred, cfg):
    """Compute evaluation metrics for the model."""
    transformers_logger = logging.getLogger("transformers")
    model_type = cfg.experiments.model.type
    
    try:
        # Process predictions based on model type
        all_predictions, all_true_labels = _process_predictions(eval_pred, model_type)
        
        # Compute metrics
        metrics = _calculate_metrics(all_true_labels, all_predictions)
        
        # Log results
        transformers_logger.info(metrics)
        
        return metrics
        
    except Exception as e:
        transformers_logger.error(f"Error computing metrics: {str(e)}")
        raise


def _calculate_metrics(true_labels: List[int], predictions: List[int]) -> dict:
    """Calculate all evaluation metrics."""
    # Basic metrics
    accuracy = accuracy_score(true_labels, predictions)
    qwk = cohen_kappa_score(
        true_labels,
        predictions,
        weights="quadratic",
        labels=ALL_LABELS,
    )
    rmse = root_mean_squared_error(true_labels, predictions)
    horizontal_discrepancy = enem_accuracy_score(true_labels, predictions)
    
    # F1 scores
    macro_f1 = f1_score(
        true_labels,
        predictions,
        average="macro",
        labels=ALL_LABELS,
        zero_division=np.nan,
    )
    micro_f1 = f1_score(
        true_labels,
        predictions,
        average="micro",
        labels=ALL_LABELS,
        zero_division=np.nan,
    )
    weighted_f1 = f1_score(
        true_labels,
        predictions,
        average="weighted",
        labels=ALL_LABELS,
        zero_division=np.nan,
    )

    results = {
        "accuracy": float(accuracy),
        "RMSE": float(rmse),
        "QWK": float(qwk),
        "HDIV": float(1 - horizontal_discrepancy),
        "Macro_F1": macro_f1,
        "Micro_F1": micro_f1,
        "Weighted_F1": weighted_f1,
    }
    
    # Add confusion matrix metrics
    results.update(_calculate_confusion_matrix_metrics(true_labels, predictions))
    
    return results


def _calculate_confusion_matrix_metrics(true_labels: List[int], predictions: List[int]) -> dict:
    """Calculate per-class confusion matrix metrics."""
    cm = confusion_matrix(true_labels, predictions, labels=ALL_LABELS)
    n_classes = cm.shape[0]
    
    metrics = {}
    for i in range(n_classes):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - (TP + FP + FN)
        
        metrics.update({
            f"TP_{i}": TP,
            f"TN_{i}": TN,
            f"FP_{i}": FP,
            f"FN_{i}": FN,
        })
    
    return metrics


def save_evaluation_results_to_csv(
    training_id, evaluation_results, timestamp, file_path="evaluation_results.csv"
):
    """Save evaluation results to a CSV file."""
    # Add metadata to results
    evaluation_results_with_metadata = evaluation_results.copy()
    evaluation_results_with_metadata.update({
        "timestamp": timestamp,
        "id": training_id,
    })
    
    ordered_dict = OrderedDict(evaluation_results_with_metadata)

    # Determine if we need to write headers
    write_headers = not os.path.exists(file_path)

    with open(file_path, "a", newline="") as csvfile:
        fieldnames = list(ordered_dict.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_headers:
            writer.writeheader()
        writer.writerow(ordered_dict)