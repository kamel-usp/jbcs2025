import csv
import logging
import os

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, root_mean_squared_error

from models.fine_tuning_models.model_types_enum import ModelTypesEnum


def enem_accuracy_score(true_values, predicted_values):
    assert len(true_values) == len(predicted_values), (
        "Mismatched length between true and predicted values."
    )

    non_divergent_count = sum(
        [1 for t, p in zip(true_values, predicted_values) if abs(t - p) <= 80]
    )

    return non_divergent_count / len(true_values)


def compute_metrics(eval_pred, cfg):
    transformers_logger = logging.getLogger("transformers")
    if cfg.experiments.model.type in [
        ModelTypesEnum.ENCODER_CLASSIFICATION.value,
        ModelTypesEnum.LLAMA31_CLASSIFICATION_LORA.value,
        ModelTypesEnum.PHI35_CLASSIFICATION_LORA.value,
        ModelTypesEnum.PHI4_CLASSIFICATION_LORA.value,
    ]:
        logits, all_true_labels = eval_pred
        all_predictions = np.argmax(logits, axis=1)
        all_true_labels = list(map(lambda x: x * 40, all_true_labels))
        all_predictions = list(map(lambda x: x * 40, all_predictions))
    elif cfg.experiments.model.type in [
        ModelTypesEnum.CHATGPT_4O.value,
        ModelTypesEnum.MARITACA_SABIA.value,
        ModelTypesEnum.DEEPSEEK_R1.value,
    ]:
        all_predictions, all_true_labels = eval_pred
    # elif model.config.problem_type == "regression":
    #     rounded_tensor = np.round(logits)
    #     # Clamp the values to the range [0, 5]
    #     clamped_tensor = np.clip(rounded_tensor, a_min=0, a_max=5)
    #     all_predictions = np.argmax(clamped_tensor, axis=1)
    else:
        raise AttributeError("problem_type from model.config is None!")
    # all_predictions = corn_label_from_logits(torch.from_numpy(logits))
    # revert back
    # Initialize the metrics
    accuracy = accuracy_score(all_true_labels, all_predictions)
    qwk = cohen_kappa_score(
        all_true_labels,
        all_predictions,
        weights="quadratic",
        labels=[0, 40, 80, 120, 160, 200],
    )
    rmse = root_mean_squared_error(all_true_labels, all_predictions)
    horizontal_discrepancy = enem_accuracy_score(all_true_labels, all_predictions)
    results = {
        "accuracy": float(accuracy),
        "RMSE": float(rmse),
        "QWK": float(qwk),
        "HDIV": float(1 - horizontal_discrepancy),
    }
    transformers_logger.info(results)
    return results


def save_evaluation_results_to_csv(
    training_id, evaluation_results, timestamp, file_path="evaluation_results.csv"
):
    # Add a timestamp to the evaluation results
    evaluation_results_with_timestamp = evaluation_results.copy()
    evaluation_results_with_timestamp["timestamp"] = timestamp
    evaluation_results_with_timestamp["id"] = training_id

    # Determine if we need to write headers
    write_headers = not os.path.exists(file_path)

    with open(file_path, "a", newline="") as csvfile:
        fieldnames = list(evaluation_results_with_timestamp.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_headers:
            writer.writeheader()
        writer.writerow(evaluation_results_with_timestamp)
