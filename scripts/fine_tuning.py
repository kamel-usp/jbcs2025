# train.py
import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from preprocess import load_tokenizer, tokenize_dataset
from functools import partial

# Append the parent directory to sys.path using pathlib
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from models.fine_tuning_models.classification_head import load_model  # NOQA
from metrics.metrics import compute_metrics  # NOQA

EVALUATION_STRATEGY = "epoch"
SAVE_TOTAL_LIMIT = 1
REPORT_TO_LIST = ["none"]

@hydra.main(version_base="1.1", config_path="../configs", config_name="config.yaml")
def fine_tune_pipeline(experiment_config: DictConfig):
    # Load the dataset
    dataset = load_dataset(
        experiment_config.dataset.name,
        experiment_config.dataset.split,
        cache_dir=experiment_config.cache_dir,
    )

    # Load the tokenizer
    tokenizer = load_tokenizer(
        experiment_config.experiments.model.name, cache_dir=experiment_config.cache_dir
    )

    # Tokenize the dataset
    tokenized_dataset = tokenize_dataset(
        dataset,
        tokenizer,
        text_column="essay_text",
        grade_index=experiment_config.experiments.dataset.grade_index,
    )

    # Load the model
    model = load_model(experiment_config)

    # Set up training arguments
    training_args = TrainingArguments(
        seed=experiment_config.training_params.seed,
        data_seed=experiment_config.training_params.seed,
        output_dir=experiment_config.experiments.model.output_dir,
        evaluation_strategy=EVALUATION_STRATEGY,
        # Fine Tuning Related
        per_device_train_batch_size=experiment_config.training_params.train_batch_size,
        per_device_eval_batch_size=experiment_config.training_params.eval_batch_size,
        gradient_accumulation_steps=experiment_config.training_params.gradient_accumulation_steps,
        gradient_checkpointing=experiment_config.training_params.gradient_checkpointing,
        warmup_steps=experiment_config.training_params.warmup_steps,
        learning_rate=experiment_config.training_params.learning_rate,
        num_train_epochs=experiment_config.training_params.num_train_epochs,
        weight_decay=experiment_config.training_params.weight_decay,
        # For logging and saving
        logging_dir=experiment_config.experiments.model.logging_dir,
        logging_steps=experiment_config.training_params.logging_steps,
        save_strategy=EVALUATION_STRATEGY,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model=experiment_config.training_params.metric_for_best_model,
        bf16=experiment_config.training_params.bf16,
        report_to=REPORT_TO_LIST,
    )

    compute_metrics_partial = partial(compute_metrics, model=model)

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        args=training_args,
        compute_metrics=compute_metrics_partial,
        callbacks=(
            [EarlyStoppingCallback(early_stopping_patience=5)]
            if tokenized_dataset["validation"]
            else []
        ),
    )
    # Start training
    return trainer, tokenized_dataset


if __name__ == "__main__":
    fine_tune_pipeline()
