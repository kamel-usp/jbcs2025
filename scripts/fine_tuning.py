# train.py
import sys
from functools import partial
from logging import Logger
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import load_dataset
from omegaconf import DictConfig
from preprocess import load_tokenizer, tokenize_dataset
from sklearn.utils.class_weight import compute_class_weight
from transformers import EarlyStoppingCallback, TrainingArguments

from trainer.weighted_class_trainer import WeightedLossTrainer

# Append the parent directory to sys.path using pathlib
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from models.fine_tuning_models.model_factory import ModelFactory  # NOQA
from metrics.metrics import compute_metrics  # NOQA

transformers.logging.set_verbosity_info()
LOGGING_LEVEL = "info"
EVALUATION_STRATEGY = "epoch"
SAVE_TOTAL_LIMIT = 1
REPORT_TO_LIST = ["tensorboard"]


def fine_tune_pipeline(experiment_config: DictConfig, logger: Logger):
    # Load the dataset
    dataset = load_dataset(
        experiment_config.dataset.name,
        experiment_config.dataset.split,
        cache_dir=experiment_config.cache_dir,
    )

    # Load the tokenizer
    tokenizer = load_tokenizer(
        experiment_config.experiments.model.type,
        experiment_config.experiments.model.name,
        cache_dir=experiment_config.cache_dir,
    )

    # Tokenize the dataset
    tokenized_dataset = tokenize_dataset(
        dataset,
        tokenizer,
        text_column="essay_text",
        grade_index=experiment_config.experiments.dataset.grade_index,
        model_type=experiment_config.experiments.model.type,
        logger=logger,
    )

    # Load the model
    model = ModelFactory.create_model(experiment_config, logger)
    train_batch_size = experiment_config.experiments.training_params.train_batch_size
    gradient_acc_steps = (
        experiment_config.experiments.training_params.gradient_accumulation_steps
    )
    eval_batch_size = experiment_config.experiments.training_params.eval_batch_size
    num_train_epochs = experiment_config.training_params.num_train_epochs
    # Set up training arguments
    training_args = TrainingArguments(
        seed=experiment_config.training_params.seed,
        data_seed=experiment_config.training_params.seed,
        output_dir=str(Path(experiment_config.experiments.model.output_dir).resolve()),
        eval_strategy=EVALUATION_STRATEGY,
        # Fine Tuning Related
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_acc_steps,
        gradient_checkpointing=experiment_config.experiments.training_params.gradient_checkpointing,
        warmup_ratio=experiment_config.experiments.training_params.warmup_ratio,
        learning_rate=experiment_config.experiments.training_params.learning_rate,
        num_train_epochs=num_train_epochs,
        weight_decay=experiment_config.experiments.training_params.weight_decay,
        # For logging and saving
        logging_dir=str(
            Path(experiment_config.experiments.model.logging_dir).resolve()
        ),
        logging_steps=experiment_config.training_params.logging_steps,
        log_level=LOGGING_LEVEL,
        save_strategy=EVALUATION_STRATEGY,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model=experiment_config.training_params.metric_for_best_model,
        bf16=experiment_config.training_params.bf16,
        report_to=REPORT_TO_LIST,
    )
    effective_batch_size = train_batch_size * gradient_acc_steps
    steps_per_epoch = len(tokenized_dataset["train"]) // effective_batch_size
    total_steps = steps_per_epoch * num_train_epochs
    logger.info(
        f"Total steps: {total_steps}. Number of warmup steps: {training_args.get_warmup_steps(total_steps)}"
    )
    compute_metrics_partial = partial(compute_metrics, cfg=experiment_config)
    train_labels = tokenized_dataset["train"]["label"]
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=np.array(train_labels),
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    # Initialize the Trainer
    trainer = WeightedLossTrainer(
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
        class_weights=class_weights_tensor,
    )
    # Start training
    return trainer, tokenized_dataset


if __name__ == "__main__":
    fine_tune_pipeline()
