# train.py
import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from preprocess import load_tokenizer, tokenize_dataset

# Append the parent directory to sys.path using pathlib
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from models.fine_tuning_models.model import load_model #NOQA

EVALUATION_STRATEGY = "epoch"

@hydra.main(version_base="1.1", config_path="../configs", config_name="config.yaml")
def fine_tune_pipeline(cfg: DictConfig):
    # Load the dataset
    dataset = load_dataset(cfg.dataset.name, cfg.dataset.split, cache_dir=cfg.cache_dir)

    # Load the tokenizer
    tokenizer = load_tokenizer(cfg.experiments.model.name, cache_dir=cfg.cache_dir)

    # Tokenize the dataset
    tokenized_dataset = tokenize_dataset(
        dataset,
        tokenizer,
        text_column="essay_text",
        grade_index=cfg.experiments.dataset.grade_index,
    )

    # Load the model
    model = load_model(cfg)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=cfg.experiments.model.output_dir,
        evaluation_strategy=EVALUATION_STRATEGY,
        learning_rate=cfg.training_params.learning_rate,
        per_device_train_batch_size=cfg.training_params.train_batch_size,
        per_device_eval_batch_size=cfg.training_params.eval_batch_size,
        num_train_epochs=cfg.training_params.num_train_epochs,
        weight_decay=cfg.training_params.weight_decay,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    fine_tune_pipeline()
