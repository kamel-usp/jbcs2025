import logging
import os
import shutil
import sys
from pathlib import Path

import hydra
import pandas as pd
import yaml
from dotenv import load_dotenv
from huggingface_hub import HfApi, login, upload_folder
from omegaconf import DictConfig, OmegaConf

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from models.fine_tuning_models.model_types_enum import ModelTypesEnum  # NOQA: E402
from scripts.run_experiment import get_experiment_id  # NOQA: E402

# Load environment variables from .env file
load_dotenv()


def parse_experiment_config_from_log(log_path: Path) -> DictConfig:
    """
    Parse the experiment configuration from run_experiment.log file.
    The config is in YAML format at the beginning of the log file.
    """
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    # Find the start of the YAML config (after the first log line)
    yaml_lines = []
    in_yaml = False
    
    for line in lines:
        # Skip the first log line with timestamp
        if not in_yaml and line.strip() and not line.startswith('['):
            in_yaml = True
        
        # Stop when we hit another log line with timestamp
        if in_yaml and line.startswith('['):
            break
            
        if in_yaml:
            yaml_lines.append(line)
    
    # Parse the YAML content
    yaml_content = ''.join(yaml_lines)
    config_dict = yaml.safe_load(yaml_content)
    
    # Convert to OmegaConf DictConfig
    return OmegaConf.create(config_dict)


def create_model_card(experiment_cfg: DictConfig, model_dir: Path, logger: logging.Logger):
    """
    Create a README.md file as a model card if one doesn't already exist.
    """
    results = pd.read_csv(model_dir / "evaluation_results.csv")
    results.rename(
        index={0: "val_before_train", 1: "val_after_train", 2: "test_data"},
        inplace=True,
    )
    test_series = results.iloc[-1]
    model_location = model_dir / experiment_cfg.experiments.model.output_dir / "best_model"
    model_card_path = model_location / "README.md"
    logger.info(f"README location: {model_card_path}")
    main_library = None
    experiment_id = get_experiment_id(experiment_cfg)
    if experiment_cfg.experiments.model.type == ModelTypesEnum.ENCODER_CLASSIFICATION.value:
        main_library = "transformers"
    elif experiment_cfg.experiments.model.type in [
        ModelTypesEnum.LLAMA31_CLASSIFICATION_LORA.value,
        ModelTypesEnum.PHI35_CLASSIFICATION_LORA.value,
        ModelTypesEnum.PHI4_CLASSIFICATION_LORA.value,
    ]:
        main_library = "peft"
    columns_to_use = [
        "eval_accuracy",
        "eval_RMSE",
        "eval_QWK",
        "eval_Macro_F1",
        "eval_Weighted_F1",
        "eval_Micro_F1",
        "eval_HDIV",
    ]
    model_card_content = f"""
---
language:
- pt
- en
tags:
- aes
datasets:
- kamel-usp/aes_enem_dataset
base_model: {experiment_cfg.experiments.model.name}
metrics:
- accuracy
- qwk
library_name: {main_library}
model-index:
  - name: {experiment_id}
    results:
      - task:
          type: text-classification
          name: Automated Essay Score
        dataset:
          name: Automated Essay Score ENEM Dataset
          type: kamel-usp/aes_enem_dataset
          config: JBCS2025
          split: test
        metrics:
          - name: Macro F1
            type: f1
            value: {test_series["eval_Macro_F1"]}
          - name: QWK
            type: qwk
            value: {test_series["eval_QWK"]}
          - name: Weighted Macro F1
            type: f1
            value: {test_series["eval_Weighted_F1"]}
---
# Model ID: {experiment_id}
## Results
{test_series[columns_to_use].to_markdown()}
        """
    try:
        with open(model_card_path, "w") as f:
            f.write(model_card_content)
            logger.info("Model card (README.md) created successfully.")
    except Exception as error_message:
        logger.error(f"Failed to create model card: {error_message}")
        raise error_message


def push_model_to_hf(cfg: DictConfig, logger: logging.Logger):
    # Use the new model_push configuration
    model_dir = Path(cfg.model_push.model_path)
    
    # Parse experiment configuration from run_experiment.log
    log_path = model_dir / "run_experiment.log"
    if not log_path.exists():
        raise FileNotFoundError(f"run_experiment.log not found at {log_path}")
    
    logger.info(f"Parsing experiment configuration from {log_path}")
    experiment_cfg = parse_experiment_config_from_log(log_path)
    
    best_dir = model_dir / experiment_cfg.experiments.model.output_dir / "best_model"
    experiment_id = get_experiment_id(experiment_cfg)
    logger.info(f"Detected experiment ID: {experiment_id}")
    
    # Use organization from config
    org = cfg.model_push.organization
    repo_name = f"jbcs2025_{experiment_id}"
    full_repo_id = f"{org}/{repo_name}"

    api = HfApi()

    # Create (or reuse) the repository on the Hugging Face Hub under the organization.
    try:
        api.create_repo(repo_id=full_repo_id, exist_ok=True)
        logger.info(f"Repository '{full_repo_id}' created or already exists.")
    except Exception as e:
        logger.error(f"Error creating repository '{full_repo_id}': {e}")
        raise e

    create_model_card(experiment_cfg, model_dir, logger)

    # Copy additional files from model_dir to best_dir so they can be uploaded
    files_to_copy = [
        "run_experiment.log",
        "evaluation_results.csv",
        "emissions.csv"
    ]
    
    for filename in files_to_copy:
        source_file = model_dir / filename
        destination_file = best_dir / filename
        
        if source_file.exists():
            try:
                shutil.copy(str(source_file), str(destination_file))
                logger.info(f"{filename} copied to best_dir successfully.")
            except Exception as error_message:
                logger.error(f"Failed to copy {filename} to best_dir: {error_message}")
                # Continue with other files even if one fails
        else:
            logger.warning(f"{filename} not found at {source_file}")

    # Upload the model folder to the repository using the HTTP-based upload_folder.
    try:
        upload_folder(
            folder_path=str(best_dir),
            repo_id=full_repo_id,
            commit_message="Pushing fine-tuned model to Hugging Face Hub",
        )
        logger.info(f"Model successfully pushed to '{full_repo_id}'.")
    except Exception as e:
        logger.error(f"Error uploading folder to '{full_repo_id}': {e}")
        raise e

    # Use collection details from config
    collection_slug = cfg.model_push.collection.slug
    collection_name = cfg.model_push.collection.name

    # Attempt to retrieve the collection; if not found, create a new one.
    try:
        _ = api.get_collection(collection_slug)
        logger.info(
            f"Collection '{collection_name}' found in organization '{org}' under slug: {collection_slug}"
        )
    except Exception as _:
        logger.info(
            f"Collection '{collection_name}' not found. Creating new collection."
        )
        _ = api.create_collection(title=collection_name, namespace=org)

    # Add the repository to the collection.
    try:
        api.add_collection_item(
            collection_slug=collection_slug, item_id=full_repo_id, item_type="model"
        )
        logger.info(
            f"Repository '{full_repo_id}' added to collection '{collection_name}'."
        )
    except Exception as error_message:
        logger.error(
            f"Error adding repository to collection '{collection_name}': {error_message}"
        )
        raise error_message


@hydra.main(version_base="1.1", config_path="../configs/", config_name="push_model_config")
def main(cfg: DictConfig):
    # Set up logging
    logger = logging.getLogger(__name__)
    
    # Get HF token from environment variable
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN environment variable not set. Please set it in your .env file."
        )
    
    # Login with token from environment
    login(token=hf_token)
    logger.info("Successfully logged in to Hugging Face Hub")
    
    push_model_to_hf(cfg, logger)


if __name__ == "__main__":
    main()
