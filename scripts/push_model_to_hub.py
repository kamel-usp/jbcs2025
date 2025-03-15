import logging
import os

import hydra
import pandas as pd
from huggingface_hub import HfApi, login, upload_folder
from omegaconf import DictConfig


def create_model_card(cfg: DictConfig, model_dir: str, logger: logging.Logger):
    """
    Create a README.md file as a model card if one doesn't already exist.
    """
    results = pd.read_csv(f"{model_dir}/evaluation_results.csv")
    results.rename(
        index={0: "val_before_train", 1: "val_after_train", 2: "test_data"},
        inplace=True,
    )
    test_series = results.iloc[-1]
    model_location = os.path.join(
        model_dir, f"{cfg.experiments.model.output_dir}/best_model"
    )
    model_card_path = os.path.join(model_location, "README.md")
    logger.info(f"README location: {model_card_path}")
    if not os.path.exists(model_card_path):
        model_card_content = f"""
---
language:
- pt
- en
tags:
- aes
datasets:
- kamel-usp/aes_enem_dataset
base_model: {cfg.experiments.model.name}
metrics:
- accuracy
- qwk
model-index:
  - name: {cfg.experiments.training_id}
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
            type: accuracy
            value: {test_series["eval_Macro F1"]}
          - name: QWK
            type: qwk
            value: {test_series["eval_QWK"]}
---
# Model ID: {cfg.experiments.training_id}
## Results
{test_series[["eval_accuracy", "eval_RMSE", "eval_QWK", "eval_Macro F1", "eval_HDIV"]].to_markdown()}
        """
        try:
            with open(model_card_path, "w") as f:
                f.write(model_card_content)
                logger.info("Model card (README.md) created successfully.")
        except Exception as e:
            logger.error(f"Failed to create model card: {e}")
            raise e
    else:
        logger.info("Model card (README.md) already exists. Skipping creation.")


def push_model_to_hf(cfg: DictConfig, logger: logging.Logger):
    # Path where your fine-tuned model is saved.
    model_dir = cfg.post_training_results.model_path
    # Organization name and repository naming
    org = "kamel-usp"
    repo_name = f"jbcs2025_{cfg.experiments.training_id}"
    full_repo_id = f"{org}/{repo_name}"

    api = HfApi()

    # Create (or reuse) the repository on the Hugging Face Hub under the organization.
    try:
        api.create_repo(repo_id=full_repo_id, exist_ok=True)
        logger.info(f"Repository '{full_repo_id}' created or already exists.")
    except Exception as e:
        logger.error(f"Error creating repository '{full_repo_id}': {e}")
        raise e

    # Optionally, create a model card if needed.
    create_model_card(cfg, model_dir, logger)

    # Upload the model folder to the repository using the HTTP-based upload_folder.
    try:
        upload_folder(
            folder_path=f"{model_dir}/{cfg.experiments.model.output_dir}/best_model",
            repo_id=full_repo_id,
            commit_message="Pushing fine-tuned model to Hugging Face Hub",
        )
        logger.info(f"Model successfully pushed to '{full_repo_id}'.")
    except Exception as e:
        logger.error(f"Error uploading folder to '{full_repo_id}': {e}")
        raise e

    # Define collection details.
    collection_slug = f"{org}/jbcs2025-67d5e73a4b89c1f0c878159c"
    collection_name = "JBCS2025"

    # Attempt to retrieve the collection; if not found, create a new one.
    try:
        _ = api.get_collection(collection_slug)
        logger.info(
            f"Collection '{collection_name}' found in organization '{org}' under slug: {collection_slug}"
        )
    except Exception as e:
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
    except Exception as e:
        logger.error(f"Error adding repository to collection '{collection_name}': {e}")
        raise e


@hydra.main(version_base="1.1", config_path="../configs/", config_name="config")
def main(cfg: DictConfig):
    # Set up logging
    logger = logging.getLogger(__name__)
    login(token="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")
    push_model_to_hf(cfg, logger)


if __name__ == "__main__":
    main()
