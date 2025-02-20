import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import hydra
import numpy as np
import torch
from fine_tuning import fine_tune_pipeline
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed

# Append the parent directory to sys.path using pathlib
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from metrics.metrics import save_evaluation_results_to_csv  # NOQA

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.INFO)
transformers_logger.propagate = True
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.1", config_path="../configs/", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))

    set_seed(cfg.training_params.seed)
    torch.manual_seed(cfg.training_params.seed)
    np.random.seed(cfg.training_params.seed)
    random.seed(cfg.training_params.seed)

    # If using CUDA
    torch.cuda.manual_seed_all(cfg.training_params.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    logger.info("Starting the training process.")
    trainer, tokenized_dataset = fine_tune_pipeline(cfg, logger)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    evaluate_baseline = trainer.evaluate()
    save_evaluation_results_to_csv(
        cfg.experiments.training_id,
        evaluate_baseline,
        current_time,
        "baseline_evaluation",
    )
    trainer.train()
    evaluate_after_training = trainer.evaluate()
    save_evaluation_results_to_csv(
        cfg.experiments.training_id,
        evaluate_after_training,
        current_time,
        "evaluation_after_training",
    )
    logger.info("Training completed successfully.")
    logger.info("Running on Test")
    evaluate_test = trainer.evaluate(tokenized_dataset["test"])
    save_evaluation_results_to_csv(
        cfg.experiments.training_id,
        evaluate_test,
        current_time,
        "test_set_after_training",
    )
    trainer.save_model(cfg.experiments.model.best_model_dir)
    logger.info("Fine Tuning Finished.")


if __name__ == "__main__":
    main()
