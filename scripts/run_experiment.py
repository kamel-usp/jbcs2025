import argparse
from fine_tuning import fine_tune_pipeline
from transformers import set_seed
import hydra
import torch
import random
import numpy as np

RANDOM_SEED = 42

def main():
    parser = argparse.ArgumentParser(description="Run an experiment with fine_tuning.")
    # You can add additional command line arguments here if needed
    parser.add_argument("--config-path", type=str, default="../configs",
                        help="Path to the Hydra configuration file.")
    args = parser.parse_args()

    print("Starting experiment using config:", args.config_path)
    set_seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # If using CUDA
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    hydra.initialize(config_path=args.config_path)
    cfg = hydra.compose(config_name="config")
    fine_tune_pipeline(cfg)

if __name__ == "__main__":
    main()
