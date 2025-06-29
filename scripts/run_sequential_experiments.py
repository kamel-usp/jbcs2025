# python scripts/run_sequential_experiments.py configs/sequential_experiments.yaml --mode inference
import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import yaml
from tqdm.auto import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("sequential_experiments.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class SequentialExperimentRunner:
    """Run multiple experiments sequentially with configurable parameters."""

    def __init__(self, config_path: str, mode: str = "train"):
        """
        Initialize the runner.

        Args:
            config_path: Path to the configuration file
            mode: Either 'train' or 'inference'
        """
        self.config_path = Path(config_path)
        self.mode = mode
        self.load_config()

    def load_config(self):
        """Load configuration from YAML file."""
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {self.config_path}")

    def build_command(self, experiment: dict[str, any]) -> list[str]:
        """
        Build the command to run an experiment.

        Args:
            experiment: Experiment configuration

        Returns:
            Command as a list of strings
        """
        if self.mode == "train":
            script = "scripts/run_experiment.py"
        else:
            script = "scripts/run_inference_experiment.py"

        cmd = ["python", script]
        
        # Override the experiment configuration
        if "reference" in experiment:
            cmd.append(f"experiments={experiment['reference']}")

        return cmd

    def run_experiment(self, experiment: dict[str, any]) -> bool:
        """
        Run a single experiment.

        Args:
            experiment: Experiment configuration

        Returns:
            True if successful, False otherwise
        """
        experiment_name = experiment.get("name")
        logger.info(f"Starting experiment: {experiment_name}")

        cmd = self.build_command(experiment)
        logger.info(f"Command: {' '.join(cmd)}")

        try:
            # Run the experiment
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )

            # Stream output in real-time
            for line in process.stdout:
                print(line, end="")

            # Wait for completion
            return_code = process.wait()

            if return_code == 0:
                logger.info(f"Experiment {experiment_name} completed successfully")
                return True
            else:
                logger.error(
                    f"Experiment {experiment_name} failed with return code {return_code}"
                )
                return False

        except Exception as e:
            logger.error(f"Error running experiment {experiment_name}: {e}")
            return False

    def run_all(self):
        """Run all experiments sequentially."""
        experiments = self.config.get("experiments", [])
        logger.info(f"Running {len(experiments)} experiments in {self.mode} mode")

        results = []
        for i, experiment in tqdm(enumerate(experiments), desc="Running experiments"):
            logger.info(f"\n{'='*60}")
            logger.info(f"Experiment {i+1}/{len(experiments)}")
            logger.info(f"{'='*60}\n")

            success = self.run_experiment(experiment)
            results.append(
                {
                    "name": experiment.get("name"),
                    "success": success,
                }
            )

            if not success and self.config.get("stop_on_failure", True):
                logger.error("Stopping due to experiment failure")
                break

            # Add delay between experiments if specified
            delay = self.config.get("delay_between_experiments", 5)
            if i < len(experiments) - 1:
                logger.info(f"Waiting {delay} seconds before next experiment...")
                time.sleep(delay)

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("Experiment Summary:")
        logger.info(f"{'='*60}")
        for result in results:
            status = "✓" if result["success"] else "✗"
            logger.info(f"{status} {result['name']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run multiple experiments sequentially"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the sequential experiments configuration file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "inference"],
        default="train",
        help="Mode to run experiments in (default: train)",
    )

    args = parser.parse_args()

    runner = SequentialExperimentRunner(args.config, args.mode)
    runner.run_all()


if __name__ == "__main__":
    main()
