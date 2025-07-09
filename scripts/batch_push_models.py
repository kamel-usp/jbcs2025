import argparse
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("batch_push_models.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def is_time_folder(folder_name: str) -> bool:
    """
    Check if a folder name matches time patterns like HH-MM-SS or similar date/time formats.
    
    Args:
        folder_name: Name of the folder to check
        
    Returns:
        True if folder name matches a time pattern, False otherwise
    """
    # Common time patterns
    time_patterns = [
        r'^\d{2}-\d{2}-\d{2}$',  # HH-MM-SS
        r'^\d{2}_\d{2}_\d{2}$',  # HH_MM_SS
        r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
        r'^\d{8}_\d{6}$',        # YYYYMMDD_HHMMSS
    ]
    
    for pattern in time_patterns:
        if re.match(pattern, folder_name):
            return True
    return False


def update_push_config(model_path: Path, config_path: Path) -> None:
    """
    Update the push_model_config.yaml file with the new model path.
    
    Args:
        model_path: Path to the model directory
        config_path: Path to the push_model_config.yaml file
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update the model path
    config['model_push']['model_path'] = str(model_path)
    
    # Write back to file
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Updated config with model path: {model_path}")


def push_model(config_path: Path) -> bool:
    """
    Run the push_model_to_hub.py script with the updated config.
    
    Args:
        config_path: Path to the push_model_config.yaml file
        
    Returns:
        True if successful, False otherwise
    """
    # Get the project root directory (parent of scripts directory)
    project_root = Path(__file__).resolve().parent.parent
    
    # Convert config_path to absolute path
    abs_config_path = config_path.resolve()
    
    cmd = [
        "python",
        "scripts/push_model_to_hub.py",
        "--config-path",
        str(abs_config_path.parent),
        "--config-name",
        abs_config_path.name
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    logger.info(f"Working directory: {project_root}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(project_root)  # Set working directory to project root
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end="")
        
        return_code = process.wait()
        
        if return_code == 0:
            logger.info("Model pushed successfully")
            return True
        else:
            logger.error(f"Push failed with return code {return_code}")
            return False
            
    except Exception as e:
        logger.error(f"Error pushing model: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch push models to Hugging Face Hub"
    )
    parser.add_argument(
        "date_folder",
        type=str,
        help="Path to the date folder containing model directories (e.g., /workspace/jbcs2025/outputs/2025-07-04)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/push_model_config.yaml",
        help="Path to the push model configuration file (default: configs/push_model_config.yaml)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=2,
        help="Delay in seconds between pushes (default: 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be pushed without actually pushing",
    )
    
    args = parser.parse_args()
    
    date_folder = Path(args.date_folder)
    config_path = Path(args.config)
    
    if not date_folder.exists():
        logger.error(f"Date folder does not exist: {date_folder}")
        sys.exit(1)
    
    if not config_path.exists():
        logger.error(f"Config file does not exist: {config_path}")
        sys.exit(1)
    
    # Find all subdirectories that are not time-formatted
    model_dirs = []
    for item in date_folder.iterdir():
        if item.is_dir() and not is_time_folder(item.name):
            # Check if it contains the required files
            if (item / "run_experiment.log").exists() and (item / "evaluation_results.csv").exists():
                model_dirs.append(item)
            else:
                logger.warning(f"Skipping {item.name} - missing required files")
    
    logger.info(f"Found {len(model_dirs)} model directories to push")
    
    if args.dry_run:
        logger.info("DRY RUN - Models that would be pushed:")
        for model_dir in model_dirs:
            logger.info(f"  - {model_dir.name}")
        return
    
    # Process each model
    results = []
    for i, model_dir in enumerate(tqdm(model_dirs, desc="Pushing models")):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {i+1}/{len(model_dirs)}: {model_dir.name}")
        logger.info(f"{'='*60}\n")
        
        try:
            # Update config file
            update_push_config(model_dir, config_path)
            
            # Push the model
            success = push_model(config_path)
            
            results.append({
                "model": model_dir.name,
                "success": success
            })
            
        except Exception as e:
            logger.error(f"Error processing {model_dir.name}: {e}")
            results.append({
                "model": model_dir.name,
                "success": False
            })
        
        # Add delay between pushes (except for the last one)
        if i < len(model_dirs) - 1 and args.delay > 0:
            logger.info(f"Waiting {args.delay} seconds before next push...")
            time.sleep(args.delay)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Push Summary:")
    logger.info(f"{'='*60}")
    
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    logger.info(f"Total: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    
    if failed > 0:
        logger.info("\nFailed models:")
        for result in results:
            if not result["success"]:
                logger.info(f"  ✗ {result['model']}")

if __name__ == "__main__":
    main()