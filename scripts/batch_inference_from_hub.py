import argparse
import logging
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("batch_inference_from_hub.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def download_config_from_hub(model_id: str, cache_dir: Optional[str] = None) -> Optional[Dict]:
    """
    Download run_experiment.log from Hugging Face Hub and parse the configuration.
    
    Args:
        model_id: HF Hub model ID (e.g., 'kamel-usp/jbcs2025_model_name')
        cache_dir: Optional cache directory for downloads
        
    Returns:
        Parsed configuration dictionary or None if failed
    """
    try:
        # Download the log file
        log_path = hf_hub_download(
            repo_id=model_id,
            filename="run_experiment.log",
            cache_dir=cache_dir
        )
        
        # Parse the configuration
        with open(log_path, 'r') as f:
            lines = f.readlines()
        
        # Find the YAML configuration section
        yaml_lines = []
        in_yaml = False
        
        for line in lines:
            # Start capturing after the first log line
            if not in_yaml and line.strip() and not line.startswith('['):
                in_yaml = True
            
            # Stop when we hit another log line
            if in_yaml and line.startswith('['):
                break
                
            if in_yaml:
                yaml_lines.append(line)
        
        # Parse YAML
        yaml_content = ''.join(yaml_lines)
        config = yaml.safe_load(yaml_content)
        
        # Update model name to point to HF Hub model
        if 'experiments' in config and 'model' in config['experiments']:
            config['experiments']['model']['name'] = model_id
            # Update the best_model_dir to use the hub model
            config['experiments']['model']['best_model_dir'] = model_id
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to download/parse config from {model_id}: {e}")
        return None


def save_temp_config(config: Dict, model_id: str) -> Path:
    """
    Save configuration to a temporary YAML file.
    
    Args:
        config: Configuration dictionary
        model_id: Model ID for naming
        
    Returns:
        Path to the temporary config file
    """
    # Create a safe filename from model_id
    safe_name = re.sub(r'[^\w\-_]', '_', model_id)
    
    # Create temp directory in the experiments folder
    project_root = Path(__file__).resolve().parent.parent
    temp_dir = project_root / "configs" / "experiments" / "temp_inference"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config - extract just the experiments part
    config_path = temp_dir / f"{safe_name}.yaml"
    
    # Save only the experiments section to match the expected format
    experiments_config = config.get('experiments', {})
    with open(config_path, 'w') as f:
        yaml.dump(experiments_config, f, default_flow_style=False, sort_keys=False)
    
    return config_path


def run_inference_experiment(config_path: Path, output_dir: Optional[str] = None) -> bool:
    """
    Run inference experiment with the given configuration.
    
    Args:
        config_path: Path to the configuration file
        output_dir: Optional output directory override
        
    Returns:
        True if successful, False otherwise
    """
    project_root = Path(__file__).resolve().parent.parent
    
    # Get relative path from experiments directory
    experiments_dir = project_root / "configs" / "experiments"
    relative_config_path = config_path.relative_to(experiments_dir)
    
    # Format as experiments=folder/config_name (without .yaml)
    config_ref = str(relative_config_path).replace('\\', '/').replace('.yaml', '')
    
    cmd = [
        "python",
        "scripts/run_inference_experiment.py",
        f"experiments={config_ref}"
    ]
    
    # Add output directory override if specified
    if output_dir:
        cmd.append(f"hydra.run.dir={output_dir}")
    else:
        # Use default pattern
        cmd.append("hydra.run.dir=inference_output/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}")
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(project_root)
        )
        
        # Stream output
        for line in process.stdout:
            print(line, end="")
        
        return_code = process.wait()
        
        if return_code == 0:
            logger.info("Inference completed successfully")
            return True
        else:
            logger.error(f"Inference failed with return code {return_code}")
            return False
            
    except Exception as e:
        logger.error(f"Error running inference: {e}")
        return False


def process_model_list(models_file: Path) -> List[str]:
    """
    Read model IDs from a file.
    
    Args:
        models_file: Path to file containing model IDs (one per line)
        
    Returns:
        List of model IDs
    """
    with open(models_file, 'r') as f:
        models = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return models


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run batch inference experiments from Hugging Face Hub models"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs='+',
        help="List of HF Hub model IDs to run inference on",
    )
    parser.add_argument(
        "--models-file",
        type=str,
        help="Path to file containing model IDs (one per line)",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="inference_output",
        help="Base output directory (default: inference_output)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=2,
        help="Delay in seconds between experiments (default: 2)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Cache directory for HF Hub downloads",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up temporary config files after completion",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without actually running",
    )
    
    args = parser.parse_args()
    
    # Get list of models
    models = []
    if args.models:
        models.extend(args.models)
    if args.models_file:
        models.extend(process_model_list(Path(args.models_file)))
    
    if not models:
        logger.error("No models specified. Use --models or --models-file")
        sys.exit(1)
    
    logger.info(f"Processing {len(models)} models")
    
    if args.dry_run:
        logger.info("DRY RUN - Models that would be processed:")
        for model in models:
            logger.info(f"  - {model}")
        return
    
    # Process each model
    results = []
    temp_configs = []
    
    for i, model_id in enumerate(tqdm(models, desc="Running inference")):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {i+1}/{len(models)}: {model_id}")
        logger.info(f"{'='*60}\n")
        
        try:
            # Download and parse configuration
            config = download_config_from_hub(model_id, args.cache_dir)
            if not config:
                logger.error(f"Failed to get configuration for {model_id}")
                results.append({
                    "model": model_id,
                    "success": False,
                    "error": "Failed to download/parse configuration"
                })
                continue
            
            # Save temporary config
            config_path = save_temp_config(config, model_id)
            temp_configs.append(config_path)
            
            # Generate output directory
            timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
            output_dir = f"{args.output_base}/{timestamp}"
            
            # Run inference
            success = run_inference_experiment(config_path, output_dir)
            
            results.append({
                "model": model_id,
                "success": success,
                "output_dir": output_dir if success else None
            })
            
        except Exception as e:
            logger.error(f"Error processing {model_id}: {e}")
            results.append({
                "model": model_id,
                "success": False,
                "error": str(e)
            })
        
        # Add delay between experiments
        if i < len(models) - 1 and args.delay > 0:
            logger.info(f"Waiting {args.delay} seconds before next experiment...")
            time.sleep(args.delay)
    
    # Clean up temporary configs if requested
    if args.cleanup:
        logger.info("Cleaning up temporary configuration files...")
        for config_path in temp_configs:
            try:
                config_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete {config_path}: {e}")
        
        # Remove temp directory if empty
        project_root = Path(__file__).resolve().parent.parent
        temp_dir = project_root / "configs" / "experiments" / "temp_inference"
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Inference Summary:")
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
                error = result.get("error", "Unknown error")
                logger.info(f"  ✗ {result['model']} - {error}")
    
    if successful > 0:
        logger.info("\nSuccessful models:")
        for result in results:
            if result["success"]:
                logger.info(f"  ✓ {result['model']} -> {result['output_dir']}")


if __name__ == "__main__":
    main()
