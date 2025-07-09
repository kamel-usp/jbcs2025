# jbcs2025

## Project Setup

This project uses [`uv`](https://github.com/astral-sh/uv) as the package manager for managing dependencies efficiently.

### Prerequisites
Ensure you have `uv` installed. You can install it using:
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Or using `pip`:
```sh
pip install uv
```

### Setup Instructions

1. **Clone the repository:**
   ```sh
   git clone <repository-url>
   cd jbcs2025
   ```

2. **Synchronize dependencies:**
   ```sh
   uv sync
   ```

3. **Activate the virtual environment:**
   - macOS/Linux:
     ```sh
     source .venv/bin/activate
     ```
   - Windows (PowerShell):
     ```sh
     .venv\Scripts\activate
     ```
4. **Install flash-attn package (Only needed if you want to fine-tune/inference with Decoder LMs like Llama8B or Phi4/3.5)**

Due to some order installation issue, `flash-attn` has to be installed separatedly. Then run **inside the environment**
```sh
uv pip install flash-attn --no-build-isolation
```

### Managing API Keys and Secrets

This project uses environment variables for managing API keys (such as `OpenAI or Maritaca AI`) and other sensitive information.

  1. Create a .env file in the project root:
  ```sh
  touch .env  # On Linux/macOS
  # Or manually create the file on Windows
  ```
  2. Add your API keys to the .env file:
  ```sh
  #API Keys
  MARITACA_API_KEY=your_maritaca_api_key_here
  OPENAI_API_KEY=your_openai_api_key_here
  HUGGINGFACE_TOKEN=your_hf_token_here
  ```
  3. How it works:
  - The project uses python-dotenv to load environment variables from the `.env` file
  - API keys are referenced in YAML configuration files like this:
  ```sh
  model:
    api_key: ${env:MARITACA_API_KEY,""}  # Falls back to empty string if not found
  ```
  - For additional security, you can also set environment variables directly in your system.
  4. Security best practices:
  - Never commit .env files to Git (it's already in .gitignore)
  - Don't share API keys in code, chat, or documentation
  - Rotate API keys periodically
  
### Dependency Management
- To add a new dependency, use:
  ```sh
  uv add <package-name>
  ```

### Deactivating the Environment
To deactivate the virtual environment, run:
```sh
deactivate
```

## Usage

The project provides several scripts for different machine learning workflows:

### 1. Training and Fine-tuning Models

#### Single Experiment
Run a single experiment using the main configuration:
```sh
python scripts/run_experiment.py
```

You can override configuration parameters:
```sh
python scripts/run_experiment.py experiments.model.name=microsoft/phi-3.5-mini-instruct experiments.dataset.grade_index=0
```

#### Inference Experiments
Run inference experiments on pre-trained models:
```sh
python scripts/run_inference_experiment.py
```

#### Sequential Experiments
Run multiple experiments in sequence using a configuration file:
```sh
python scripts/run_sequential_experiments.py configs/sequential_experiments.yaml --mode train
```

For inference mode:
```sh
python scripts/run_sequential_experiments.py configs/sequential_experiments.yaml --mode inference
```

### 2. Model Management and Hugging Face Hub Integration

#### Push Single Model to Hub
Push a trained model to Hugging Face Hub:
```sh
python scripts/push_model_to_hub.py /path/to/model/directory
```

#### Batch Push Models
Push multiple models from a date folder to Hugging Face Hub:
```sh
python scripts/batch_push_models.py outputs/2025-07-06 --config configs/push_model_config.yaml
```

Options:
- `--delay`: Delay between pushes in seconds (default: 2)
- `--dry-run`: Show what would be pushed without actually pushing

#### Generate Hub Models List
Generate a list of model IDs for batch inference:
```sh
python scripts/generate_hub_models_list.py --output configs/hub_models_list.txt
```

Use `--dry-run` to preview the list without writing to file.

#### Batch Inference from Hub
Run inference on multiple models from Hugging Face Hub:
```sh
python scripts/batch_inference_from_hub.py --models model1 model2 model3
```

Or use a file containing model IDs:
```sh
python scripts/batch_inference_from_hub.py --models-file configs/hub_models_list.txt
```

Options:
- `--output-base`: Base output directory (default: inference_output)
- `--delay`: Delay between experiments in seconds (default: 10)

### 3. Configuration Management

Configuration files are located in the `configs/` directory:

- `config.yaml`: Main configuration file
- `sequential_experiments.yaml`: Configuration for running multiple experiments
- `push_model_config.yaml`: Configuration for model pushing to Hub
- `hub_models_list.txt`: List of Hub model IDs for batch processing

Example experiment configurations are organized in:
- `configs/experiments/api_models_*/`: API-based model configs (e.g. Sabia; ChatGPT or DeepSeek)
- `configs/experiments/(base|large|slm_decoder)_models/`: Fine-tuned model configs for different architectures

### 4. Output Structure

- `outputs/YYYY-MM-DD/HH-MM-SS`: Training outputs organized by date
- `inference_output/YYYY-MM-DD/HH-MM-SS/`: Inference results with timestamps

```

---

Now you're ready to contribute to `jbcs2025`! 🎉
