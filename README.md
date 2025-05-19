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

This project uses environment variables for managing API keys (such as `open AI or Maritaca AI`) and other sensitive information.

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

---

Now you're ready to contribute to `jbcs2025`! 🎉
