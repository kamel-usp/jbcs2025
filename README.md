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
4. **Install flash-attn package**

Due to some order installation issue, `flash-attn` has to be installed separatedly. Then run **inside the environment**
> uv pip install flash-attn --no-build-isolation


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