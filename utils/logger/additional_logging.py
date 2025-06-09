
from logging import Logger
import torch
import pynvml


def log_gpu_info(logger: Logger) -> None:
    """
    Print the name and power-limit (≈ TDP) of every visible NVIDIA GPU.
    Falls back to `nvidia-smi` when pynvml is not available.
    """
    if not torch.cuda.is_available():
        logger.info("CUDA not available – running on CPU.")
        return

    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            for idx in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(idx)
                name = pynvml.nvmlDeviceGetName(h)
                # Power-limit is returned in mW
                tdp = pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000
                logger.info(f"GPU {idx}: {name} | TDP ≈ {tdp:.0f} W")
        except Exception as e:
            logger.warning(f"pynvml failed to fetch GPU info: {e}")
    else:
        import subprocess

        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,power.limit",
                    "--format=csv,noheader",
                ],
                encoding="utf-8",
            )
            for line in out.strip().splitlines():
                idx, name, tdp = [s.strip() for s in line.split(",")]
                logger.info(f"GPU {idx}: {name} | TDP ≈ {tdp} W")
        except Exception as e:
            logger.warning(f"Unable to read GPU info via nvidia-smi: {e}")