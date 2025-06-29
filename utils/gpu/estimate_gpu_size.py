import json
from typing import Optional, Set, Dict

from huggingface_hub import (
    get_safetensors_metadata,
    repo_info,
    hf_hub_download,
)
from huggingface_hub.utils import (
    NotASafetensorsRepoError,
    EntryNotFoundError,
)

# ──────────────────────────────────────────────────────────────
#  Tweak these two maps if you want a different heuristic
# ──────────────────────────────────────────────────────────────
# Bytes *per parameter* we actually hold on the GPU once the
# model is loaded in the chosen dtype (weights only).
DTYPE_BYTES: Dict[str, float] = {"F32": 4, "F16": 2, "BF16": 2, "I8": 1, "I4": 0.5}

# How many bytes per parameter end up in GPU RAM *including*
# master weights, gradients and optimiser state.
# Hugging Face doc:  6 B / param in mixed-precision inference,
#                    18 B / param in mixed-precision training.
INFERENCE_BPP = {"F32": 4, "F16": 6, "BF16": 6, "I8": 2, "I4": 2}
TRAINING_BPP  = {"F32": 20, "F16": 18, "BF16": 18, "I8": 12, "I4": 12}

_BYTES_IN_GIB = 1 << 30  # 2**30

# ──────────────────────────────────────────────────────────────
#  Private helpers
# ──────────────────────────────────────────────────────────────
def _param_count_from_repo(repo_id: str, revision: Optional[str] = None) -> int:
    """
    Return the total parameter count of *all* .safetensors files in a repo
    (or raise NotASafetensorsRepoError if the repo has only *.bin weights).
    """
    meta = get_safetensors_metadata(repo_id, revision=revision)
    return sum(meta.parameter_count.values())

def _adapter_bytes(repo_id: str, revision: Optional[str] = None) -> int:
    """
    Sum the on-disk sizes of any *.safetensors files sitting in a PEFT/LoRA
    repo that lacks `model.safetensors`.
    """
    info = repo_info(repo_id, revision=revision, files_metadata=True)
    return sum(f.size for f in info.siblings if f.rfilename.endswith(".safetensors"))

def _add_base_if_peft(repo_id: str,
                      revision: Optional[str],
                      seen: Set[str]) -> int:
    """
    If *repo_id* is an adapter, grab its `base_model_name_or_path` and recurse.
    Avoid infinite loops with *seen*.
    """
    if repo_id in seen:
        return 0
    seen.add(repo_id)

    # Quick probe: does the repo even list adapter_config.json?
    info = repo_info(repo_id, revision=revision, files_metadata=False)
    if not any(s.rfilename == "adapter_config.json" for s in info.siblings):
        return 0

    try:
        cfg_path = hf_hub_download(repo_id, "adapter_config.json",
                                   revision=revision, resume_download=False)
        with open(cfg_path) as f:
            base = json.load(f).get("base_model_name_or_path")
    except (FileNotFoundError, EntryNotFoundError):
        return 0

    return _estimate_params(base, revision, seen) if base else 0

def _estimate_params(repo_id: str,
                     revision: Optional[str] = None,
                     seen: Optional[Set[str]] = None) -> int:
    """
    Return *total* parameter count for repo + its base (if LoRA/IA³ adapter).
    """
    seen = seen or set()
    try:
        params = _param_count_from_repo(repo_id, revision)
    except NotASafetensorsRepoError:
        # Fallback for LoRA-only repos without model.safetensors
        # We can only convert bytes → params if we assume fp16 storage.
        bytes_on_disk = _adapter_bytes(repo_id, revision)
        params = int(bytes_on_disk / DTYPE_BYTES["F16"])
    params += _add_base_if_peft(repo_id, revision, seen)
    return params

# ──────────────────────────────────────────────────────────────
#  Public function: returns (inference GiB, training GiB)
# ──────────────────────────────────────────────────────────────
def estimate_vram_gib(
    repo_id: str,
    revision: Optional[str] = None,
    load_dtype: str = "F16",
    extra_cuda_buffer_gib: float = 0.75,
) -> tuple[float, float]:
    """
    Estimate the GPU RAM (GiB) needed for *inference* **and** *full training*
    of a Hugging Face model or adapter.

    Parameters
    ----------
    repo_id : str
        Model or adapter repo (e.g. "meta-llama/Llama-3.1-8B").
    revision : str | None
        Optional git hash or branch name.
    load_dtype : {"F16","BF16","F32","I8","I4"}
        Precision you plan to *load* the weights in.
    extra_cuda_buffer_gib : float
        Constant overhead reserved by the CUDA context & allocator.

    Returns
    -------
    inference_gib : float
        Estimated peak VRAM to *run* the model (batch-size 1, no kv-cache).
    training_gib : float
        Estimated peak VRAM to *fine-tune* the model with AdamW mixed-precision.
    """
    params = _estimate_params(repo_id, revision)
    inf_bytes = params * INFERENCE_BPP[load_dtype]
    trn_bytes = params * TRAINING_BPP[load_dtype]

    inference_gib = (inf_bytes / _BYTES_IN_GIB) + extra_cuda_buffer_gib
    training_gib  = (trn_bytes / _BYTES_IN_GIB) + extra_cuda_buffer_gib
    return round(inference_gib, 2), round(training_gib, 2)
