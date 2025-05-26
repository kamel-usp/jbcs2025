from logging import Logger

from omegaconf import DictConfig
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    Phi3ForSequenceClassification,
)

ID2LABEL = {0: 0, 1: 40, 2: 80, 3: 120, 4: 160, 5: 200}

LABEL2ID = {0: 0, 40: 1, 80: 2, 120: 3, 160: 4, 200: 5}


def load_model_with_classification_head(cfg: DictConfig):
    model_cfg = cfg.experiments.model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg.name,
        num_labels=model_cfg.num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        cache_dir=cfg.cache_dir,
    )
    return model


def load_phi3_classification_lora(cfg: DictConfig, logger: Logger):
    model_cfg = cfg.experiments.model
    checkpoint_path = model_cfg.checkpoint_path

    if checkpoint_path:
        # Load fine-tuned PEFT model from checkpoint
        model = load_pretrained_peft_model(cfg, logger, checkpoint_path)
    else:
        # Initialize new PEFT model for training
        lora_config = LoraConfig(
            r=model_cfg.lora_r,
            lora_alpha=model_cfg.lora_alpha,
            lora_dropout=model_cfg.lora_dropout,
            task_type=TaskType.SEQ_CLS,
            target_modules=model_cfg.lora_target_modules,
        )
        base_model = Phi3ForSequenceClassification.from_pretrained(
            model_cfg.name,
            num_labels=model_cfg.num_labels,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            cache_dir=cfg.cache_dir,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )

        model = get_peft_model(base_model, lora_config)
        logger.info("Initialized new PEFT model for training")

    model.config.pad_token_id = model.config.eos_token_id
    logger.info(model.print_trainable_parameters())
    return model


def load_slm_decoder_classification_lora(cfg: DictConfig, logger: Logger):
    model_cfg = cfg.experiments.model
    checkpoint_path = model_cfg.checkpoint_path

    if checkpoint_path:
        # Load fine-tuned PEFT model from checkpoint
        model = load_pretrained_peft_model(cfg, logger, checkpoint_path)
    else:
        # Initialize new PEFT model for training
        lora_config = LoraConfig(
            r=model_cfg.lora_r,
            lora_alpha=model_cfg.lora_alpha,
            lora_dropout=model_cfg.lora_dropout,
            task_type=TaskType.SEQ_CLS,
            target_modules=model_cfg.lora_target_modules,
        )
        # Load base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_cfg.name,
            num_labels=model_cfg.num_labels,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            cache_dir=cfg.cache_dir,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        model = get_peft_model(base_model, lora_config)
        if (
            cfg.experiments.training_params.gradient_checkpointing
            and not checkpoint_path
        ):
            model.enable_input_require_grads()
        logger.info("Initialized new PEFT model for training")

    model.config.pad_token_id = model.config.eos_token_id
    logger.info(model.print_trainable_parameters())
    return model


def load_pretrained_peft_model(cfg: DictConfig, logger: Logger, checkpoint_path: str):
    """
    Load a pre-trained PEFT model from a checkpoint.

    Args:
        cfg: Configuration object
        logger: Logger instance
        checkpoint_path: Path to the PEFT checkpoint (e.g., "kamel-usp/jbcs2025_phi35-balanced-C1")

    Returns:
        Loaded PEFT model ready for inference
    """
    model_cfg = cfg.experiments.model

    # Determine if this is a Phi model based on the model name
    is_phi_model = "phi-3.5" in model_cfg.name.lower()

    # Load base model with appropriate class
    if is_phi_model:
        base_model = Phi3ForSequenceClassification.from_pretrained(
            model_cfg.name,
            num_labels=model_cfg.num_labels,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            cache_dir=cfg.cache_dir,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
    else:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_cfg.name,
            num_labels=model_cfg.num_labels,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            cache_dir=cfg.cache_dir,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )

    # Load PEFT adapter
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    logger.info(f"Loaded pre-trained PEFT model from {checkpoint_path}")
    return model
