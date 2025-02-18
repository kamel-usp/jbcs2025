from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    Phi3ForSequenceClassification,
)
from logging import Logger

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
    lora_config = LoraConfig(
        r=model_cfg.lora_r,
        lora_alpha=model_cfg.lora_alpha,
        lora_dropout=model_cfg.lora_dropout,
        task_type=TaskType.SEQ_CLS,
        target_modules=model_cfg.lora_target_modules,
    )

    model = Phi3ForSequenceClassification.from_pretrained(
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

    model = get_peft_model(model, lora_config)
    model.config.pad_token_id = model.config.eos_token_id
    logger.info(model.print_trainable_parameters())
    return model

def load_phi4_classification_lora(cfg: DictConfig, logger: Logger):
    model_cfg = cfg.experiments.model
    lora_config = LoraConfig(
        r=model_cfg.lora_r,
        lora_alpha=model_cfg.lora_alpha,
        lora_dropout=model_cfg.lora_dropout,
        task_type=TaskType.SEQ_CLS,
        target_modules=model_cfg.lora_target_modules,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
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

    model = get_peft_model(model, lora_config)
    model.config.pad_token_id = model.config.eos_token_id
    if cfg.experiments.training_params.gradient_checkpointing:
        model.enable_input_require_grads()
    logger.info(model.print_trainable_parameters())
    return model
