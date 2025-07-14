from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    Phi3ForSequenceClassification,
    PreTrainedModel,
)

from models.fine_tuning_models.model_loader.base_loader import BaseModelLoader
from models.fine_tuning_models.model_types_enum import ModelArchitecture

class DecoderModelLoraLoader(BaseModelLoader):
    """Loader for LoRA-based models."""
    
    def load(self) -> PreTrainedModel:
        # Check if the model name contains "kamel-usp" to identify fine-tuned models
        if "kamel-usp" in self.model_cfg.name.lower():
            return self._load_pretrained_peft_model()
        else:
            return self._initialize_new_peft_model()
    
    def _get_lora_config(self) -> LoraConfig:
        """Create LoRA configuration."""
        return LoraConfig(
            r=self.model_cfg.lora_r,
            lora_alpha=self.model_cfg.lora_alpha,
            lora_dropout=self.model_cfg.lora_dropout,
            task_type=TaskType.SEQ_CLS,
            target_modules=self.model_cfg.lora_target_modules,
        )
    
    def _get_base_model_class(self):
        """Get the appropriate base model class."""
        if self.model_config.architecture == ModelArchitecture.PHI35:
            return Phi3ForSequenceClassification
        else:
            return AutoModelForSequenceClassification
    
    def _initialize_new_peft_model(self) -> PeftModel:
        """Initialize a new PEFT model for training."""
        lora_config = self._get_lora_config()
        base_model_class = self._get_base_model_class()
        
        base_model = base_model_class.from_pretrained(
            self.model_cfg.name,
            **self._get_decoder_model_kwargs()
        )
        
        model = get_peft_model(base_model, lora_config)
        
        if (self.model_config.architecture != ModelArchitecture.PHI35 and 
            self.cfg.experiments.training_params.gradient_checkpointing):
            model.enable_input_require_grads()
        
        self.logger.info(f"Initialized new PEFT model for {self.model_config.loss_type.value} loss")
        
        # Set padding token
        model.config.pad_token_id = model.config.eos_token_id
        # Weird bug for Llama-3.1 intruct models
        if self.model_cfg.name in ["meta-llama/Llama-3.1-8B-Instruct", "/tmp/clean_cache/gervasio-8b-portuguese-ptpt-decoder"]:
            model.config.pad_token_id = model.config.eos_token_id[0]
        self.logger.info(model.print_trainable_parameters())
        
        return model
    
    def _load_pretrained_peft_model(self) -> PeftModel:
        """Load a pretrained PEFT model from hub."""
        peft_config = PeftConfig.from_pretrained(self.model_cfg.name)
        base_model_class = self._get_base_model_class()
        base_model_name = peft_config.base_model_name_or_path
        self.logger.info(f"Loading PEFT model configuration from {self.model_cfg.name}")
        self.logger.info(f"Base model name: {base_model_name}")
        base_model = base_model_class.from_pretrained(
            base_model_name,
            **self._get_decoder_model_kwargs()
        )
        
        # Load the PEFT model directly from the hub repository
        model = PeftModel.from_pretrained(base_model, self.model_cfg.name)
        self.logger.info(f"Loaded fine-tuned PEFT model from {self.model_cfg.name}")
        
        # Set padding token
        model.config.pad_token_id = model.config.eos_token_id
        self.logger.info(model.print_trainable_parameters())
        
        return model