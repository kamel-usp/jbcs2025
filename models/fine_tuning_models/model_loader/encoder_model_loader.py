from transformers import AutoModelForSequenceClassification, PreTrainedModel

from models.fine_tuning_models.model_loader.base_loader import BaseModelLoader



class EncoderModelLoader(BaseModelLoader):
    """Loader for encoder models (e.g., BERT, DeBERTa)."""

    def load(self) -> PreTrainedModel:
        return AutoModelForSequenceClassification.from_pretrained(
            self.model_cfg.name,
            **self._get_base_model_kwargs(),
        )
