import hydra
from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification

@hydra.main(config_path="configs", config_name="config.yaml")
def load_model(cfg: DictConfig):
    model_cfg = cfg.model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg.name,
        num_labels=model_cfg.num_labels
    )
    return model

if __name__ == "__main__":
    model = load_model()
    # You can now use the `model` instance as needed
