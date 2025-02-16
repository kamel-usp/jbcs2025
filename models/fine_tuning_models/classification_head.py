import hydra
from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification

@hydra.main(config_path="configs", config_name="config.yaml", version_base="1.1")
def load_model(cfg: DictConfig):
    model_cfg = cfg.experiments.model
    id2label = {
        0: 0, 
        1: 40,
        2: 80,
        3: 120,
        4: 160,
        5: 200
    }
    label2id = {
        0: 0, 
        40: 1,
        80: 2,
        120: 3,
        160: 4,
        200: 5
    }
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg.name,
        num_labels=model_cfg.num_labels,
        id2label=id2label,
        label2id=label2id,
        cache_dir=cfg.cache_dir
    )
    return model

if __name__ == "__main__":
    model = load_model()
    # You can now use the `model` instance as needed
