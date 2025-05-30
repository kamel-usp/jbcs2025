from dataclasses import dataclass
from models.fine_tuning_models.model_types_enum import LossType, ModelArchitecture


@dataclass
class ModelConfig:
    architecture: ModelArchitecture
    loss_type: LossType
    num_labels: int
    
    @property
    def is_ordinal(self) -> bool:
        return self.loss_type in [LossType.CORAL, LossType.CORN]
    
    @property
    def is_lora(self) -> bool:
        return self.architecture != ModelArchitecture.ENCODER
    
    @classmethod
    def from_model_type(cls, model_type: str, num_classes: int = 6) -> "ModelConfig":
        """Parse model type string to extract architecture and loss type."""
        model_type_upper = model_type.upper()
        
        # Determine architecture
        if "ENCODER" in model_type_upper:
            architecture = ModelArchitecture.ENCODER
        elif "PHI35" in model_type_upper:
            architecture = ModelArchitecture.PHI35
        elif "PHI4" in model_type_upper:
            architecture = ModelArchitecture.PHI4
        elif "LLAMA31" in model_type_upper:
            architecture = ModelArchitecture.LLAMA31
        else:
            raise ValueError(f"Unknown architecture in model type: {model_type}")
        
        # Determine loss type
        if "CORAL" in model_type_upper:
            loss_type = LossType.CORAL
        elif "CORN" in model_type_upper:
            loss_type = LossType.CORN
        else:
            loss_type = LossType.CROSS_ENTROPY
        
        # Calculate number of labels
        num_labels = num_classes - 1 if loss_type != LossType.CROSS_ENTROPY else num_classes
        
        return cls(
            architecture=architecture,
            loss_type=loss_type,
            num_labels=num_labels
        )