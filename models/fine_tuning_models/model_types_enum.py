from enum import Enum

class ModelTypesEnum(Enum):
    ENCODER_CLASSIFICATION = "encoder_classification"
    PHI35_CLASSIFICATION_LORA = "phi35_classification_lora"
    PHI4_CLASSIFICATION_LORA = "phi4_classification_lora"
    LLAMA31_CLASSIFICATION_LORA = "llama31_classification_lora"