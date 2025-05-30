from enum import Enum

class LossType(Enum):
    CROSS_ENTROPY = "ce"
    CORAL = "coral"
    CORN = "corn"


class ModelArchitecture(Enum):
    ENCODER = "encoder"
    PHI35 = "phi35"
    PHI4 = "phi4"
    LLAMA31 = "llama31"
    
class ModelTypesEnum(Enum):
    # Standard classification models
    ENCODER_CLASSIFICATION = "encoder_classification"
    PHI35_CLASSIFICATION_LORA = "phi35_classification_lora"
    PHI4_CLASSIFICATION_LORA = "phi4_classification_lora"
    LLAMA31_CLASSIFICATION_LORA = "llama31_classification_lora"
    
    # Ordinal regression models - CORAL
    ENCODER_ORDINAL_CORAL = "encoder_ordinal_coral"
    PHI35_ORDINAL_CORAL_LORA = "phi35_ordinal_coral_lora"
    PHI4_ORDINAL_CORAL_LORA = "phi4_ordinal_coral_lora"
    LLAMA31_ORDINAL_CORAL_LORA = "llama31_ordinal_coral_lora"
    
    # Ordinal regression models - CORN
    ENCODER_ORDINAL_CORN = "encoder_ordinal_corn"
    PHI35_ORDINAL_CORN_LORA = "phi35_ordinal_corn_lora"
    PHI4_ORDINAL_CORN_LORA = "phi4_ordinal_corn_lora"
    LLAMA31_ORDINAL_CORN_LORA = "llama31_ordinal_corn_lora"
    
    # API models
    CHATGPT_4O = "openai_chatgpt_4o"
    DEEPSEEK_R1 = "deepseek_r1"
    MARITACA_SABIA = "maritaca_sabia_3"
