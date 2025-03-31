import sys
from logging import Logger
from pathlib import Path

from datasets import DatasetDict
from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

# Append the parent directory to sys.path using pathlib
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from models.fine_tuning_models.model_types_enum import ModelTypesEnum  # NOQA: E402
from scripts.constants.prompts.phi_models import (  # noqa: E402
    CONCEPT1_SYSTEM,
    CONCEPT2_SYSTEM,
    CONCEPT3_SYSTEM,
    CONCEPT4_SYSTEM,
    CONCEPT5_SYSTEM,
)

LABEL2ID = {0: 0, 40: 1, 80: 2, 120: 3, 160: 4, 200: 5}


def load_tokenizer(model_type: str, model_name: str, cache_dir: str):
    """
    Load the tokenizer for the specified model.

    Args:
        model_name (str): The name or path of the pre-trained model.

    Returns:
        tokenizer: The loaded tokenizer.
    """
    if model_type == ModelTypesEnum.PHI35_CLASSIFICATION_LORA.value:
        return AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            pad_token="<|dummy_id_0|>",
            trust_remote_code=True,
        )
    if model_type == ModelTypesEnum.LLAMA31_CLASSIFICATION_LORA.value:
        return AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            pad_token="<|finetune_right_pad_id|>",
        )
    return AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)


def process_grades(example, grade_index: int):
    """
    Extract the grade at the specified index from the 'grades' list.

    Args:
        example (dict): A single example from the dataset.
        grade_index (int): The index of the grade to extract.

    Returns:
        dict: The example with the 'label' field added.
    """

    if example.get("grades") and len(example["grades"]) > grade_index:
        example["label"] = LABEL2ID[example["grades"][grade_index]]
        return example
    raise ValueError("Please Provide a Valid Label ID")


def get_tokenize_function(
    experiment_config: DictConfig,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    grade_index: int,
    logger: Logger,
):
    model_type = experiment_config.experiments.model.type
    use_essay_prompt = experiment_config.experiments.model.use_essay_prompt
    padding = None
    truncation = None
    tokenize_function_def = None
    if model_type == ModelTypesEnum.ENCODER_CLASSIFICATION.value:
        padding = "max_length"
        truncation = True
        padding_side = "right"
        max_length = 512

        def tokenize_function(examples: dict):
            return tokenizer(
                examples["essay_text"],
                padding=padding,
                truncation=truncation,
                padding_side=padding_side,
                max_length=max_length,
            )

        tokenize_function_def = tokenize_function

    if model_type in [
        ModelTypesEnum.PHI35_CLASSIFICATION_LORA.value,
        ModelTypesEnum.PHI4_CLASSIFICATION_LORA.value,
    ]:
        padding = "longest"
        truncation = False
        padding_side = "left"

        def tokenize_function(examples: dict):
            def _prompt_template(essay_text, supporting_text, prompt):
                instructions_text = None
                if grade_index == 0:
                    instructions_text = f"<|system|>\n{CONCEPT1_SYSTEM}<|end|>\n"
                elif grade_index == 1:
                    instructions_text = f"<|system|>\n{CONCEPT2_SYSTEM}<|end|>\n"
                elif grade_index == 2:
                    instructions_text = f"<|system|>\n{CONCEPT3_SYSTEM}<|end|>\n"
                elif grade_index == 3:
                    instructions_text = f"<|system|>\n{CONCEPT4_SYSTEM}<|end|>\n"
                elif grade_index == 4:
                    instructions_text = f"<|system|>\n{CONCEPT5_SYSTEM}<|end|>\n"
                if use_essay_prompt is False:
                    user_role = f"<|user|>Qual é a nota da redação a seguir?\n\n{essay_text}<|end|>\n"
                elif use_essay_prompt is True:
                    user_role = f"<|user|>Considere os textos de apoio:\n\n{supporting_text}.\n\n"
                    user_role += f"Agora, o tema da redação é descrito a seguir:\n\n{prompt}\n\n"
                    user_role += f"Com base no tema e nos textos, qual é a nota da redação a seguir?\n\n{essay_text}<|end|>\n"
                assistant_role = "<|assistant|>"
                instructions_text += user_role
                instructions_text += assistant_role
                return instructions_text

            def _prepare_instruction_template(examples):
                result = []
                for essay_text, supporting_text, prompt in zip(
                    examples["essay_text"],
                    examples["supporting_text"],
                    examples["prompt"],
                ):
                    result.append(_prompt_template(essay_text, supporting_text, prompt))
                return result

            return tokenizer(
                _prepare_instruction_template(examples),
                return_tensors="pt",
                padding=padding,
                truncation=truncation,
                padding_side=padding_side,
            )

        tokenize_function_def = tokenize_function
    if model_type in [ModelTypesEnum.LLAMA31_CLASSIFICATION_LORA.value]:
        padding = "longest"
        truncation = False
        padding_side = "left"

        def tokenize_function(examples: dict):
            def _prompt_template(essay_text, supporting_text, prompt):
                instructions_text = None
                system_prefix = (
                    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
                )
                user_role = "<|start_header_id|>user<|end_header_id|>"
                end_of_instruction = "<|eot_id|>"
                if grade_index == 0:
                    instructions_text = (
                        f"{system_prefix}\n\n{CONCEPT1_SYSTEM}{end_of_instruction}\n"
                    )
                elif grade_index == 1:
                    instructions_text = (
                        f"{system_prefix}\n\n{CONCEPT2_SYSTEM}{end_of_instruction}\n"
                    )
                elif grade_index == 2:
                    instructions_text = (
                        f"{system_prefix}\n\n{CONCEPT3_SYSTEM}{end_of_instruction}\n"
                    )
                elif grade_index == 3:
                    instructions_text = (
                        f"{system_prefix}\n\n{CONCEPT4_SYSTEM}{end_of_instruction}\n"
                    )
                elif grade_index == 4:
                    instructions_text = (
                        f"{system_prefix}\n\n{CONCEPT5_SYSTEM}{end_of_instruction}\n"
                    )
                if use_essay_prompt is False:
                    user_role = f"{user_role}\n\nQual é a nota da redação a seguir?\n\n{essay_text}{end_of_instruction}\n"
                elif use_essay_prompt is True:
                    user_role = f"{user_role}\n\nConsidere os textos de apoio:\n\n{supporting_text}.\n\n"
                    user_role += f"Agora, o tema da redação é descrito a seguir:\n\n{prompt}\n\n"
                    user_role += f"Com base no tema e nos textos, qual é a nota da redação a seguir?\n\n{essay_text}{end_of_instruction}\n"
                assistant_role = "<|start_header_id|>assistant<|end_header_id|>"
                instructions_text += user_role
                instructions_text += assistant_role
                return instructions_text

            def _prepare_instruction_template(examples):
                result = []
                for essay_text, supporting_text, prompt in zip(
                    examples["essay_text"],
                    examples["supporting_text"],
                    examples["prompt"],
                ):
                    result.append(_prompt_template(essay_text, supporting_text, prompt))
                return result

            return tokenizer(
                _prepare_instruction_template(examples),
                return_tensors="pt",
                padding=padding,
                truncation=truncation,
                padding_side=padding_side,
            )

        tokenize_function_def = tokenize_function
    if tokenize_function_def is None:
        raise ValueError(
            "tokenize_function_def should be a function. However, it is being set to None."
        )
    logger.info(
        f"Tokenizer function parameters- Padding:{padding}; Truncation: {truncation}"
    )
    return tokenize_function_def


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    experiment_config: DictConfig,
    logger: Logger,
):
    """
    Tokenize the dataset and process the 'grades' column.

    Args:
        dataset (DatasetDict): The dataset to tokenize.
        tokenizer: The tokenizer to use.
        experiment_config: hydra configs

    Returns:
        DatasetDict: The tokenized dataset.
    """
    grade_index = experiment_config.experiments.dataset.grade_index
    # Process the 'grades' column
    dataset = dataset.map(lambda x: process_grades(x, grade_index))
    tokenize_function = get_tokenize_function(
        experiment_config, tokenizer, grade_index, logger
    )
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset
