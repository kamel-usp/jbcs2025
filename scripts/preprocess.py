import sys
from logging import Logger
from pathlib import Path
from typing import List

from datasets import DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

# Append the parent directory to sys.path using pathlib
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from models.fine_tuning_models.model_config import ModelConfig # NOQA: E402
from models.fine_tuning_models.model_types_enum import ModelArchitecture, ModelTypesEnum  # NOQA: E402
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
    model_type: str,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    text_column: str,
    grade_index: int,
    logger: Logger,
):
    padding = None
    truncation = None
    tokenize_function_def = None
    model_config = ModelConfig.from_model_type(model_type)
    if model_config.architecture == ModelArchitecture.ENCODER:
        padding = "max_length"
        truncation = True
        padding_side = "right"
        max_length = 512

        def tokenize_function(examples: List[str]):
            return tokenizer(
                examples[text_column],
                padding=padding,
                truncation=truncation,
                padding_side=padding_side,
                max_length=max_length,
            )

        tokenize_function_def = tokenize_function

    if model_config.architecture in [
        ModelArchitecture.PHI35,
        ModelArchitecture.PHI4,
    ]:
        padding = "longest"
        truncation = False
        padding_side = "left"

        def tokenize_function(examples: List[str]):
            def _prompt_template(essay_example):
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

                user_role = f"<|user|>Qual é a nota da redação a seguir?\n\n{essay_example}<|end|>\n"
                assistant_role = "<|assistant|>"
                instructions_text += user_role
                instructions_text += assistant_role
                return instructions_text

            def _prepare_instruction_template(examples: List[str]):
                result = []
                for example in examples:
                    result.append(_prompt_template(example))
                return result

            return tokenizer(
                _prepare_instruction_template(examples[text_column]),
                return_tensors="pt",
                padding=padding,
                truncation=truncation,
                padding_side=padding_side,
            )

        tokenize_function_def = tokenize_function
    if model_config.architecture in [ModelArchitecture.LLAMA31]:
        padding = "longest"
        truncation = False
        padding_side = "left"

        def tokenize_function(examples: List[str]):
            def _prompt_template(essay_example):
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
                user_role = f"{user_role}\n\nQual é a nota da redação a seguir?\n\n{essay_example}{end_of_instruction}\n"
                assistant_role = "<|start_header_id|>assistant<|end_header_id|>"
                instructions_text += user_role
                instructions_text += assistant_role
                return instructions_text

            def _prepare_instruction_template(examples: List[str]):
                result = []
                for example in examples:
                    result.append(_prompt_template(example))
                return result

            return tokenizer(
                _prepare_instruction_template(examples[text_column]),
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
    text_column: str,
    grade_index: int,
    model_type: str,
    logger: Logger,
):
    """
    Tokenize the dataset and process the 'grades' column.

    Args:
        dataset (DatasetDict): The dataset to tokenize.
        tokenizer: The tokenizer to use.
        text_column (str): The name of the text column in the dataset.
        grade_index (int): The index of the grade to extract.

    Returns:
        DatasetDict: The tokenized dataset.
    """
    # Process the 'grades' column
    dataset = dataset.map(lambda x: process_grades(x, grade_index))
    tokenize_function = get_tokenize_function(
        model_type, tokenizer, text_column, grade_index, logger
    )
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset
