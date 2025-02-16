import sys
from logging import Logger
from pathlib import Path
from typing import List

from datasets import DatasetDict
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


def load_tokenizer(model_name: str, cache_dir: str):
    """
    Load the tokenizer for the specified model.

    Args:
        model_name (str): The name or path of the pre-trained model.

    Returns:
        tokenizer: The loaded tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    return tokenizer


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
    logger: Logger,
):
    padding = None
    truncation = None
    tokenize_function = None
    if model_type == ModelTypesEnum.ENCODER_CLASSIFICATION.value:
        padding = "max_length"
        truncation = True

        def tokenize_function(examples: List[str]):
            return tokenizer(
                examples[text_column], padding=padding, truncation=truncation
            )

    if model_type == ModelTypesEnum.DECODER_CLASSIFICATION_LORA.value:

        def tokenize_function(examples: List[str]):
            padding = "longest"
            truncation = False

            def _prompt_template(self, essay_example):
                instructions_text = None
                if self.reference_concept == 0:
                    instructions_text = f"<|system|>\n{CONCEPT1_SYSTEM}<|end|>\n"
                elif self.reference_concept == 1:
                    instructions_text = f"<|system|>\n{CONCEPT2_SYSTEM}<|end|>\n"
                elif self.reference_concept == 2:
                    instructions_text = f"<|system|>\n{CONCEPT3_SYSTEM}<|end|>\n"
                elif self.reference_concept == 3:
                    instructions_text = f"<|system|>\n{CONCEPT4_SYSTEM}<|end|>\n"
                elif self.reference_concept == 4:
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
            )

    if tokenize_function is None:
        raise ValueError(
            "tokenize_function should be a function. However, it is being set to None."
        )
    logger.info(
        f"Tokenizer function parameters- Padding:{padding}; Truncation: {truncation}"
    )
    return tokenize_function


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
        model_type, tokenizer, text_column, logger
    )
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset
