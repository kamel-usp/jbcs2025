from transformers import AutoTokenizer
from datasets import DatasetDict

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
        example["label"] = example["grades"][grade_index]
        return example
    raise ValueError("Please Provide a Valid Label ID")

def tokenize_dataset(dataset: DatasetDict, tokenizer, text_column: str, grade_index: int):
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
    
    # Tokenize the text column
    def tokenize_function(examples):
        return tokenizer(examples[text_column], padding="max_length", truncation=True)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset
