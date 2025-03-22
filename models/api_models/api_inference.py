import asyncio
import json
import re
from logging import Logger
from typing import Dict, List

import openai
from datasets import (
    Dataset,
    load_dataset,
)
from omegaconf import DictConfig
from pydantic import ValidationError
from tqdm.auto import tqdm

from metrics.metrics import compute_metrics
from models.api_models.open_ai_api import Competencia
from models.fine_tuning_models.model_factory import ModelFactory
from models.fine_tuning_models.model_types_enum import ModelTypesEnum
from scripts.constants.prompts.sabia_model import (
    CONCEPT1_SYSTEM,
    CONCEPT2_SYSTEM,
    CONCEPT3_SYSTEM,
    CONCEPT4_SYSTEM,
    CONCEPT5_SYSTEM,
)

CONCURRENCY_LIMIT = 3
EXPONENTIAL_BACKOFF_DELAY = 120


def parse_deepseek_response(api_response: str) -> tuple[str, Competencia]:
    # Extract the text between <think> and </think> tags
    think_pattern = r"<think>(.*?)</think>"
    think_match = re.search(think_pattern, api_response, re.DOTALL)
    if not think_match:
        raise ValueError("Failed to find <think> tags in the response.")
    think_text = think_match.group(1).strip()

    # Extract the JSON block marked by triple backticks with "json"
    json_pattern = r"```json\s*(\{.*?\})\s*```"
    json_match = re.search(json_pattern, api_response, re.DOTALL)
    if not json_match:
        raise ValueError("Failed to find JSON block in the response.")
    json_str = json_match.group(1).strip()

    try:
        evaluation = Competencia.model_validate_json(json_str)
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError("JSON parsing or validation failed") from e

    return think_text, evaluation


async def get_completion_with_retry(
    client: openai.AsyncOpenAI,
    model_name: str,
    messages: list[dict],
    experiment_config,
    max_retries: int = 5,
) -> list:
    retries = 0
    while True:
        try:
            if experiment_config.experiments.model.type in [
                ModelTypesEnum.CHATGPT_4O.value,
                ModelTypesEnum.MARITACA_SABIA.value,
            ]:
                completion = await client.beta.chat.completions.parse(
                    model=model_name,
                    messages=messages,
                    response_format=Competencia,
                    max_tokens=experiment_config.experiments.model.max_tokens,
                    temperature=experiment_config.experiments.model.temperature,
                    seed=experiment_config.experiments.model.seed,
                )
                return completion.choices[0].message.parsed
            if experiment_config.experiments.model.type in [
                ModelTypesEnum.DEEPSEEK_R1.value
            ]:
                completion = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=experiment_config.experiments.model.max_tokens,
                    top_p=experiment_config.experiments.model.top_p,
                    stop=list(experiment_config.experiments.model.stop),
                    temperature=experiment_config.experiments.model.temperature,
                )
                content = completion.choices[0].message.content
                think_text, evaluation = parse_deepseek_response(content)
                return think_text, evaluation
        except (openai.RateLimitError, ValueError) as e:
            if retries >= max_retries:
                raise e
            wait_time = (2**retries) * EXPONENTIAL_BACKOFF_DELAY  # Exponential backoff
            print(f"Rate limit exceeded, retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)
            retries += 1


async def get_completion(
    client: openai.AsyncOpenAI,
    model_name: str,
    messages: list[dict],
    experiment_config: DictConfig,
    semaphore: asyncio.Semaphore,
) -> list:
    async with semaphore:
        return await get_completion_with_retry(
            client=client,
            model_name=model_name,
            messages=messages,
            experiment_config=experiment_config,
        )


async def run_with_progress(coro, pbar):
    result = await coro
    pbar.update(1)
    return result


async def get_all_completions(
    model: openai.AsyncOpenAI,
    processed_dataset: list[list[dict]],
    experiment_config,
    concurrency_limit: int = 5,
) -> list:
    semaphore = asyncio.Semaphore(concurrency_limit)

    # Create tasks in the same order as processed_dataset
    tasks = [
        get_completion(
            client=model,
            model_name=experiment_config.experiments.model.name,
            messages=current_prompt,
            experiment_config=experiment_config,
            semaphore=semaphore,
        )
        for current_prompt in processed_dataset
    ]

    # Wrap each task to update the progress bar upon completion.
    with tqdm(total=len(tasks)) as pbar:
        wrapped_tasks = [run_with_progress(task, pbar) for task in tasks]
        # asyncio.gather preserves the order of tasks
        ordered_results = await asyncio.gather(*wrapped_tasks)

    return ordered_results


def _prompt_template(
    essay_example: str, grade_index: int, experiment_config: DictConfig
):
    instructions_text = None
    if experiment_config.experiments.model.type in [
        ModelTypesEnum.CHATGPT_4O.value,
        ModelTypesEnum.MARITACA_SABIA.value,
    ]:
        if grade_index == 0:
            instructions_text = {"role": "system", "content": CONCEPT1_SYSTEM}
        elif grade_index == 1:
            instructions_text = {"role": "system", "content": CONCEPT2_SYSTEM}
        elif grade_index == 2:
            instructions_text = {"role": "system", "content": CONCEPT3_SYSTEM}
        elif grade_index == 3:
            instructions_text = {"role": "system", "content": CONCEPT4_SYSTEM}
        elif grade_index == 4:
            instructions_text = {"role": "system", "content": CONCEPT5_SYSTEM}
        user_text = {
            "role": "user",
            "content": f"Qual é a nota da redação a seguir?\n\n{essay_example}",
        }
        return [instructions_text, user_text]
    if experiment_config.experiments.model.type in [ModelTypesEnum.DEEPSEEK_R1.value]:
        if grade_index == 0:
            instructions_text = CONCEPT1_SYSTEM
        elif grade_index == 1:
            instructions_text = CONCEPT2_SYSTEM
        elif grade_index == 2:
            instructions_text = CONCEPT3_SYSTEM
        elif grade_index == 3:
            instructions_text = CONCEPT4_SYSTEM
        elif grade_index == 4:
            instructions_text = CONCEPT5_SYSTEM
        user_text = {
            "role": "user",
            "content": f"{instructions_text}\n\nQual é a nota da redação a seguir?\n\n{essay_example}",
        }
        return [user_text]


def _prepare_instruction_template(
    examples: List[str], grade_index: int, experiment_config: DictConfig
):
    result = []
    for example in examples:
        result.append(_prompt_template(example, grade_index, experiment_config))
    return result


def save_inference_results_jsonl(
    dataset_test: Dataset,
    labels: list,
    grade_index: int,
    thinking_text: list[str],
    all_results: list,
    logger: Logger,
    experiment_id: str,
    jsonl_filename: str = "inference_results.jsonl",
) -> None:
    """
    Save inference results to a JSON Lines file.

    Parameters:
        dataset_test (Dataset): The test split of the dataset.
        labels (list): List of labels.
        grade_index (int): The grade index used.
        thinking_text (list[str]): List of thinking texts from the model.
        all_results (list): List of Competencia (EssayEvaluation) objects.
        jsonl_filename (str): Filename for saving the JSON Lines file.
        logger: Logger for logging.
        experiment_id (str): Identifier for the experiment.
    """
    # Retrieve additional fields from the dataset if available.
    ids = dataset_test["id"]
    id_prompts = dataset_test["id_prompt"]
    test_essays = dataset_test["essay_text"]
    reference = dataset_test["reference"]

    rows = []
    for idx, essay in enumerate(test_essays):
        row = {
            "id": ids[idx],
            "id_prompt": id_prompts[idx],
            "essay_text": essay,
            "label": labels[idx],
            "grade_index": grade_index,
            "reference": reference[idx],
            "thinking_text": thinking_text[idx],
            "justificativa": all_results[idx].justificativa,
            "pontuacao": all_results[idx].pontuacao,
        }
        rows.append(row)

    # Write each row as a JSON object on a new line
    with open(f"{experiment_id}_{jsonl_filename}", "w", encoding="utf-8") as jsonlfile:
        for row in rows:
            json_line = json.dumps(row, ensure_ascii=False)
            jsonlfile.write(json_line + "\n")

    logger.info(f"Inference results saved to {experiment_id}_{jsonl_filename}")


def api_inference_pipeline(
    experiment_config: DictConfig, logger: Logger
) -> Dict[str, float]:
    dataset = load_dataset(
        experiment_config.dataset.name,
        experiment_config.dataset.split,
        cache_dir=experiment_config.cache_dir,
    )
    test_set = dataset["test"]
    grade_index = experiment_config.experiments.dataset.grade_index
    test_essays = test_set["essay_text"]
    labels = test_set.map(lambda x: {"labels": x["grades"][grade_index]})["labels"]
    processed_dataset = _prepare_instruction_template(
        test_essays, grade_index, experiment_config
    )
    model = ModelFactory.create_model(experiment_config, logger)
    logger.info(f"Starting inference on {experiment_config.experiments.model.name}")
    all_results = asyncio.run(
        get_all_completions(
            model,
            processed_dataset,
            experiment_config,
            concurrency_limit=CONCURRENCY_LIMIT,
        )
    )
    thinking_text = ["" for _ in all_results]  # empty reference to not break api
    if experiment_config.experiments.model.type in [ModelTypesEnum.DEEPSEEK_R1.value]:
        thinking_text = [result[0] for result in all_results]
        all_results = [result[1] for result in all_results]
    logger.info("Inference Done. Storing results.")
    save_inference_results_jsonl(
        dataset_test=test_set,
        labels=labels,
        grade_index=grade_index,
        thinking_text=thinking_text,
        all_results=all_results,
        logger=logger,
        experiment_id=experiment_config.experiments.training_id,
    )
    predictions = [row.pontuacao for row in all_results]
    return compute_metrics((predictions, labels), experiment_config)
