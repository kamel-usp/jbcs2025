import asyncio
from logging import Logger
from typing import Dict, List

import openai
from datasets import load_dataset
from omegaconf import DictConfig
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
            if experiment_config.experiments.model.type in [ModelTypesEnum.CHATGPT_4O.value, ModelTypesEnum.MARITACA_SABIA.value]:
                completion = await client.beta.chat.completions.parse(
                    model=model_name,
                    messages=messages,
                    response_format=Competencia,
                    max_tokens=experiment_config.experiments.model.max_tokens,
                    temperature=experiment_config.experiments.model.temperature,
                    seed=experiment_config.experiments.model.seed,
                )
                return completion.choices[0].message.parsed
            if experiment_config.experiments.model.type in [ModelTypesEnum.DEEPSEEK_R1.value]:
                completion = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=experiment_config.experiments.model.max_tokens
                )
                reasoning_conent = completion.choices[0].message.reasoning_conent
                content = completion.choices[0].message.content
                return content
        except openai.RateLimitError as e:
            if retries >= max_retries:
                raise e
            wait_time = (2**retries) * 60  # Exponential backoff
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


def _prompt_template(essay_example: str, grade_index: int):
    instructions_text = None
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


def _prepare_instruction_template(examples: List[str], grade_index: int):
    result = []
    for example in examples:
        result.append(_prompt_template(example, grade_index))
    return result


def api_inference_pipeline(
    experiment_config: DictConfig, logger: Logger
) -> Dict[str, float]:
    dataset = load_dataset(
        experiment_config.dataset.name,
        experiment_config.dataset.split,
        cache_dir=experiment_config.cache_dir,
    )
    grade_index = experiment_config.experiments.dataset.grade_index
    test_essays = dataset["test"]["essay_text"]
    labels = dataset["test"].map(lambda x: {"labels": x["grades"][grade_index]})[
        "labels"
    ]
    processed_dataset = _prepare_instruction_template(test_essays, grade_index)
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
    predictions = [row.pontuacao for row in all_results]
    return compute_metrics((predictions, labels), experiment_config)
