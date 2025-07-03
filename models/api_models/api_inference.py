import asyncio
import json
import re
import math
from collections import Counter
from dataclasses import dataclass
from logging import Logger
from typing import Dict, List

import numpy as np
import openai
from datasets import (
    Dataset,
    load_dataset,
)
from omegaconf import DictConfig
from pydantic import ValidationError
from tqdm.asyncio import tqdm_asyncio

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

CONCURRENCY_LIMIT = 50 
EXPONENTIAL_BACKOFF_DELAY = 30
REQUEST_TIMEOUT = 300


@dataclass
class AggregatedCompetencia:
    pontuacao: int
    justificativa: list


@dataclass
class ReasoningCompetencia:
    thinking: str
    competencia: Competencia


@dataclass
class AggregatedReasoningCompetencia:
    thinking_list: list
    aggregated_competencia: AggregatedCompetencia


def parse_deepseek_response(api_response: str) -> tuple[str, Competencia]:
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

    return evaluation


async def get_completion_with_retry(
    client: openai.AsyncOpenAI,
    model_name: str,
    messages: list[dict],
    number_of_calls_per_model: int,
    experiment_config,
    logger: Logger,
    line_id: int = -1,
    max_retries: int = 5,
) -> AggregatedCompetencia | AggregatedReasoningCompetencia:
    api_retries = 0
    responses = []  # Move outside retry loop to preserve successful responses
    successful_calls = 0  # Track successful calls separately
    
    while successful_calls < number_of_calls_per_model:
        try:
            if experiment_config.experiments.model.type in [
                ModelTypesEnum.CHATGPT_4O.value,
                ModelTypesEnum.MARITACA_SABIA.value,
            ]:
                # Early consensus optimization
                early_consensus_threshold = max(1, math.ceil(number_of_calls_per_model / 2))
                score_counts = Counter()
                
                # Continue from where we left off
                for call_num in range(successful_calls, number_of_calls_per_model):
                    logger.info(f"[Line {line_id}] Making API call {call_num + 1}/{number_of_calls_per_model} to {model_name}")
                    completion = await client.beta.chat.completions.parse(
                        model=model_name,
                        messages=messages,
                        response_format=Competencia,
                        max_tokens=experiment_config.experiments.model.max_tokens,
                        temperature=experiment_config.experiments.model.temperature,
                        seed=experiment_config.experiments.model.seed,
                        timeout=REQUEST_TIMEOUT
                    )
                    response = completion.choices[0].message.parsed
                    responses.append(response)
                    successful_calls += 1
                    score_counts[response.pontuacao] += 1
                    
                    # Check for early consensus
                    if any(count >= early_consensus_threshold for count in score_counts.values()):
                        logger.info(f"[Line {line_id}] Early consensus reached after {successful_calls} calls")
                        break
                
                # If we reach here, all calls were successful
                essay_scores = [resp.pontuacao for resp in responses]
                explanations = [resp.justificativa for resp in responses]
                if essay_scores:
                    most_common_score = Counter(essay_scores).most_common(1)[0][0]
                final_answer = AggregatedCompetencia(
                    pontuacao=most_common_score, justificativa=explanations
                )
                return final_answer
                
            if experiment_config.experiments.model.type in [
                ModelTypesEnum.DEEPSEEK_R1.value
            ]:
                parsing_retries = 0
                score_counts = Counter()
                early_consensus_threshold = max(1, math.ceil(number_of_calls_per_model / 2))
                
                while successful_calls < number_of_calls_per_model:
                    try:
                        logger.info(f"[Line {line_id}] Making API call {successful_calls + 1}/{number_of_calls_per_model} to {model_name}")
                        completion = await client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            stop=list(experiment_config.experiments.model.stop),
                            stream=False,
                            timeout=REQUEST_TIMEOUT,
                        )
                        think_text = completion.choices[0].message.reasoning_content
                        content = completion.choices[0].message.content

                        # This might raise ValueError
                        evaluation = parse_deepseek_response(content)

                        # If we get here, parsing was successful
                        answer = ReasoningCompetencia(
                            thinking=think_text, competencia=evaluation
                        )
                        responses.append(answer)
                        successful_calls += 1
                        parsing_retries = 0  # Reset parsing retries after success
                        
                        # Track scores for early consensus
                        score_counts[evaluation.pontuacao] += 1
                        if any(count >= early_consensus_threshold for count in score_counts.values()):
                            logger.info(f"[Line {line_id}] Early consensus reached after {successful_calls} calls")
                            break

                    except ValueError as parse_error:
                        # For JSON parsing errors, use a separate retry counter
                        parsing_retries += 1
                        wait_time = (2**parsing_retries) * EXPONENTIAL_BACKOFF_DELAY
                        logger.warning(
                            f"[Line {line_id}] JSON parsing error: {parse_error}. "
                            f"Parsing retry #{parsing_retries} in {wait_time} seconds..."
                        )
                        await asyncio.sleep(wait_time)
                        continue

                # Only reach this point when we have enough valid responses
                essay_scores = [resp.competencia.pontuacao for resp in responses]
                explanations = [resp.competencia.justificativa for resp in responses]
                thinking_list = [resp.thinking for resp in responses]
                if essay_scores:
                    most_common_score = Counter(essay_scores).most_common(1)[0][0]
                final_answer = AggregatedReasoningCompetencia(
                    thinking_list=thinking_list,
                    aggregated_competencia=AggregatedCompetencia(
                        pontuacao=most_common_score, justificativa=explanations
                    ),
                )
                return final_answer
        except Exception as e:
            # For non-parsing errors (API issues, network problems, etc.)
            if api_retries >= max_retries:
                logger.error(f"[Line {line_id}] API error after {max_retries} retries: {str(e)}. Giving up.")
                raise e
            api_retries += 1
            wait_time = (2**api_retries) * EXPONENTIAL_BACKOFF_DELAY
            logger.warning(
                f"[Line {line_id}] API error occurred: {str(e)}. API retry #{api_retries}/{max_retries} in {wait_time} seconds..."
            )
            await asyncio.sleep(wait_time)


async def get_completion(
    client: openai.AsyncOpenAI,
    model_name: str,
    messages: list[dict],
    experiment_config: DictConfig,
    number_repetition_eval: int,
    semaphore: asyncio.Semaphore,
    logger: Logger,
    line_id: int = -1,
) -> AggregatedReasoningCompetencia | AggregatedCompetencia:
    async with semaphore:
        return await get_completion_with_retry(
            client=client,
            model_name=model_name,
            messages=messages,
            experiment_config=experiment_config,
            number_of_calls_per_model=number_repetition_eval,
            logger=logger,
            line_id=line_id,
        )


async def run_with_progress(coro, pbar):
    result = await coro
    pbar.update(1)
    return result


async def get_all_completions(
    model: openai.AsyncOpenAI,
    processed_dataset: list[list[dict]],
    experiment_config,
    number_repetition_eval,
    logger: Logger,
    concurrency_limit: int = CONCURRENCY_LIMIT,  # Use the increased limit
) -> List[AggregatedCompetencia | AggregatedReasoningCompetencia]:
    semaphore = asyncio.Semaphore(concurrency_limit)

    # Create tasks in the same order as processed_dataset
    tasks = [
        get_completion(
            client=model,
            model_name=experiment_config.experiments.model.name,
            messages=current_prompt,
            experiment_config=experiment_config,
            semaphore=semaphore,
            number_repetition_eval=number_repetition_eval,
            logger=logger,
            line_id=idx,
        )
        for idx, current_prompt in enumerate(processed_dataset)
    ]

    ordered_results = await tqdm_asyncio.gather(
        *tasks,
        desc="Processing completions",
        total=len(tasks)
    )

    return ordered_results


def _prompt_template(example: dict, grade_index: int, experiment_config: DictConfig):
    instructions_text = None
    if experiment_config.experiments.model.type in [
        ModelTypesEnum.CHATGPT_4O.value,
        ModelTypesEnum.MARITACA_SABIA.value,
    ]:
        user_text = {}
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
            "content": f"Qual é a nota da redação a seguir?\n\n{example['essay_text']}",
        }
        if experiment_config.experiments.dataset.use_full_context is True:
            user_text = {
                "role": "user",
                "content": (
                    f"Considere os textos de apoio:\n\n{example['supporting_text']}.\n\n"
                    f"Agora, o tema da redação é descrito a seguir:\n\n{example['prompt']}\n\n"
                    f"Com base no tema e nos textos, qual é a nota da redação a seguir?\n\n"
                    f"{example['essay_text']}"
                ),
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
            "content": f"{instructions_text}\n\nQual é a nota da redação a seguir?\n\n{example['essay_text']}",
        }
        if experiment_config.experiments.dataset.use_full_context is True:
            user_text = {
                "role": "user",
                "content": (
                    f"{instructions_text}\n\nConsidere os textos de apoio:\n\n{example['supporting_text']}.\n\n"
                    f"Agora, o tema da redação é descrito a seguir:\n\n{example['prompt']}\n\n"
                    f"Com base no tema e nos textos, qual é a nota da redação a seguir?\n\n"
                    f"{example['essay_text']}"
                ),
            }
        return [user_text]


def _prepare_instruction_template(
    test_set,
    grade_index: int,
    experiment_config: DictConfig,
):
    result = []
    for example in test_set:
        result.append(_prompt_template(example, grade_index, experiment_config))
    return result


def save_inference_results_jsonl(
    dataset_test: Dataset,
    labels: list,
    grade_index: int,
    thinking_text: list[str],
    all_results: List[AggregatedCompetencia | AggregatedReasoningCompetencia],
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
    from scripts.run_experiment import get_experiment_id  # avoid circular import

    experiment_id = get_experiment_id(experiment_config)
    dataset = load_dataset(
        experiment_config.dataset.name,
        experiment_config.dataset.split,
        cache_dir=experiment_config.cache_dir,
    )
    test_set = dataset["test"]
    grade_index = experiment_config.experiments.dataset.grade_index
    labels = test_set.map(lambda x: {"labels": x["grades"][grade_index]})["labels"]
    processed_dataset = _prepare_instruction_template(
        test_set, grade_index, experiment_config
    )
    model = ModelFactory.create_model(experiment_config, logger)
    logger.info(f"Starting inference on {experiment_config.experiments.model.name}")
    logger.info(
        f"We will run inference {experiment_config.experiments.model.number_repetition_eval} times per row"
    )
    all_results = asyncio.run(
        get_all_completions(
            model,
            processed_dataset,
            experiment_config,
            logger=logger,
            concurrency_limit=CONCURRENCY_LIMIT,
            number_repetition_eval=experiment_config.experiments.model.number_repetition_eval,
        )
    )
    thinking_text = ["" for _ in all_results]  # empty reference to not break api
    if experiment_config.experiments.model.type in [ModelTypesEnum.DEEPSEEK_R1.value]:
        thinking_text = [result.thinking_list for result in all_results]
        all_results = [result.aggregated_competencia for result in all_results]
    logger.info("Inference Done. Storing results.")
    save_inference_results_jsonl(
        dataset_test=test_set,
        labels=labels,
        grade_index=grade_index,
        thinking_text=thinking_text,
        all_results=all_results,
        logger=logger,
        experiment_id=experiment_id,
    )
    predictions = np.array([row.pontuacao for row in all_results])
    labels = np.array(labels)
    return (
        compute_metrics((predictions, labels), experiment_config),
        predictions,
        labels,
        test_set,
    )
