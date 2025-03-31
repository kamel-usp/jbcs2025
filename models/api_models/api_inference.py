import asyncio
import json
import re
from collections import Counter
from dataclasses import dataclass
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

CONCURRENCY_LIMIT = 1
EXPONENTIAL_BACKOFF_DELAY = 120
NUMBER_REPETITION_EVAL = 1


@dataclass
class AggregatedCompetencia:
    abordagem: list
    aglomerado: list
    outro_tipo_textual: list
    partes_embrionárias: list
    conclusão_incompleta: list
    cópias_dos_motivadores: list
    repertório_baseado_motivadores: list
    repertório_legitimado: list
    repertório_pertinente: list
    repertório_produtivo: list
    justificativa: list
    pontuacao: int


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
    experiment_config,
    max_retries: int = 5,
    ensamble_model_call: int = 10,
) -> AggregatedCompetencia | AggregatedReasoningCompetencia:
    retries = 0
    while True:
        try:
            responses = []
            if experiment_config.experiments.model.type in [
                ModelTypesEnum.CHATGPT_4O.value,
                ModelTypesEnum.MARITACA_SABIA.value,
            ]:
                for _ in range(ensamble_model_call):
                    completion = await client.beta.chat.completions.parse(
                        model=model_name,
                        messages=messages,
                        response_format=Competencia,
                        max_tokens=experiment_config.experiments.model.max_tokens,
                        temperature=experiment_config.experiments.model.temperature,
                        seed=experiment_config.experiments.model.seed,
                    )
                    # Append the parsed response from the first choice.
                    responses.append(completion.choices[0].message.parsed)
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
                for _ in range(ensamble_model_call):
                    completion = await client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        stop=list(experiment_config.experiments.model.stop),
                        stream=False,
                    )
                    think_text = completion.choices[0].message.reasoning_content
                    content = completion.choices[0].message.content
                    evaluation = parse_deepseek_response(content)
                    answer = ReasoningCompetencia(
                        thinking=think_text, competencia=evaluation
                    )
                    responses.append(answer)
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
) -> AggregatedReasoningCompetencia | AggregatedCompetencia:
    async with semaphore:
        return await get_completion_with_retry(
            client=client,
            model_name=model_name,
            messages=messages,
            experiment_config=experiment_config,
            ensamble_model_call=NUMBER_REPETITION_EVAL,
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
        )
        for current_prompt in processed_dataset
    ]

    # Wrap each task to update the progress bar upon completion.
    with tqdm(total=len(tasks)) as pbar:
        wrapped_tasks = [run_with_progress(task, pbar) for task in tasks]
        # asyncio.gather preserves the order of tasks
        ordered_results = await asyncio.gather(*wrapped_tasks)

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
        if experiment_config.experiments.model.use_essay_prompt is False:
            user_text = {
                "role": "user",
                "content": f"Qual é a nota da redação a seguir?\n\n{example['essay_text']}",
            }
        elif experiment_config.experiments.model.use_essay_prompt is True:
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
        if grade_index == 1:
            row = {
                "id": ids[idx],
                "id_prompt": id_prompts[idx],
                "essay_text": essay,
                "label": labels[idx],
                "grade_index": grade_index,
                "reference": reference[idx],
                "thinking_text": thinking_text[idx],
                "justificativa": all_results[idx].justificativa,
                "abordagem": all_results[idx].abordagem, 
                "aglomerado": all_results[idx].aglomerado,
                "outro_tipo_textual": all_results[idx].outro_tipo_textual,
                "partes_embrionárias": all_results[idx].partes_embrionárias, 
                "conclusão_incompleta": all_results[idx].conclusão_incompleta,
                "cópias_dos_motivadores": all_results[idx].cópias_dos_motivadores, 
                "repertório_baseado_motivadores": all_results[idx].repertório_baseado_motivadores,
                "repertório_legitimado": all_results[idx].repertório_legitimado, 
                "repertório_pertinente": all_results[idx].repertório_pertinente, 
                "repertório_produtivo": all_results[idx].repertório_produtivo,
                "pontuacao": all_results[idx].pontuacao,
            }
        else:
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
    labels = test_set.map(lambda x: {"labels": x["grades"][grade_index]})["labels"]
    processed_dataset = _prepare_instruction_template(
        test_set, grade_index, experiment_config
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
        experiment_id=experiment_config.experiments.training_id,
    )
    predictions = [row.pontuacao for row in all_results]
    return compute_metrics((predictions, labels), experiment_config)
