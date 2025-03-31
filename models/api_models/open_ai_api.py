import logging
from typing import Literal

import openai
from omegaconf import DictConfig
from pydantic import BaseModel


class Competencia(BaseModel):
    """
    Data model representing a competency with a justification and a fixed set of scores.
    """
    #C1
    #sintaxe: Literal["excelente", "boa", "regular", "deficitária", "inexistente"]
    #desvios: Literal["menos de dois", "poucos", "alguns", "muitos"]
    #C5
    #ação: Literal["presente", "ausente", "elemento nulo"]
    #agente: Literal["presente", "ausente", "elemento nulo"]
    #modo_meio: Literal["presente", "ausente"]
    #efeito: Literal["presente", "ausente"]
    #detalhamento: Literal["presente", "ausente"]
    #C4
    #elementos_coesivos: Literal["ausentes", "raros", "pontuais", "regulares", "constantes", "expressivos"]
    #repetições: Literal["excessivas", "muitas", "algumas", "poucas", "raras", "ausentes"]
    #inadequações: Literal["excessivas", "muitas", "algumas", "poucas", "sem"]
    #operador_argumentativo_interparagrafos: Literal["ausente", "um" , "dois"]
    #operador_argumentativo_intraparagrafos: Literal["presente", "ausente"]
    #monobloco: Literal["sim", "não"]
    #C3
    #tangencia: Literal["sim", "não"] 
    #abordagem_completa: Literal["sim", "não"]
    #direção: Literal["sim", "não"]
    #projeto_de_texto: Literal["muitas falhas", "algumas falhas", "poucas falhas", "estratégico"]
    #desenvolvimento: Literal["sem desenvolvimento", "apenas uma", "algumas", "maior parte", "todas"]
    #contradição: Literal["sim", "não"]
    #C2
    abordagem: Literal["fuga ao tema", "tangência", "abordagem completa"] 
    aglomerado: Literal["sim", "não"]
    outro_tipo_textual: Literal["sim", "não"]
    partes_embrionárias: Literal["duas", "uma", "nenhuma"] 
    conclusão_incompleta: Literal["sim", "não"]
    cópias_dos_motivadores: Literal["sim", "não"] 
    repertório_baseado_motivadores: Literal["sim", "não"]
    repertório_legitimado: Literal["sim", "não"] 
    repertório_pertinente: Literal["sim", "não"]
    repertório_produtivo: Literal["sim", "não"]
    justificativa: str
    pontuacao: Literal[0, 40, 80, 120, 160, 200]


def create_openai_client(cfg: DictConfig, logger: logging.Logger) -> openai.AsyncOpenAI:
    """
    Factory function to create an asynchronous OpenAI client using the provided configuration.

    Args:
        cfg (DictConfig): Configuration object with API details.
        logger (logging.Logger): Logger instance for logging client initialization.

    Returns:
        openai.AsyncOpenAI: An instance of the asynchronous OpenAI client.
    """
    logger.info(f"Setting up model {cfg.experiments.model.name} through OpenAI API")
    client = openai.AsyncOpenAI(
        api_key=cfg.experiments.model.api_key,
        base_url=cfg.experiments.model.api_url,
    )
    logger.info("OpenAI client initialized successfully.")
    return client