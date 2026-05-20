import re
from typing import Dict, List

import pandas as pd

from rlie.utils.logger_config import LoggerConfig
from rlie.utils.register import Register


extract_label_register = Register("extract_label")


def _logger():
    return LoggerConfig.get_logger("extract_label")


def _last_match(patterns: List[str], text: str) -> str | None:
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]
    return None


def _normalize_label(value: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


@extract_label_register.register("default")
def default_extract_label(text):
    logger = _logger()
    if text is None:
        logger.warning("Could not extract label from empty text")
        return "other"
    text = text.lower()
    answer = _last_match([r"final answer:\s+([^\.!\?;,{}]+)"], text)
    if answer:
        return answer.strip()
    logger.warning("Could not extract label from text: %s", text)
    return "other"


@extract_label_register.register("aigc_detect")
@extract_label_register.register("llamagc_detect")
def aigc_extract_label(text):
    logger = _logger()
    if text is None:
        logger.warning("Could not extract label from empty text")
        return "other"
    answer = _last_match([r"final answer:\s+(ai|human)"], text.lower())
    if answer == "ai":
        return "AI"
    if answer == "human":
        return "HUMAN"
    logger.warning("Could not extract label from text: %s", text)
    return "other"


@extract_label_register.register("headline_binary")
def headline_binary_extract_label(text):
    logger = _logger()
    if text is None:
        logger.warning("Could not extract label from empty text")
        return "other"
    answer = _last_match([r"answer:\s+(headline 1|headline 2|other)"], text.lower())
    if answer == "headline 1":
        return "Headline 1 has more clicks than Headline 2."
    if answer == "headline 2":
        return "Headline 2 has more clicks than Headline 1."
    logger.warning("Could not extract label from text: %s", text)
    return "other"


@extract_label_register.register("deceptive_reviews")
def deceptive_reviews_extract_label(text):
    logger = _logger()
    if text is None:
        logger.warning("Could not extract label from empty text")
        return "other"
    answer = _last_match([r"final answer:\s+(truthful|deceptive|other)"], text.lower())
    if answer in {"truthful", "deceptive"}:
        return answer
    logger.warning("Could not extract label from text: %s", text)
    return "other"


@extract_label_register.register("retweet")
def retweet_extract_label(text):
    logger = _logger()
    if text is None:
        logger.warning("Could not extract label from empty text")
        return "other"
    answer = _last_match([r"answer: the (\w+) tweet"], text.lower())
    if answer:
        return answer
    logger.warning("Could not extract label from text: %s", text)
    return "other"


@extract_label_register.register("persuasive_pairs")
def persuasive_pairs_extract_label(text):
    logger = _logger()
    if text is None:
        logger.warning("Could not extract label from empty text")
        return "other"
    patterns = [
        r"answer: the (\w+) argument",
        r"answer: \[the (\w+) argument",
        r"answer: (\w+) argument",
    ]
    answer = _last_match(patterns, text.lower())
    if answer == "first":
        return "first"
    if answer == "second":
        return "second"
    logger.warning("Could not extract label from text: %s", text)
    return "other"


@extract_label_register.register("dreaddit")
def dreaddit_extract_label(text):
    logger = _logger()
    if text is None:
        logger.warning("Could not extract label from empty text")
        return "other"
    answer = _last_match(
        [
            r"answer: (\w+) stress",
            r"answer: \[(\w+) stress",
        ],
        text.lower(),
    )
    if answer == "has":
        return "has stress"
    if answer == "no":
        return "no stress"
    logger.warning("Could not extract label from text: %s", text)
    return "other"


@extract_label_register.register("paper_citation")
def paper_citation_extract_label(text):
    logger = _logger()
    if text is None:
        logger.warning("Could not extract label from empty text")
        return "other"
    answer = _last_match(
        [
            r"final answer:\s+(impactful|unimpactful|not applicable|other)",
            r"answer:\s+(impactful|unimpactful|not applicable|other)",
        ],
        text.lower(),
    )
    if answer in {"impactful", "unimpactful"}:
        return answer
    if answer == "not applicable":
        return "other"
    logger.warning("Could not extract label from text: %s", text)
    return "other"


def map_labels(df: pd.DataFrame, label_mapping: Dict[str, int], label_column: str) -> List[int]:
    mapped: List[int] = []
    for value in df[label_column]:
        key = _normalize_label(value)
        if key not in label_mapping:
            raise KeyError(
                f"Label '{value}' (normalized: '{key}') not found in mapping {label_mapping}."
            )
        mapped.append(label_mapping[key])
    return mapped
