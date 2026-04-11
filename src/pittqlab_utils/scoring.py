import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .llm import PittAIClient


@dataclass
class RubricCategory:
    name: str
    max_points: int
    descriptor: str
    levels: Dict[str, str]


@dataclass
class Rubric:
    name: str
    categories: List[RubricCategory]
    bonus_categories: List[RubricCategory] = field(default_factory=list)
    anchor_examples: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CategoryScore:
    category_name: str
    score: int
    max_points: int
    justification: str
    flags: List[str] = field(default_factory=list)


@dataclass
class ScoringResult:
    document_id: str
    category_scores: List[CategoryScore]
    bonus_scores: List[CategoryScore]
    total_score: float
    max_score: float
    flags: List[str]
    raw_responses: Dict[str, str]


def _select_anchor_examples(rubric: Rubric, category_name: str) -> List[Dict[str, Any]]:
    matching = [
        example for example in rubric.anchor_examples
        if example.get("category") == category_name
    ]
    return matching[:2]


def _build_system_prompt(category: RubricCategory, rubric: Rubric) -> str:
    anchor_examples = _select_anchor_examples(rubric, category.name)

    lines = [
        "You are an admissions scoring assistant.",
        "Score exactly one rubric category for the provided document text.",
        "",
        "Rubric category details:",
        "category_name: {0}".format(category.name),
        "max_points: {0}".format(category.max_points),
        "descriptor: {0}".format(category.descriptor),
        "levels: {0}".format(json.dumps(category.levels, ensure_ascii=True)),
    ]

    if anchor_examples:
        lines.append("")
        lines.append("Anchor examples for calibration (few-shot):")
        for index, example in enumerate(anchor_examples, start=1):
            lines.append(
                "Example {0}: {1}".format(
                    index,
                    json.dumps(example, ensure_ascii=True),
                )
            )

    lines.extend(
        [
            "",
            "Return ONLY valid JSON with this exact schema:",
            '{"score": <int>, "justification": "<2-3 sentences>", "flags": ["flag1", "flag2"]}',
        ]
    )
    return "\n".join(lines)


def _parse_category_response(
    category: RubricCategory,
    raw_text: str,
) -> CategoryScore:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return CategoryScore(
            category_name=category.name,
            score=0,
            max_points=category.max_points,
            justification="[parse error: {0}]".format(raw_text),
            flags=["low_confidence"],
        )

    score_value = parsed.get("score", 0)
    try:
        score = int(score_value)
    except (TypeError, ValueError):
        score = 0

    justification_value = parsed.get("justification", "")
    justification = str(justification_value)

    raw_flags = parsed.get("flags", [])
    flags: List[str] = []
    if isinstance(raw_flags, list):
        for flag in raw_flags:
            if isinstance(flag, str):
                flags.append(flag)

    return CategoryScore(
        category_name=category.name,
        score=score,
        max_points=category.max_points,
        justification=justification,
        flags=flags,
    )


def _score_category(
    document_text: str,
    category: RubricCategory,
    rubric: Rubric,
    client: PittAIClient,
    model: Optional[str],
) -> Tuple[CategoryScore, str]:
    system_prompt = _build_system_prompt(category, rubric)

    try:
        response = client.chat(
            prompt=document_text,
            system=system_prompt,
            model=model,
            json_mode=True,
        )
    except Exception as exc:  # pragma: no cover - tested via public function path
        message = "[API error: {0}]".format(exc)
        return (
            CategoryScore(
                category_name=category.name,
                score=0,
                max_points=category.max_points,
                justification=message,
                flags=["api_error"],
            ),
            message,
        )

    raw_text = response.text
    parsed_score = _parse_category_response(category, raw_text)
    return parsed_score, raw_text


def score_with_rubric(
    document_text: str,
    rubric: Rubric,
    client: PittAIClient,
    document_id: str = "",
    model: Optional[str] = None,
) -> ScoringResult:
    category_scores: List[CategoryScore] = []
    bonus_scores: List[CategoryScore] = []
    raw_responses: Dict[str, str] = {}

    for category in rubric.categories:
        score, raw_response = _score_category(document_text, category, rubric, client, model)
        category_scores.append(score)
        raw_responses[category.name] = raw_response

    for category in rubric.bonus_categories:
        score, raw_response = _score_category(document_text, category, rubric, client, model)
        bonus_scores.append(score)
        raw_responses[category.name] = raw_response

    total_score = float(
        sum(item.score for item in category_scores) +
        sum(item.score for item in bonus_scores)
    )
    max_score = float(sum(category.max_points for category in rubric.categories))

    all_flags: List[str] = []
    for item in category_scores + bonus_scores:
        all_flags.extend(item.flags)
    unique_flags = list(dict.fromkeys(all_flags))

    return ScoringResult(
        document_id=document_id,
        category_scores=category_scores,
        bonus_scores=bonus_scores,
        total_score=total_score,
        max_score=max_score,
        flags=unique_flags,
        raw_responses=raw_responses,
    )
