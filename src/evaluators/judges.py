"""
LLM Judge implementations: pointwise, pairwise, and reference-based.

Each judge takes a prompt/response (or pair) and returns a structured judgment.
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal
from openai import OpenAI


@dataclass
class Judgment:
    """A single judgment from an LLM judge."""
    judge_model: str
    eval_format: str
    sample_id: str
    run_index: int
    score: Optional[float] = None        # pointwise: 1-10, pairwise: -1/0/1
    winner: Optional[str] = None          # "A", "B", or "tie"
    reasoning: str = ""
    tokens_used: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def get_client():
    return OpenAI()


# ---------------------------------------------------------------------------
# Pairwise judge
# ---------------------------------------------------------------------------

PAIRWISE_PROMPT = """You are an impartial judge. Compare the two responses below to the given question and decide which is better.

[Question]
{question}

[Response A]
{response_a}

[Response B]
{response_b}

Evaluate based on helpfulness, relevance, accuracy, depth, and clarity.
First provide a brief explanation, then output your verdict as exactly one of:
[[A]] if Response A is better
[[B]] if Response B is better
[[tie]] if they are equally good"""

PAIRWISE_WITH_TIE_PROMPT = PAIRWISE_PROMPT  # same, tie is explicit option


def judge_pairwise(
    model: str,
    question: str,
    response_a: str,
    response_b: str,
    sample_id: str = "",
    run_index: int = 0,
    allow_tie: bool = True,
) -> Judgment:
    """Pairwise comparison: which response is better?"""
    client = get_client()
    prompt = PAIRWISE_PROMPT.format(
        question=question, response_a=response_a, response_b=response_b
    )

    start = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            max_tokens=512,
        )
        content = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0
        latency = (time.time() - start) * 1000

        # Parse verdict
        winner = None
        score = 0
        if "[[A]]" in content:
            winner = "A"
            score = 1
        elif "[[B]]" in content:
            winner = "B"
            score = -1
        elif "[[tie]]" in content.lower():
            winner = "tie"
            score = 0

        return Judgment(
            judge_model=model, eval_format="pairwise", sample_id=sample_id,
            run_index=run_index, score=score, winner=winner,
            reasoning=content, tokens_used=tokens, latency_ms=latency,
        )
    except Exception as e:
        return Judgment(
            judge_model=model, eval_format="pairwise", sample_id=sample_id,
            run_index=run_index, error=str(e),
            latency_ms=(time.time() - start) * 1000,
        )


# ---------------------------------------------------------------------------
# Pointwise judge
# ---------------------------------------------------------------------------

POINTWISE_PROMPT_10 = """You are an impartial judge. Rate the following response to the given question on a scale of 1 to 10.

[Question]
{question}

[Response]
{response}

Evaluate based on helpfulness, relevance, accuracy, depth, and clarity.
First provide a brief explanation, then output your rating as exactly: [[score]] where score is an integer from 1 to 10."""

POINTWISE_PROMPT_5 = """You are an impartial judge. Rate the following response to the given question on a scale of 1 to 5.

[Question]
{question}

[Response]
{response}

Evaluate based on helpfulness, relevance, accuracy, depth, and clarity.
First provide a brief explanation, then output your rating as exactly: [[score]] where score is an integer from 1 to 5."""


def judge_pointwise(
    model: str,
    question: str,
    response_text: str,
    sample_id: str = "",
    run_index: int = 0,
    scale: int = 10,
) -> Judgment:
    """Pointwise scoring on a 1-N scale."""
    client = get_client()
    prompt_template = POINTWISE_PROMPT_10 if scale == 10 else POINTWISE_PROMPT_5
    prompt = prompt_template.format(question=question, response=response_text)

    start = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            max_tokens=512,
        )
        content = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0
        latency = (time.time() - start) * 1000

        # Parse score - look for [[N]]
        import re
        match = re.search(r'\[\[(\d+)\]\]', content)
        score = int(match.group(1)) if match else None

        return Judgment(
            judge_model=model, eval_format=f"pointwise_1to{scale}",
            sample_id=sample_id, run_index=run_index, score=score,
            reasoning=content, tokens_used=tokens, latency_ms=latency,
        )
    except Exception as e:
        return Judgment(
            judge_model=model, eval_format=f"pointwise_1to{scale}",
            sample_id=sample_id, run_index=run_index, error=str(e),
            latency_ms=(time.time() - start) * 1000,
        )


# ---------------------------------------------------------------------------
# Reference-based judge
# ---------------------------------------------------------------------------

REFERENCE_PROMPT = """You are an impartial judge. Rate the following response compared to the reference answer.

[Question]
{question}

[Reference Answer]
{reference}

[Response to Evaluate]
{response}

How well does the response match the reference in terms of correctness and completeness?
First provide a brief explanation, then output your rating as exactly: [[score]] where score is an integer from 1 to 10."""


def judge_reference(
    model: str,
    question: str,
    response_text: str,
    reference: str,
    sample_id: str = "",
    run_index: int = 0,
) -> Judgment:
    """Reference-based grading."""
    client = get_client()
    prompt = REFERENCE_PROMPT.format(
        question=question, reference=reference, response=response_text
    )

    start = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            max_tokens=512,
        )
        content = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0
        latency = (time.time() - start) * 1000

        import re
        match = re.search(r'\[\[(\d+)\]\]', content)
        score = int(match.group(1)) if match else None

        return Judgment(
            judge_model=model, eval_format="reference_based",
            sample_id=sample_id, run_index=run_index, score=score,
            reasoning=content, tokens_used=tokens, latency_ms=latency,
        )
    except Exception as e:
        return Judgment(
            judge_model=model, eval_format="reference_based",
            sample_id=sample_id, run_index=run_index, error=str(e),
            latency_ms=(time.time() - start) * 1000,
        )
