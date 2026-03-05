"""
LLM API client for OpenAI-compatible endpoints.

Uses https://www.dmxapi.com/v1 as the default base URL.
Logs every call's prompt, response, token usage, cost, and latency.
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from openai import OpenAI

logger = logging.getLogger("transform_agent")


@dataclass
class LLMCallRecord:
    """Record of a single LLM API call."""
    messages: List[Dict[str, str]]
    response_text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Pricing per 1M tokens (USD) — update as needed
MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}

DEFAULT_BASE_URL = "https://www.dmxapi.com/v1"


def _calculate_cost(model: str, input_tokens: int, output_tokens: int) -> tuple:
    """Calculate cost based on model pricing."""
    pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
    input_cost = input_tokens * pricing["input"] / 1_000_000
    output_cost = output_tokens * pricing["output"] / 1_000_000
    return input_cost, output_cost, input_cost + output_cost


def call_llm(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0,
    max_tokens: int = 2500,
    top_p: float = 0.9,
    base_url: str = None,
    api_key: str = None,
) -> LLMCallRecord:
    """
    Call an OpenAI-compatible LLM API.

    Args:
        model: Model name (e.g., "gpt-4o")
        messages: List of message dicts [{"role": "...", "content": "..."}]
        temperature: Sampling temperature
        max_tokens: Max output tokens
        top_p: Top-p sampling
        base_url: API base URL (default: https://www.dmxapi.com/v1)
        api_key: API key (default: from OPENAI_API_KEY env var)

    Returns:
        LLMCallRecord with response text, token usage, cost, and latency
    """
    base_url = base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)
    api_key = api_key or os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")

    client = OpenAI(api_key=api_key, base_url=base_url)

    start_time = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )

    latency_ms = (time.time() - start_time) * 1000

    response_text = response.choices[0].message.content or ""
    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0
    total_tokens = usage.total_tokens if usage else 0

    input_cost, output_cost, total_cost = _calculate_cost(model, input_tokens, output_tokens)

    record = LLMCallRecord(
        messages=list(messages),  # snapshot, not reference
        response_text=response_text,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        latency_ms=round(latency_ms, 2),
        input_cost=round(input_cost, 6),
        output_cost=round(output_cost, 6),
        total_cost=round(total_cost, 6),
    )

    logger.info(f"LLM call: model={model}, tokens={input_tokens}+{output_tokens}={total_tokens}, "
                f"cost=${total_cost:.6f}, latency={latency_ms:.0f}ms")

    return record
