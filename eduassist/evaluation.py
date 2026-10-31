"""Offline-testable metrics and production eligibility rules."""

import json
import math
import statistics
from typing import Any


def request_cost(
    input_tokens: int,
    output_tokens: int,
    input_price: float,
    output_price: float,
    cached_tokens: int = 0,
    cached_price: float = 0.0,
) -> dict[str, float]:
    values = (input_tokens, output_tokens, input_price, output_price, cached_tokens, cached_price)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Tokens and prices must be finite and nonnegative.")
    if cached_tokens > input_tokens:
        raise ValueError("Cached tokens cannot exceed input tokens.")
    input_cost = input_tokens * input_price / 1_000_000
    output_cost = output_tokens * output_price / 1_000_000
    adjusted_input = (
        (input_tokens - cached_tokens) * input_price + cached_tokens * cached_price
    ) / 1_000_000
    return {
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": input_cost + output_cost,
        "cache_adjusted_cost_usd": adjusted_input + output_cost,
    }


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-JSON numeric constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key.")
        result[key] = value
    return result


def validate_response(text: str) -> dict[str, bool]:
    try:
        payload = json.loads(
            text, parse_constant=_reject_constant, object_pairs_hook=_unique_object
        )
    except (ValueError, TypeError):
        return {"valid_json": False, "schema_valid": False}
    schema_valid = (
        isinstance(payload, dict)
        and set(payload) == {"answer", "confidence", "key_points"}
        and isinstance(payload["answer"], str)
        and isinstance(payload["confidence"], str)
        and payload["confidence"] in {"high", "medium", "low"}
        and isinstance(payload["key_points"], list)
        and all(isinstance(point, str) for point in payload["key_points"])
    )
    return {"valid_json": True, "schema_valid": schema_valid}


def latency_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    values = [float(row["latency_seconds"]) for row in rows]
    if not values:
        raise ValueError("No requests to summarize.")
    return {
        "request_count": len(values),
        "mean_seconds": statistics.mean(values),
        "median_seconds": statistics.median(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
        "stddev_seconds": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def summarize_version(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No requests to evaluate.")
    successful = [row for row in rows if row["status"] == "success"]
    priced = [row for row in rows if row.get("total_cost_usd") is not None]
    metrics: dict[str, Any] = {
        "request_count": len(rows),
        "success_rate": len(successful) / len(rows),
        "valid_json": sum(bool(row["valid_json"]) for row in rows) / len(rows),
        "schema_valid_rate": sum(bool(row["schema_valid"]) for row in rows) / len(rows),
        "avg_latency_seconds": statistics.mean(row["latency_seconds"] for row in rows),
        "avg_input_tokens": None,
        "avg_output_tokens": None,
        "avg_cost_per_request": None,
        "projected_monthly_cost_usd": None,
    }
    if priced:
        metrics.update({
            "avg_input_tokens": statistics.mean(row["input_tokens"] for row in priced),
            "avg_output_tokens": statistics.mean(row["output_tokens"] for row in priced),
            "avg_cost_per_request": statistics.mean(row["total_cost_usd"] for row in priced),
        })
        metrics["projected_monthly_cost_usd"] = metrics["avg_cost_per_request"] * 10_000
    return metrics


def select_candidate(comparison: list[dict[str, Any]]) -> str | None:
    eligible = [
        row for row in comparison
        if row["request_count"] == 5
        and row["success_rate"] == 1.0
        and row["schema_valid_rate"] == 1.0
        and row["avg_cost_per_request"] is not None
    ]
    if not eligible:
        return None
    best = min(eligible, key=lambda row: (
        row["avg_cost_per_request"], row["avg_latency_seconds"], row["prompt_version"]
    ))
    return str(best["prompt_version"])