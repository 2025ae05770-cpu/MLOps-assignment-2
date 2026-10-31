"""Live API evaluation with persistent MLflow prompt and request evidence."""

import argparse
import csv
import difflib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from eduassist.evaluation import (
    latency_summary, request_cost, select_candidate, summarize_version, validate_response,
)
from eduassist.prompts import PROMPTS, QUESTIONS

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty table.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def tracking_uri(root: Path = ROOT) -> str:
    runtime = root / "runtime"
    runtime.mkdir(exist_ok=True, parents=True)
    return os.environ.get("MLFLOW_TRACKING_URI") or "sqlite:///" + (runtime / "mlflow.db").as_posix()


def call_question(client: Any, prompt: Any, question: dict[str, str], config: dict[str, Any]) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    timer = time.perf_counter()
    row: dict[str, Any] = {
        "question_id": question["question_id"], "question": question["question"],
        "prompt_version": "", "registry_version": str(prompt.version),
        "prompt_uri": f"prompts:/{prompt.name}/{prompt.version}",
        "model_requested": config["model"], "model_name": "", "api_version": config["api_version"],
        "start_time_utc": started, "end_time_utc": "", "latency_seconds": 0.0,
        "status": "error", "http_status": None, "error_type": "", "finish_reason": "",
        "input_tokens": None, "output_tokens": None, "total_tokens": None,
        "cached_input_tokens": None, "reasoning_tokens": None,
        "input_cost_usd": None, "output_cost_usd": None, "total_cost_usd": None,
        "cache_adjusted_cost_usd": None, "valid_json": False, "schema_valid": False,
        "response_text": "", "response_id": "", "pricing_date": config["pricing"]["date"],
        "currency": config["pricing"]["currency"], "run_id": "",
    }
    try:
        raw = client.chat.completions.with_raw_response.create(
            model=config["model"],
            messages=[{"role": "user", "content": prompt.format(question=question["question"])}],
            max_completion_tokens=config["max_completion_tokens"],
            reasoning_effort=config["reasoning_effort"],
            service_tier="default",
        )
        row["http_status"] = raw.status_code
        response = raw.parse()
        row["model_name"] = response.model
        row["response_id"] = response.id
        if response.usage is not None:
            usage = response.usage
            cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
            reasoning = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
            row.update({
                "input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens, "cached_input_tokens": cached,
                "reasoning_tokens": reasoning,
            })
            prices = config["pricing"]
            row.update(request_cost(
                usage.prompt_tokens, usage.completion_tokens,
                prices["input_per_million"], prices["output_per_million"],
                cached, prices["cached_input_per_million"],
            ))
        if not response.choices:
            row["status"] = "empty_response"
        else:
            choice = response.choices[0]
            text = choice.message.content or ""
            row["finish_reason"] = choice.finish_reason
            row["response_text"] = text
            if getattr(choice.message, "refusal", None):
                row["status"] = "refused"
            elif choice.finish_reason != "stop":
                row["status"] = "incomplete"
            elif not text.strip():
                row["status"] = "empty_response"
            elif response.usage is None:
                row["status"] = "missing_usage"
            else:
                row["status"] = "success"
                row.update(validate_response(text))
    except Exception as error:
        row["error_type"] = type(error).__name__
        row["http_status"] = getattr(error, "status_code", row["http_status"])
    finally:
        row["latency_seconds"] = time.perf_counter() - timer
        row["end_time_utc"] = datetime.now(timezone.utc).isoformat()
    return row


def run_experiment(root: Path = ROOT, client: Any = None, overwrite: bool = False) -> dict[str, Any]:
    import mlflow
    import mlflow.genai
    from openai import OpenAI

    if (root / "latency_tokens_cost.csv").exists() and not overwrite:
        raise FileExistsError("Results exist. Archive them or explicitly use --overwrite (15 new paid calls).")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config["pricing"]["currency"] != "USD":
        raise ValueError("This implementation reports USD; configure USD published rates.")
    if client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY privately in your environment before running.")
        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"], base_url="https://api.openai.com/v1",
            timeout=config["timeout_seconds"], max_retries=0,
        )
    uri = tracking_uri(root)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    if mlflow.get_experiment_by_name(config["experiment_name"]) is None:
        mlflow.create_experiment(
            config["experiment_name"], artifact_location=(root / "runtime" / "artifacts").as_uri()
        )
    mlflow.set_experiment(config["experiment_name"])
    batch_id = str(uuid4())
    evidence = root / "outputs"
    evidence.mkdir(parents=True, exist_ok=True)
    prompts = {}
    for label, definition in PROMPTS.items():
        prompts[label] = mlflow.genai.register_prompt(
            name=config["prompt_name"], template=definition["template"],
            commit_message=definition["commit_message"],
            tags={"assignment_version": label, "batch_id": batch_id},
        )
    diffs = []
    for old, new in [("v1", "v2"), ("v2", "v3")]:
        diffs.extend(difflib.unified_diff(
            PROMPTS[old]["template"].splitlines(keepends=True),
            PROMPTS[new]["template"].splitlines(keepends=True),
            fromfile=old, tofile=new,
        ))
    (evidence / "prompt_diff.txt").write_text("\n".join(diffs), encoding="utf-8")
    write_json(evidence / "questions.json", QUESTIONS)
    write_json(evidence / "prompts.json", PROMPTS)
    schedule = [(label, question) for label in PROMPTS for question in QUESTIONS]
    random.Random(config["schedule_seed"]).shuffle(schedule)
    rows = []
    for order, (label, question) in enumerate(schedule, start=1):
        row = call_question(client, prompts[label], question, config)
        row.update({"prompt_version": label, "request_order": order, "batch_id": batch_id})
        rows.append(row)
        write_csv(root / "latency_tokens_cost.csv", rows)
        print(f"{order:02d}/15 {label} {question['question_id']}: {row['status']} ({row['latency_seconds']:.3f}s)")
    comparison = []
    for label in PROMPTS:
        version_rows = [row for row in rows if row["prompt_version"] == label]
        metrics = summarize_version(version_rows)
        with mlflow.start_run(run_name=f"prompt_{label}", tags={"batch_id": batch_id}) as run:
            for row in version_rows:
                row["run_id"] = run.info.run_id
            params = {
                "prompt_version": label, "registry_version": prompts[label].version,
                "prompt_uri": version_rows[0]["prompt_uri"], "model": config["model"],
                "api_version": config["api_version"], "pricing_date": config["pricing"]["date"],
                "currency": "USD", "max_completion_tokens": config["max_completion_tokens"],
                "reasoning_effort": config["reasoning_effort"], "sdk_retries": 0,
                "schedule_seed": config["schedule_seed"], "response_format": "prompt_only",
                "input_price_per_million": config["pricing"]["input_per_million"],
                "output_price_per_million": config["pricing"]["output_per_million"],
                "cached_input_price_per_million": config["pricing"]["cached_input_per_million"],
            }
            mlflow.log_params(params)
            mlflow.log_metrics({key: value for key, value in metrics.items() if value is not None})
            version_path = evidence / f"requests_{label}.csv"
            write_csv(version_path, version_rows)
            mlflow.log_artifact(str(version_path), "evaluation")
            mlflow.log_dict(PROMPTS[label], "evaluation/prompt.json")
            mlflow.log_dict(config, "evaluation/config.json")
            mlflow.log_dict(QUESTIONS, "evaluation/questions.json")
            mlflow.log_artifact(str(evidence / "prompt_diff.txt"), "evaluation")
            comparison.append({"prompt_version": label, "registry_version": prompts[label].version,
                               "run_id": run.info.run_id, **metrics, "recommended": False})
    candidate = select_candidate(comparison)
    for summary in comparison:
        summary["recommended"] = summary["prompt_version"] == candidate
    rows.sort(key=lambda row: (row["prompt_version"], row["question_id"]))
    write_csv(root / "latency_tokens_cost.csv", rows)
    write_csv(root / "prompt_evaluation_results.csv", comparison)
    summaries = {
        label: latency_summary([row for row in rows if row["prompt_version"] == label])
        for label in PROMPTS
    }
    write_json(evidence / "latency_summary.json", summaries)
    references = {question["question_id"]: question["reference"] for question in QUESTIONS}
    review = [{
        "batch_id": batch_id, "prompt_version": row["prompt_version"],
        "question_id": row["question_id"], "question": row["question"],
        "reference": references[row["question_id"]], "response_text": row["response_text"],
        "accuracy_0_to_2": "", "clarity_0_to_2": "", "relevance_0_to_2": "",
        "reviewer": "", "notes": "",
    } for row in rows]
    write_csv(root / "human_review.csv", review)
    manifest = {
        "batch_id": batch_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config, "tracking_uri": uri, "candidate": candidate,
        "production_promoted": False, "production_version": None,
        "runs": {row["prompt_version"]: row["run_id"] for row in comparison},
        "registry_versions": {label: str(prompt.version) for label, prompt in prompts.items()},
        "packages": {name: importlib.metadata.version(name) for name in ["mlflow", "openai"]},
        "results_sha256": hashlib.sha256((root / "latency_tokens_cost.csv").read_bytes()).hexdigest(),
    }
    write_json(evidence / "manifest.json", manifest)
    print("Results saved. Complete human_review.csv before promotion and final report generation.")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 15 live EduAssist API evaluations.")
    parser.add_argument("--overwrite", action="store_true", help="Replace outputs with 15 new paid calls.")
    args = parser.parse_args()
    run_experiment(overwrite=args.overwrite)


if __name__ == "__main__":
    main()