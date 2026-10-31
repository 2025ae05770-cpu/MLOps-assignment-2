"""Human quality gate, production alias, and measured 500-700 word PDF report."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any
from xml.sax.saxutils import escape

from eduassist.evaluation import select_candidate, summarize_version
from eduassist.pipeline import ROOT, write_csv, write_json
from eduassist.prompts import PROMPTS, QUESTIONS


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_evidence(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads((root / "outputs" / "manifest.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((root / "latency_tokens_cost.csv").read_bytes()).hexdigest()
    if digest != manifest["results_sha256"]:
        raise ValueError("Request evidence changed after evaluation; rerun rather than editing measured results.")
    rows: list[dict[str, Any]] = list(read_csv(root / "latency_tokens_cost.csv"))
    expected = {(label, question["question_id"]) for label in PROMPTS for question in QUESTIONS}
    keys = [(row["prompt_version"], row["question_id"]) for row in rows]
    if len(rows) != 15 or set(keys) != expected:
        raise ValueError("Expected exactly three versions evaluated on the same five questions.")
    numeric = ["latency_seconds", "input_tokens", "output_tokens", "total_cost_usd"]
    for row in rows:
        if row["batch_id"] != manifest["batch_id"]:
            raise ValueError("Mixed evaluation batches.")
        for field in numeric:
            row[field] = float(row[field]) if row[field] else None
        for field in ["valid_json", "schema_valid"]:
            row[field] = row[field] == "True"
    reviews = read_csv(root / "human_review.csv")
    review_keys = [(row["prompt_version"], row["question_id"]) for row in reviews]
    if len(reviews) != 15 or set(review_keys) != expected:
        raise ValueError("Human review must contain all 15 unique request identifiers.")
    lookup = {(row["prompt_version"], row["question_id"]): row for row in rows}
    for review in reviews:
        original = lookup[(review["prompt_version"], review["question_id"])]
        if review["batch_id"] != manifest["batch_id"] or review["response_text"] != original["response_text"]:
            raise ValueError("Human review does not match the recorded responses and batch.")
        if not review["reviewer"].strip() or not review["notes"].strip():
            raise ValueError("Fill reviewer, notes, and all three 0-2 quality scores for every response.")
        for field in ["accuracy_0_to_2", "clarity_0_to_2", "relevance_0_to_2"]:
            if review[field] not in {"0", "1", "2"}:
                raise ValueError(f"Invalid or missing {field}; use an integer 0, 1, or 2.")
            review[field] = int(review[field])
        review["quality_score"] = sum(review[field] for field in [
            "accuracy_0_to_2", "clarity_0_to_2", "relevance_0_to_2"
        ])
        review["quality_pass"] = (
            review["accuracy_0_to_2"] == 2 and review["relevance_0_to_2"] == 2
            and review["clarity_0_to_2"] >= 1
        )
    return manifest, rows, reviews


def report_sections(
    manifest: dict[str, Any], rows: list[dict[str, Any]], comparison: list[dict[str, Any]],
    candidate: str, promoted: bool,
) -> list[tuple[str, str]]:
    config = manifest["config"]
    selected = next(row for row in comparison if row["prompt_version"] == candidate)
    baseline = [row for row in rows if row["prompt_version"] == "v1"]
    slowest = max(baseline, key=lambda row: row["latency_seconds"])
    slowest_tokens = f"{slowest['output_tokens']:.0f}" if slowest["output_tokens"] is not None else "unavailable"
    baseline_mean = statistics.mean(row["latency_seconds"] for row in baseline)
    baseline_median = statistics.median(row["latency_seconds"] for row in baseline)
    baseline_std = statistics.stdev(row["latency_seconds"] for row in baseline)
    findings = "; ".join(
        f"{row['prompt_version']}: JSON {row['valid_json']:.0%}, schema {row['schema_valid_rate']:.0%}, "
        f"quality {row['avg_quality_score']:.1f}/6"
        for row in comparison
    )
    promotion_state = (
        f"The production alias was set and verified against registry version {manifest['production_version']}."
        if promoted else "Promotion is withheld; this document is a draft, not final submission evidence."
    )
    return [
        ("Purpose and LLMOps", (
            "EduAssist AI requires an academic question-answering assistant with a reliable JSON interface. "
            "Traditional MLOps manages data preparation, feature engineering, model training, deployment, and monitoring. "
            "LLMOps extends these controls to independently changing prompts, retrieval context, provider API models, "
            "guardrails, and generated outputs. Token consumption and per-request charges become operational metrics. "
            "Unlike a conventional deterministic interface, an LLM can produce different explanations or malformed "
            "output for similar requests. Versioned prompts, reproducible settings, recorded responses, and human "
            "review therefore complement automated evaluation rather than replacing it."
        )),
        ("Experimental Method", (
            f"I evaluated the assignment's basic v1, structured v2, and JSON-output v3 templates on the same five "
            f"questions covering overfitting, learning paradigms, gradient descent, classification metrics, and LLMOps. "
            f"The model was {config['model']}, accessed through {config['api_version']}. "
            f"All fifteen calls used reasoning effort {config['reasoning_effort']} and a maximum completion budget of "
            f"{config['max_completion_tokens']} tokens. A fixed-seed shuffled schedule reduced systematic ordering bias. "
            "No JSON response mode, output repair, or hidden SDK retry was used. The API key came only from an "
            "environment variable. UTC timestamps identify requests, while a monotonic timer measures complete "
            "non-streaming response latency, including network and SDK processing."
        )),
        ("Latency and Cost Findings", (
            f"For the five v1 requests, mean latency was {baseline_mean:.3f} seconds, median {baseline_median:.3f}, "
            f"minimum {min(row['latency_seconds'] for row in baseline):.3f}, maximum "
            f"{slowest['latency_seconds']:.3f}, and sample standard deviation {baseline_std:.3f}. "
            f"The slowest baseline request was {slowest['question_id']}, with {slowest_tokens} output tokens. "
            "Generation length, provider queuing, and network variability are plausible explanations, but this "
            "small observational sample cannot establish causation. Provider-reported input and output tokens were "
            f"priced at USD {config['pricing']['input_per_million']:.3f} and "
            f"{config['pricing']['output_per_million']:.3f} per million respectively, checked on "
            f"{config['pricing']['date']}. Each request cost equals input tokens times the input rate plus output "
            "tokens times the output rate, divided by one million. Monthly projection multiplies mean request "
            "cost by 10,000; it excludes taxes, infrastructure, retries, and workload changes. A separate "
            "cache-adjusted estimate avoids claiming the assignment's uncached formula is an exact invoice."
        )),
        ("Evaluation and Promotion Justification", (
            f"The measured comparison was {findings}. I recommend {candidate} because it passed all five completed "
            "requests, the exact output schema, and the human quality gate. Syntax validity alone accepts objects "
            "or arrays that may not meet EduAssist's contract, so the additional schema check requires precisely "
            "answer, confidence, and key_points, with appropriate types and an allowed confidence value. Human "
            "review checked each response against reference concepts for accuracy, clarity, and relevance; model "
            "self-reported confidence was not treated as a calibrated probability or correctness score. "
            f"The selected version achieved mean human quality {selected['avg_quality_score']:.1f}/6, JSON validity "
            f"{selected['valid_json']:.0%}, and schema compliance {selected['schema_valid_rate']:.0%}. Its mean latency "
            f"was {selected['avg_latency_seconds']:.3f} seconds, with {selected['avg_input_tokens']:.1f} input "
            f"and {selected['avg_output_tokens']:.1f} output tokens. Mean request cost was USD "
            f"{selected['avg_cost_per_request']:.8f}, projecting USD {selected['projected_monthly_cost_usd']:.4f} "
            "monthly. Among quality-qualified, schema-compliant versions, lower cost was prioritized, with latency "
            "as a tie-breaker. Cheaper prose is unsuitable when downstream consumers require strict JSON. "
            f"{promotion_state} Immutable prompt versions, commit messages, run identifiers, prompt diffs, and "
            "the alias provide traceability and support rollback without changing application code."
        )),
        ("Limitations and Optimization", (
            "Five questions per version provide assignment-scale evidence, not a production SLA or statistically "
            "robust ranking. A perfect observed pass rate does not guarantee future validity or academic accuracy. "
            "Before deployment, expand question coverage, repeat evaluations, assess unsafe and ambiguous inputs, "
            "and monitor latency distributions and schema failures. My optimization recommendation is to shorten "
            "answers and avoid redundant context while preserving reference-tested quality. Output tokens carry "
            "a higher unit price, so this can reduce cost and generation time. Any optimized prompt should become "
            "a new registry version and pass the same quality and JSON gates before replacing production."
        )),
    ]


def finalize(root: Path = ROOT, promote: bool = False) -> dict[str, Any]:
    import mlflow
    import mlflow.genai
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    manifest, rows, reviews = load_evidence(root)
    mlflow.set_tracking_uri(manifest["tracking_uri"])
    mlflow.set_registry_uri(manifest["tracking_uri"])
    comparison = []
    for label in PROMPTS:
        subset = [row for row in rows if row["prompt_version"] == label]
        quality = [row for row in reviews if row["prompt_version"] == label]
        metrics = summarize_version(subset)
        metrics.update({
            "avg_quality_score": statistics.mean(row["quality_score"] for row in quality),
            "quality_pass_rate": sum(row["quality_pass"] for row in quality) / 5,
        })
        comparison.append({"prompt_version": label, "registry_version": manifest["registry_versions"][label],
                           "run_id": manifest["runs"][label], **metrics, "recommended": False})
    candidate = select_candidate([row for row in comparison if row["quality_pass_rate"] == 1.0])
    if candidate is None:
        raise ValueError("No prompt passes all completion, schema and human quality gates. Do not promote.")
    for summary in comparison:
        summary["recommended"] = summary["prompt_version"] == candidate
        with mlflow.start_run(run_id=summary["run_id"]):
            mlflow.log_metrics({"avg_quality_score": summary["avg_quality_score"],
                                "quality_pass_rate": summary["quality_pass_rate"]})
            mlflow.log_artifact(str(root / "human_review.csv"), "evaluation")
            mlflow.set_tag("recommended_after_review", str(summary["recommended"]).lower())
    manifest["candidate"] = candidate
    if promote:
        name = manifest["config"]["prompt_name"]
        version = manifest["registry_versions"][candidate]
        mlflow.genai.set_prompt_alias(name, alias="production", version=version)
        production = mlflow.genai.load_prompt(f"prompts:/{name}@production", cache_ttl_seconds=0)
        if str(production.version) != str(version):
            raise RuntimeError("Production alias verification failed.")
        manifest["production_promoted"] = True
        manifest["production_version"] = str(version)
        with mlflow.start_run(run_id=manifest["runs"][candidate]):
            mlflow.set_tag("production_alias", "production")
            mlflow.set_tag("production_registry_version", str(version))
    elif manifest["production_promoted"]:
        name = manifest["config"]["prompt_name"]
        production = mlflow.genai.load_prompt(f"prompts:/{name}@production", cache_ttl_seconds=0)
        if str(production.version) != manifest["registry_versions"][candidate]:
            raise ValueError("Production alias differs from reviewed candidate; explicitly promote again.")
    write_csv(root / "prompt_evaluation_results.csv", comparison)
    write_json(root / "outputs" / "manifest.json", manifest)
    sections = report_sections(manifest, rows, comparison, candidate, manifest["production_promoted"])
    count = sum(len(text.split()) for _, text in sections)
    if not 500 <= count <= 700:
        raise ValueError(f"Report contains {count} words, outside the required 500-700 range.")
    promotion_words = len(dict(sections)["Evaluation and Promotion Justification"].split())
    if promotion_words < 150:
        raise ValueError("Promotion justification must contain at least 150 words.")
    markdown = "# EduAssist AI: LLMOps Assignment 2\n\n" + "\n\n".join(
        f"## {heading}\n\n{text}" for heading, text in sections
    )
    (root / "REPORT.md").write_text(markdown + "\n", encoding="utf-8")
    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize = 10
    styles["BodyText"].leading = 14
    styles["Heading2"].textColor = colors.HexColor("#176b56")
    story = [Paragraph("EduAssist AI: LLMOps Assignment 2", styles["Title"])]
    for heading, text in sections:
        story.extend([Paragraph(escape(heading), styles["Heading2"]),
                      Paragraph(escape(text), styles["BodyText"]), Spacer(1, 8)])
    document = SimpleDocTemplate(str(root / "REPORT.pdf"), pagesize=A4,
                                 rightMargin=42, leftMargin=42, topMargin=36, bottomMargin=36)
    document.build(story)
    write_json(root / "outputs" / "report_checks.json", {
        "body_word_count": count, "promotion_justification_word_count": promotion_words,
        "production_promoted": manifest["production_promoted"],
    })
    with mlflow.start_run(run_id=manifest["runs"][candidate]):
        mlflow.log_artifact(str(root / "REPORT.pdf"), "submission")
        mlflow.log_artifact(str(root / "prompt_evaluation_results.csv"), "evaluation")
    print(f"Report: {count} words; promotion justification: {promotion_words} words. Candidate: {candidate}.")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report after completing human quality review.")
    parser.add_argument("--promote", action="store_true", help="Set and verify the production prompt alias.")
    args = parser.parse_args()
    finalize(promote=args.promote)


if __name__ == "__main__":
    main()