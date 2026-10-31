"""Validate required evidence and build a secret-free, explicit-file submission ZIP."""

import json
from pathlib import Path
import zipfile

from eduassist.pipeline import ROOT
from eduassist.report import load_evidence

SCREENSHOTS = ["experiment.png", "prompt_registry.png", "metrics.png", "prompt_diff.png", "production_alias.png"]


def build_submission(root: Path = ROOT) -> Path:
    import mlflow
    import mlflow.genai
    import nbformat

    required = [
        "EduAssist_LLMOps_Assignment2.ipynb", "EduAssist_LLMOps_Assignment2.html",
        "requirements.txt", "README.md", "config.json", "latency_tokens_cost.csv",
        "prompt_evaluation_results.csv", "human_review.csv", "REPORT.pdf", "REPORT.md",
        "outputs/manifest.json", "outputs/latency_summary.json", "outputs/report_checks.json",
        "outputs/prompt_diff.txt", "outputs/questions.json", "outputs/prompts.json",
        "outputs/requests_v1.csv", "outputs/requests_v2.csv", "outputs/requests_v3.csv",
        *[f"screenshots/{name}" for name in SCREENSHOTS],
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("Submission incomplete:\n" + "\n".join(missing))
    manifest, _, _ = load_evidence(root)
    if not manifest["production_promoted"]:
        raise ValueError("Production promotion has not been completed.")
    mlflow.set_tracking_uri(manifest["tracking_uri"])
    mlflow.set_registry_uri(manifest["tracking_uri"])
    production = mlflow.genai.load_prompt(
        f"prompts:/{manifest['config']['prompt_name']}@production", cache_ttl_seconds=0
    )
    if str(production.version) != manifest["production_version"]:
        raise ValueError("Current production alias does not match the submitted report.")
    checks = json.loads((root / "outputs" / "report_checks.json").read_text(encoding="utf-8"))
    if not 500 <= checks["body_word_count"] <= 700 or checks["promotion_justification_word_count"] < 150:
        raise ValueError("Report word counts do not meet the assignment requirements.")
    notebook = nbformat.read(root / "EduAssist_LLMOps_Assignment2.ipynb", as_version=4)
    nbformat.validate(notebook)
    output_text = json.dumps([cell.get("outputs", []) for cell in notebook.cells])
    if manifest["batch_id"] not in output_text:
        raise ValueError("Execute the notebook after this batch so its saved outputs contain the real results.")
    for name in SCREENSHOTS:
        if not (root / "screenshots" / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"screenshots/{name} must be an actual PNG screenshot.")
    required.extend(path.relative_to(root).as_posix() for path in (root / "eduassist").glob("*.py"))
    required.extend(path.relative_to(root).as_posix() for path in (root / "tests").glob("*.py"))
    target = root / "submission.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(required)):
            archive.write(root / name, arcname=f"assignment2/{name}")
    print(f"Submission ready: {target}")
    return target


if __name__ == "__main__":
    build_submission()