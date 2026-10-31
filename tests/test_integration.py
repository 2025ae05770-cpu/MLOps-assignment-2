import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from eduassist.package import SCREENSHOTS, build_submission
from eduassist.pipeline import ROOT, run_experiment, write_csv
from eduassist.report import finalize, read_csv
from test_pipeline import FakeClient


@contextlib.contextmanager
def temporary_mlflow_root():
    from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore as RegistryStore
    from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore as TrackingStore

    with tempfile.TemporaryDirectory() as temporary:
        try:
            yield Path(temporary)
        finally:
            for store in (TrackingStore, RegistryStore):
                for uri, engine in list(store._engine_map.items()):
                    if Path(temporary).as_posix() in uri:
                        engine.dispose()
                        del store._engine_map[uri]


class IntegrationTests(unittest.TestCase):
    def test_notebook_structure_syntax_and_safe_flags(self):
        import ast
        import nbformat

        notebook = nbformat.read(ROOT / "EduAssist_LLMOps_Assignment2.ipynb", as_version=4)
        nbformat.validate(notebook)
        flags = {}
        for cell in notebook.cells:
            if cell.cell_type == "code":
                tree = ast.parse(cell.source)
                for statement in tree.body:
                    if isinstance(statement, ast.Assign):
                        for target in statement.targets:
                            if isinstance(target, ast.Name) and target.id in {"RUN_LIVE", "PROMOTE_AFTER_REVIEW"}:
                                flags[target.id] = ast.literal_eval(statement.value)
        self.assertEqual(flags, {"RUN_LIVE": False, "PROMOTE_AFTER_REVIEW": False})

    def test_registry_metrics_quality_gate_report_and_alias(self):
        import mlflow
        import mlflow.genai
        import nbformat

        old_tracking = mlflow.get_tracking_uri()
        old_registry = mlflow.get_registry_uri()
        try:
            with temporary_mlflow_root() as root:
                (root / "config.json").write_text((ROOT / "config.json").read_text())
                environment = {key: value for key, value in os.environ.items()
                               if key not in {"MLFLOW_TRACKING_URI", "MLFLOW_REGISTRY_URI"}}
                with patch.dict(os.environ, environment, clear=True), contextlib.redirect_stdout(io.StringIO()):
                    client = FakeClient()
                    manifest = run_experiment(root=root, client=client)
                    self.assertEqual(len(client.calls), 15)
                    self.assertEqual(manifest["registry_versions"], {"v1": "1", "v2": "2", "v3": "3"})
                    comparison = read_csv(root / "prompt_evaluation_results.csv")
                    self.assertEqual(len(comparison), 3)
                    for row in comparison:
                        run = mlflow.get_run(row["run_id"])
                        self.assertEqual(run.data.metrics["valid_json"], 1.0)
                        self.assertIn("avg_cost_per_request", run.data.metrics)
                    with self.assertRaises(ValueError):
                        finalize(root=root, promote=True)
                    reviews = read_csv(root / "human_review.csv")
                    for review in reviews:
                        review.update({"accuracy_0_to_2": "2", "clarity_0_to_2": "2",
                                       "relevance_0_to_2": "2", "reviewer": "OFFLINE TEST ONLY",
                                       "notes": "Synthetic fixture; not academic or live API evidence."})
                    write_csv(root / "human_review.csv", reviews)
                    result = finalize(root=root, promote=True)
                    self.assertTrue(result["production_promoted"])
                    alias = mlflow.genai.load_prompt("prompts:/eduassist_academic_qa@production")
                    self.assertEqual(str(alias.version), result["production_version"])
                    self.assertTrue((root / "REPORT.pdf").read_bytes().startswith(b"%PDF"))
                    checks = json.loads((root / "outputs" / "report_checks.json").read_text())
                    self.assertTrue(500 <= checks["body_word_count"] <= 700)
                    self.assertGreaterEqual(checks["promotion_justification_word_count"], 150)
                    with self.assertRaises(FileNotFoundError):
                        build_submission(root=root)
                    for name in ["README.md", "requirements.txt", "EduAssist_LLMOps_Assignment2.ipynb"]:
                        shutil.copyfile(ROOT / name, root / name)
                    notebook_path = root / "EduAssist_LLMOps_Assignment2.ipynb"
                    notebook = nbformat.read(notebook_path, as_version=4)
                    first_code = next(cell for cell in notebook.cells if cell.cell_type == "code")
                    first_code.outputs = [nbformat.v4.new_output(
                        "stream", name="stdout", text="OFFLINE TEST BATCH: " + result["batch_id"]
                    )]
                    nbformat.write(notebook, notebook_path)
                    (root / "EduAssist_LLMOps_Assignment2.html").write_text("<p>Offline packaging fixture only.</p>")
                    (root / "screenshots").mkdir()
                    for name in SCREENSHOTS:
                        (root / "screenshots" / name).write_bytes(b"\x89PNG\r\n\x1a\nTEST ONLY")
                    (root / ".env").write_text("TEST_ONLY_DO_NOT_PACKAGE")
                    archive_path = build_submission(root=root)
                    with zipfile.ZipFile(archive_path) as archive:
                        names = archive.namelist()
                        self.assertIn("assignment2/REPORT.pdf", names)
                        self.assertIn("assignment2/screenshots/production_alias.png", names)
                        self.assertFalse(any("runtime/" in name or name.endswith(".env") for name in names))
                    with self.assertRaises(FileExistsError):
                        run_experiment(root=root, client=client)
                    with (root / "latency_tokens_cost.csv").open("a") as handle:
                        handle.write("tampered")
                    with self.assertRaises(ValueError):
                        finalize(root=root)
        finally:
            mlflow.set_tracking_uri(old_tracking)
            mlflow.set_registry_uri(old_registry)


if __name__ == "__main__":
    unittest.main()