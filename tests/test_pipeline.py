import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from eduassist.pipeline import ROOT, call_question


class FakePrompt:
    name = "test_prompt"
    version = 3

    def format(self, **kwargs):
        return "Question: " + kwargs["question"]


class FakeClient:
    def __init__(self, text='{"answer":"Test fixture","confidence":"high","key_points":[]}'):
        self.text = text
        self.finish_reason = "stop"
        self.usage = SimpleNamespace(
            prompt_tokens=100, completion_tokens=50, total_tokens=150,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
        )
        self.chat = SimpleNamespace(completions=SimpleNamespace(with_raw_response=self))
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = SimpleNamespace(
            model=kwargs["model"], id="offline-test-fixture", usage=self.usage,
            choices=[SimpleNamespace(finish_reason=self.finish_reason,
                                     message=SimpleNamespace(content=self.text, refusal=None))],
        )
        return SimpleNamespace(status_code=200, parse=lambda: response)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "config.json").read_text())
        self.question = {"question_id": "Q1", "question": "Test?"}

    def test_success_usage_cost_and_api_settings(self):
        client = FakeClient()
        row = call_question(client, FakePrompt(), self.question, self.config)
        self.assertEqual(row["status"], "success")
        self.assertEqual(row["total_tokens"], 150)
        self.assertAlmostEqual(row["total_cost_usd"], 0.0003)
        self.assertTrue(row["schema_valid"])
        self.assertGreater(row["latency_seconds"], 0)
        self.assertNotIn("response_format", client.calls[0])
        self.assertEqual(client.calls[0]["service_tier"], "default")

    def test_truncated_json_fails_even_if_parseable(self):
        client = FakeClient()
        client.finish_reason = "length"
        row = call_question(client, FakePrompt(), self.question, self.config)
        self.assertEqual(row["status"], "incomplete")
        self.assertFalse(row["valid_json"])
        self.assertIsNotNone(row["total_cost_usd"])

    def test_missing_usage_not_zero_cost(self):
        client = FakeClient()
        client.usage = None
        row = call_question(client, FakePrompt(), self.question, self.config)
        self.assertEqual(row["status"], "missing_usage")
        self.assertIsNone(row["total_cost_usd"])

    def test_api_error_is_sanitized(self):
        class FailingClient(FakeClient):
            def create(self, **kwargs):
                raise RuntimeError("Secret must not appear in evidence")
        row = call_question(FailingClient(), FakePrompt(), self.question, self.config)
        self.assertEqual(row["error_type"], "RuntimeError")
        self.assertNotIn("Secret", json.dumps(row))


if __name__ == "__main__":
    unittest.main()