import unittest

from eduassist.evaluation import (
    latency_summary, request_cost, select_candidate, summarize_version, validate_response,
)


class EvaluationTests(unittest.TestCase):
    def test_cost_and_monthly_projection(self):
        cost = request_cost(1000, 500, 0.15, 0.60, 200, 0.075)
        self.assertAlmostEqual(cost["total_cost_usd"], 0.00045)
        self.assertAlmostEqual(cost["total_cost_usd"] * 10_000, 4.5)
        self.assertAlmostEqual(cost["cache_adjusted_cost_usd"], 0.000435)

    def test_invalid_cost_inputs(self):
        for args in [(-1, 2, 0.15, 0.6), (1, 2, float("nan"), 0.6), (1, 2, 0.15, 0.6, 2)]:
            with self.assertRaises(ValueError):
                request_cost(*args)

    def test_strict_json_and_schema_are_distinct(self):
        valid = '{"answer":"A","confidence":"high","key_points":["B"]}'
        self.assertEqual(validate_response(valid), {"valid_json": True, "schema_valid": True})
        self.assertEqual(validate_response('[]'), {"valid_json": True, "schema_valid": False})
        self.assertFalse(validate_response(valid.replace('"high"', '[]'))["schema_valid"])
        for invalid in ["```json\n" + valid + "\n```", "NaN", '{"a":1,"a":2}', "text"]:
            self.assertFalse(validate_response(invalid)["valid_json"])

    def test_latency_uses_sample_standard_deviation(self):
        summary = latency_summary([{"latency_seconds": value} for value in [1, 2, 3, 4, 5]])
        self.assertEqual(summary["mean_seconds"], 3)
        self.assertEqual(summary["median_seconds"], 3)
        self.assertAlmostEqual(summary["stddev_seconds"], 1.5811388300841898)

    def test_failures_are_not_free_requests(self):
        rows = [{"status": "error", "latency_seconds": 2, "valid_json": False,
                 "schema_valid": False, "total_cost_usd": None} for _ in range(5)]
        summary = summarize_version(rows)
        self.assertEqual(summary["valid_json"], 0)
        self.assertIsNone(summary["avg_cost_per_request"])
        self.assertIsNone(select_candidate([dict(summary, prompt_version="v3")]))

    def test_promotion_requires_complete_schema_compliance(self):
        base = {"request_count": 5, "success_rate": 1.0, "schema_valid_rate": 1.0,
                "avg_cost_per_request": 0.001, "avg_latency_seconds": 1}
        comparison = [dict(base, prompt_version="v1", schema_valid_rate=0),
                      dict(base, prompt_version="v3")]
        self.assertEqual(select_candidate(comparison), "v3")


if __name__ == "__main__":
    unittest.main()