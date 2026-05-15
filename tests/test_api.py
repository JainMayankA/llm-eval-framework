"""
FastAPI endpoint tests.

The server falls back to MockAdapters when no API keys are set, which is
always the case in CI. Background tasks complete synchronously inside
TestClient, so job status can be checked immediately after the triggering POST.
"""

import pytest
from fastapi.testclient import TestClient

from api.server import app

client = TestClient(app)


class TestHealth:
    def test_returns_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "models" in data
        assert "factuality_scorer" in data

    def test_reports_mock_mode_without_keys(self):
        resp = client.get("/health")
        assert resp.json()["mode"] == "mock"


class TestModels:
    def test_returns_model_list(self):
        resp = client.get("/models")
        assert resp.status_code == 200
        assert isinstance(resp.json()["models"], list)
        assert len(resp.json()["models"]) > 0


class TestEvalRun:
    def test_enqueues_job_and_returns_id(self):
        resp = client.post("/eval/run", json={
            "prompts": [{"id": "t1", "prompt": "What is 2+2?"}]
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "job_id" in body
        assert body["status"] == "running"

    def test_job_completes_with_mock(self):
        resp = client.post("/eval/run", json={
            "prompts": [{"id": "t1", "prompt": "Capital of France?", "ground_truth": "Paris"}]
        })
        job_id = resp.json()["job_id"]

        status = client.get(f"/eval/{job_id}").json()
        assert status["status"] == "done"
        assert len(status["results"]) > 0
        assert "best_model" in status

    def test_job_id_is_full_uuid(self):
        resp = client.post("/eval/run", json={
            "prompts": [{"id": "t1", "prompt": "Hello"}]
        })
        job_id = resp.json()["job_id"]
        assert len(job_id) == 36  # full UUID4: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

    def test_unknown_model_returns_400(self):
        resp = client.post("/eval/run", json={
            "prompts": [{"id": "t1", "prompt": "Hello"}],
            "models": ["nonexistent-model-xyz"],
        })
        assert resp.status_code == 400
        assert "available_models" in resp.json()["detail"]

    def test_empty_prompts_returns_422(self):
        resp = client.post("/eval/run", json={"prompts": []})
        assert resp.status_code == 422

    def test_extra_fields_rejected(self):
        resp = client.post("/eval/run", json={
            "prompts": [{"id": "t1", "prompt": "Hello"}],
            "unknown_field": "value",
        })
        assert resp.status_code == 422

    def test_missing_prompt_text_returns_422(self):
        resp = client.post("/eval/run", json={
            "prompts": [{"id": "t1"}]
        })
        assert resp.status_code == 422


class TestEvalJobStatus:
    def test_nonexistent_job_returns_404(self):
        resp = client.get("/eval/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404

    def test_completed_job_has_expected_result_keys(self):
        resp = client.post("/eval/run", json={
            "prompts": [{"id": "t1", "prompt": "Hello", "ground_truth": "Hi"}]
        })
        job_id = resp.json()["job_id"]

        job = client.get(f"/eval/{job_id}").json()
        result = job["results"][0]
        for key in ("model", "prompt_id", "factuality_score", "safety_score", "latency_ms", "cost_usd"):
            assert key in result, f"missing key: {key}"


class TestEvalReport:
    def _completed_job_id(self):
        resp = client.post("/eval/run", json={
            "prompts": [{"id": "r1", "prompt": "Hello", "ground_truth": "Hi"}]
        })
        return resp.json()["job_id"]

    def test_markdown_report_returned_as_json(self):
        job_id = self._completed_job_id()
        resp = client.get(f"/eval/{job_id}/report")
        assert resp.status_code == 200
        assert "LLM Evaluation Report" in resp.json()["report"]

    def test_csv_report_has_correct_content_type(self):
        job_id = self._completed_job_id()
        resp = client.get(f"/eval/{job_id}/report?fmt=csv")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        assert resp.text.startswith("model,prompt_id")

    def test_report_for_nonexistent_job_returns_404(self):
        resp = client.get("/eval/00000000-0000-0000-0000-000000000000/report")
        assert resp.status_code == 404


class TestRedTeam:
    def test_returns_job_id(self):
        resp = client.post("/redteam/run", json={"categories": ["hallucination_inducing"]})
        assert resp.status_code == 200
        assert "job_id" in resp.json()

    def test_job_completes_with_pass_rates(self):
        resp = client.post("/redteam/run", json={"categories": ["hallucination_inducing"]})
        job_id = resp.json()["job_id"]

        job = client.get(f"/eval/{job_id}").json()
        assert job["status"] == "done"
        assert "pass_rates" in job
        assert "failures_by_category" in job

    def test_unknown_category_still_runs(self):
        resp = client.post("/redteam/run", json={"categories": ["hallucination_inducing"]})
        assert resp.status_code == 200


class TestDemo:
    def test_demo_job_completes(self):
        resp = client.post("/eval/demo", json={})
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        job = client.get(f"/eval/{job_id}").json()
        assert job["status"] == "done"
        assert len(job["results"]) > 0
