import logging
import os
import json
import threading
import time
from pathlib import Path
from typing import Optional, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Response
from pydantic import BaseModel, ConfigDict, Field
import uuid
from dotenv import load_dotenv

from evaluator.engine import EvalEngine, EvalSuiteError
from evaluator.models import MockAdapter, OllamaAdapter, OpenAIAdapter
from evaluator.scorers import LLMJudgeFactualityScorer, OpenAIModerationSafetyScorer
from evaluator.reporter import to_markdown, to_csv, best_model
from agents.red_team import RedTeamAgent

load_dotenv()

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="LLM Eval Framework", version="1.0.0")
ROOT_DIR = Path(__file__).resolve().parents[1]
DEMO_PROMPTS_PATH = ROOT_DIR / "benchmarks" / "demo_prompts.json"

def _configured_models():
    # OpenAI key wins, then Ollama models, then mocks for local dev/tests.
    force_mock = os.getenv("EVAL_USE_MOCK", "").lower() in {"1", "true", "yes"}
    openai_key = os.getenv("OPENAI_API_KEY")

    if openai_key and not force_mock:
        model_names = os.getenv("EVAL_OPENAI_MODELS", "gpt-4o-mini")
        return [
            OpenAIAdapter(model.strip())
            for model in model_names.split(",")
            if model.strip()
        ]

    ollama_models = os.getenv("EVAL_OLLAMA_MODELS", "").strip()
    if ollama_models and not force_mock:
        return [
            OllamaAdapter(model.strip())
            for model in ollama_models.split(",")
            if model.strip()
        ]

    return [
        MockAdapter("mock/gpt-4o-mini"),
        MockAdapter(
            "mock/claude-3-haiku",
            response="The capital of France is Paris, a major European city.",
        ),
    ]


def _configured_factuality_scorer():
    scorer = os.getenv("EVAL_FACTUALITY_SCORER", "heuristic").lower()
    if scorer not in {"llm-judge", "judge"}:
        return None
    if not os.getenv("OPENAI_API_KEY"):
        logging.warning("EVAL_FACTUALITY_SCORER=llm-judge ignored: OPENAI_API_KEY is not set")
        return None

    judge_model = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o-mini")
    judge = OpenAIAdapter(
        judge_model,
        max_tokens=int(os.getenv("EVAL_JUDGE_MAX_TOKENS", "256")),
    )
    return LLMJudgeFactualityScorer(judge)


def _configured_safety_scorer():
    scorer = os.getenv("EVAL_SAFETY_SCORER", "heuristic").lower()
    if scorer not in {"openai-moderation", "moderation"}:
        return None
    if not os.getenv("OPENAI_API_KEY"):
        logging.warning("EVAL_SAFETY_SCORER=openai-moderation ignored: OPENAI_API_KEY is not set")
        return None

    return OpenAIModerationSafetyScorer(
        model=os.getenv("EVAL_MODERATION_MODEL", "omni-moderation-latest")
    )


_engine = EvalEngine(
    models=_configured_models(),
    factuality_scorer=_configured_factuality_scorer(),
    safety_scorer=_configured_safety_scorer(),
)
_red_team = RedTeamAgent(_engine)

class _JobStore:
    """Thread-safe in-process job store with TTL eviction and a size cap.

    Not safe across multiple uvicorn workers — use Redis or a DB for that.
    """

    MAX_JOBS = 500
    TTL_SECONDS = 3600

    def __init__(self) -> None:
        self._store: dict[str, dict] = {}
        self._lock = threading.Lock()

    def create(self, job_id: str, data: dict) -> None:
        with self._lock:
            self._evict()
            if len(self._store) >= self.MAX_JOBS:
                oldest = min(self._store, key=lambda k: self._store[k]["_ts"])
                del self._store[oldest]
            self._store[job_id] = {**data, "_ts": time.monotonic()}

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            entry = self._store.get(job_id)
            return {k: v for k, v in entry.items() if k != "_ts"} if entry else None

    def update(self, job_id: str, patch: dict) -> None:
        with self._lock:
            if job_id in self._store:
                self._store[job_id].update(patch)

    def _evict(self) -> None:
        cutoff = time.monotonic() - self.TTL_SECONDS
        for k in [k for k, v in self._store.items() if v["_ts"] < cutoff]:
            del self._store[k]


_jobs = _JobStore()


def _normalize_model_names(model_names: Optional[list[str]]) -> Optional[list[str]]:
    if not model_names:
        return None

    resolved = []
    available = set(_engine.models.keys())
    for name in model_names:
        if name in available:
            resolved.append(name)
            continue

        openai_name = f"openai/{name}"
        if openai_name in available:
            resolved.append(openai_name)
            continue

        mock_name = f"mock/{name}"
        if mock_name in available:
            resolved.append(mock_name)
            continue

        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Unknown model: {name}",
                "available_models": sorted(available),
            },
        )

    return resolved


class PromptInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    ground_truth: Optional[str] = None
    category: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompts: list[PromptInput] = Field(..., min_length=1)
    models: Optional[list[str]] = None


class RedTeamRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: Optional[list[str]] = None
    categories: Optional[list[str]] = None


class DemoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: Optional[list[str]] = None


def _prompt_dicts(prompts: list[PromptInput]) -> list[dict]:
    return [p.model_dump(exclude_none=True) for p in prompts]


def _start_eval_job(
    prompts: list[dict],
    model_names: Optional[list[str]],
    background_tasks: BackgroundTasks,
) -> dict:
    job_id = str(uuid.uuid4())
    _jobs.create(job_id, {"status": "running", "results": None})

    def _run():
        try:
            results = _engine.run_suite(prompts, model_names)
            _jobs.update(job_id, {
                "status": "done",
                "results": [r.__dict__ for r in results],
                "best_model": best_model(results),
            })
        except EvalSuiteError as exc:
            logging.exception("Eval job %s completed with failed cases", job_id)
            _jobs.update(job_id, {
                "status": "failed",
                "results": [r.__dict__ for r in exc.partial_results],
                "failures": [f.__dict__ for f in exc.failures],
                "error": str(exc),
            })
        except Exception as exc:
            logging.exception("Eval job %s failed", job_id)
            _jobs.update(job_id, {"status": "failed", "error": str(exc)})

    background_tasks.add_task(_run)
    return {"job_id": job_id, "status": "running"}


@app.get("/health")
def health():
    using_mock = all(isinstance(m, MockAdapter) for m in _engine.models.values())
    return {
        "status": "ok",
        "mode": "mock" if using_mock else "openai",
        "factuality_scorer": _engine.factuality.__class__.__name__,
        "safety_scorer": _engine.safety.__class__.__name__,
        "models": list(_engine.models.keys()),
    }


@app.post("/eval/run")
def run_eval(req: EvalRequest, background_tasks: BackgroundTasks):
    model_names = _normalize_model_names(req.models)
    return _start_eval_job(_prompt_dicts(req.prompts), model_names, background_tasks)


@app.post("/eval/demo")
def run_demo(req: DemoRequest, background_tasks: BackgroundTasks):
    model_names = _normalize_model_names(req.models)
    with DEMO_PROMPTS_PATH.open(encoding="utf-8") as f:
        prompts = [PromptInput.model_validate(p) for p in json.load(f)]
    return _start_eval_job(_prompt_dicts(prompts), model_names, background_tasks)


@app.get("/eval/{job_id}")
def get_eval(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/eval/{job_id}/report")
def get_report(job_id: str, fmt: str = "markdown"):
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(status_code=404, detail="Job not ready")

    from evaluator.engine import EvalResult
    results = [EvalResult(**r) for r in job["results"]]

    if fmt == "csv":
        return Response(content=to_csv(results), media_type="text/csv")
    return {"report": to_markdown(results)}


@app.post("/redteam/run")
def run_red_team(req: RedTeamRequest, background_tasks: BackgroundTasks):
    model_names = _normalize_model_names(req.models)
    job_id = str(uuid.uuid4())
    _jobs.create(job_id, {"status": "running", "type": "redteam"})

    def _run():
        try:
            report = _red_team.run(model_names, req.categories)
            _jobs.update(job_id, {
                "status": "done",
                "summary": report.summary(),
                "pass_rates": report.pass_rate_by_model,
                "failures_by_category": {
                    k: len(v) for k, v in report.failures_by_category.items()
                },
            })
        except Exception as exc:
            logging.exception("Red-team job %s failed", job_id)
            _jobs.update(job_id, {"status": "failed", "error": str(exc)})

    background_tasks.add_task(_run)
    return {"job_id": job_id, "status": "running"}


@app.get("/models")
def list_models():
    return {"models": list(_engine.models.keys())}
