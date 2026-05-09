import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional
from evaluator.scorers import FactualityScorer, SafetyScorer, LatencyScorer
from evaluator.models import ModelAdapter

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    model: str
    prompt_id: str
    prompt: str
    response: str
    factuality_score: float       # 0.0 to 1.0
    safety_score: float           # 0.0 = unsafe, 1.0 = safe
    latency_ms: float              # raw wall-clock measurement
    cost_usd: float
    latency_score: float = 0.0     # tier-normalized from latency_ms (0.25 to 1.0)
    flagged_categories: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalFailure:
    model: str
    prompt_id: str
    error: str


class EvalSuiteError(RuntimeError):
    """Raised when one or more prompt/model evals fail inside a suite."""

    def __init__(self, failures: list[EvalFailure], partial_results: list[EvalResult]):
        self.failures = failures
        self.partial_results = partial_results
        details = "; ".join(
            f"{f.model}/{f.prompt_id}: {f.error}" for f in failures[:5]
        )
        suffix = "" if len(failures) <= 5 else f"; +{len(failures) - 5} more"
        super().__init__(f"{len(failures)} eval case(s) failed: {details}{suffix}")


class EvalEngine:
    """
    Runs a prompt through one or more LLM adapters and scores each response
    across factuality, safety, latency, and cost dimensions.
    """

    def __init__(
        self,
        models: list[ModelAdapter],
        factuality_scorer=None,
        safety_scorer=None,
    ):
        self.models = {m.name: m for m in models}
        self.factuality = factuality_scorer or FactualityScorer()
        self.safety = safety_scorer or SafetyScorer()
        self.latency = LatencyScorer()

    def run_single(self, model_name: str, prompt_id: str, prompt: str,
                   ground_truth: Optional[str] = None) -> EvalResult:
        model = self.models[model_name]

        start = time.perf_counter()
        response, usage = model.complete(prompt)
        latency_ms = (time.perf_counter() - start) * 1000

        factuality = (
            self.factuality.score(response, ground_truth, prompt)
            if ground_truth else 0.5
        )
        safety_result = self.safety.score(prompt, response)
        cost = model.estimate_cost(usage)

        return EvalResult(
            model=model_name,
            prompt_id=prompt_id,
            prompt=prompt,
            response=response,
            factuality_score=factuality,
            safety_score=safety_result.score,
            latency_ms=round(latency_ms, 2),
            cost_usd=round(cost, 6),
            latency_score=self.latency.score(latency_ms),
            flagged_categories=safety_result.flagged_categories,
            metadata={"tokens": usage},
        )

    def run_suite(
        self,
        prompts: list[dict],
        model_names: Optional[list[str]] = None,
        fail_fast: bool = False,
        max_workers: int = 8,
    ) -> list[EvalResult]:
        targets = model_names or list(self.models.keys())
        results: list[EvalResult] = []
        failures: list[EvalFailure] = []

        work = [(model_name, p) for model_name in targets for p in prompts]
        if not work:
            return results

        with ThreadPoolExecutor(max_workers=min(max_workers, len(work))) as executor:
            future_map = {
                executor.submit(
                    lambda mn=model_name, pd=p: self.run_single(
                        mn, pd["id"], pd["prompt"], pd.get("ground_truth")
                    )
                ): (model_name, p.get("id", "<missing-id>"))
                for model_name, p in work
            }

            for future in as_completed(future_map):
                model_name, prompt_id = future_map[future]
                try:
                    r = future.result()
                    results.append(r)
                    logger.info(
                        "%s | %s | fact=%.2f safe=%.2f lat=%.0fms",
                        model_name, prompt_id,
                        r.factuality_score, r.safety_score, r.latency_ms,
                    )
                except Exception as e:
                    failure = EvalFailure(model=model_name, prompt_id=prompt_id, error=str(e))
                    failures.append(failure)
                    logger.exception("Failed %s/%s", model_name, prompt_id)
                    if fail_fast:
                        for f in future_map:
                            f.cancel()
                        raise EvalSuiteError(failures, results) from e

        if failures:
            raise EvalSuiteError(failures, results)
        return results
