import pytest
from evaluator.engine import EvalEngine, EvalResult, EvalSuiteError
from evaluator.models import AnthropicAdapter, MockAdapter, OpenAIAdapter
from evaluator.reporter import to_markdown, to_csv, best_model, model_stats
from agents.red_team import RedTeamAgent
from types import SimpleNamespace


@pytest.fixture
def engine():
    return EvalEngine(models=[
        MockAdapter("mock/fast", response="Paris is the capital of France."),
        MockAdapter("mock/slow", response="I think France has a capital somewhere in Europe."),
    ])


@pytest.fixture
def prompts():
    return [
        {"id": "geo_1", "prompt": "What is the capital of France?", "ground_truth": "Paris is the capital of France."},
        {"id": "geo_2", "prompt": "What is the capital of Germany?", "ground_truth": "Berlin is the capital of Germany."},
    ]


class TestEvalEngine:
    def test_run_single_returns_result(self, engine, prompts):
        r = engine.run_single("mock/fast", "geo_1", prompts[0]["prompt"], prompts[0]["ground_truth"])
        assert r.model == "mock/fast"
        assert r.prompt_id == "geo_1"
        assert 0.0 <= r.factuality_score <= 1.0
        assert 0.0 <= r.safety_score <= 1.0
        assert r.latency_ms >= 0

    def test_run_suite_returns_all_combos(self, engine, prompts):
        results = engine.run_suite(prompts)
        assert len(results) == 4  # 2 models x 2 prompts

    def test_run_suite_subset_models(self, engine, prompts):
        results = engine.run_suite(prompts, model_names=["mock/fast"])
        assert all(r.model == "mock/fast" for r in results)
        assert len(results) == 2

    def test_cost_is_zero_for_mock(self, engine, prompts):
        r = engine.run_single("mock/fast", "geo_1", prompts[0]["prompt"])
        assert r.cost_usd == 0.0

    def test_custom_factuality_scorer_receives_prompt(self, prompts):
        class PromptAwareScorer:
            def score(self, response, ground_truth, prompt):
                assert prompt == "What is the capital of France?"
                return 0.77

        custom_engine = EvalEngine(
            models=[MockAdapter("mock/fast", response="Paris")],
            factuality_scorer=PromptAwareScorer(),
        )

        r = custom_engine.run_single(
            "mock/fast",
            "geo_1",
            prompts[0]["prompt"],
            prompts[0]["ground_truth"],
        )
        assert r.factuality_score == 0.77

    def test_run_suite_reports_failed_cases(self, engine):
        prompts = [{"id": "bad", "ground_truth": "unused"}]

        with pytest.raises(EvalSuiteError) as exc_info:
            engine.run_suite(prompts, model_names=["mock/fast"])

        assert exc_info.value.failures[0].model == "mock/fast"
        assert exc_info.value.failures[0].prompt_id == "bad"
        assert exc_info.value.partial_results == []

    def test_latency_score_is_populated_and_in_range(self, engine, prompts):
        r = engine.run_single("mock/fast", "geo_1", prompts[0]["prompt"], prompts[0]["ground_truth"])
        assert r.latency_score in {0.25, 0.5, 0.75, 1.0}

    def test_best_model_uses_tier_normalized_latency(self):
        # Slow model has lower latency_score, so even with equal factuality the
        # fast model should win on the latency component.
        fast = EvalResult(
            model="mock/fast", prompt_id="p1", prompt="x", response="y",
            factuality_score=0.8, safety_score=1.0, latency_ms=100, cost_usd=0.0,
            latency_score=1.0,
        )
        slow = EvalResult(
            model="mock/slow", prompt_id="p1", prompt="x", response="y",
            factuality_score=0.8, safety_score=1.0, latency_ms=5000, cost_usd=0.0,
            latency_score=0.25,
        )
        assert best_model([fast, slow]) == "mock/fast"


class TestOpenAIAdapter:
    def test_retries_transient_provider_errors(self):
        class FakeCompletions:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("rate limit")
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="Hello from model")
                        )
                    ],
                    usage=SimpleNamespace(prompt_tokens=4, completion_tokens=3),
                )

        completions = FakeCompletions()
        adapter = OpenAIAdapter.__new__(OpenAIAdapter)
        adapter.client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        adapter.model = "gpt-4o-mini"
        adapter.name = "openai/gpt-4o-mini"
        adapter.max_tokens = 32
        adapter.max_retries = 1

        response, usage = adapter.complete("Say hello")

        assert response == "Hello from model"
        assert usage.prompt_tokens == 4
        assert usage.completion_tokens == 3
        assert completions.calls == 2


class TestAnthropicAdapter:
    def test_retries_transient_provider_errors(self):
        class FakeMessages:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("overloaded")
                return SimpleNamespace(
                    content=[SimpleNamespace(text="Hello from Claude")],
                    usage=SimpleNamespace(input_tokens=5, output_tokens=4),
                )

        messages = FakeMessages()
        adapter = AnthropicAdapter.__new__(AnthropicAdapter)
        adapter.client = SimpleNamespace(messages=messages)
        adapter.model = "claude-3-haiku-20240307"
        adapter.name = "anthropic/claude-3-haiku-20240307"
        adapter.max_tokens = 32
        adapter.max_retries = 1

        response, usage = adapter.complete("Say hello")

        assert response == "Hello from Claude"
        assert usage.prompt_tokens == 5
        assert usage.completion_tokens == 4
        assert messages.calls == 2


class TestReporter:
    def test_markdown_contains_model_names(self, engine, prompts):
        results = engine.run_suite(prompts)
        md = to_markdown(results)
        assert "mock/fast" in md
        assert "mock/slow" in md

    def test_csv_has_header_row(self, engine, prompts):
        results = engine.run_suite(prompts)
        csv_out = to_csv(results)
        assert csv_out.startswith("model,prompt_id")

    def test_csv_row_count(self, engine, prompts):
        results = engine.run_suite(prompts)
        rows = to_csv(results).strip().split("\n")
        assert len(rows) == len(results) + 1  # +1 for header

    def test_best_model_returns_string(self, engine, prompts):
        results = engine.run_suite(prompts)
        winner = best_model(results)
        assert winner in ["mock/fast", "mock/slow"]

    def test_model_stats_keys(self, engine, prompts):
        results = engine.run_suite(prompts, model_names=["mock/fast"])
        stats = model_stats(results)
        assert "factuality_avg" in stats
        assert "latency_p95_ms" in stats
        assert "cost_per_1k_usd" in stats


class TestRedTeamAgent:
    def test_run_returns_report(self, engine):
        agent = RedTeamAgent(engine)
        report = agent.run(categories=["hallucination_inducing"])
        assert report.total_attacks > 0
        assert "mock/fast" in report.pass_rate_by_model

    def test_pass_rate_between_0_and_1(self, engine):
        agent = RedTeamAgent(engine)
        report = agent.run(categories=["prompt_injection"])
        for rate in report.pass_rate_by_model.values():
            assert 0.0 <= rate <= 1.0
