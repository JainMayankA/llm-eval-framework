# llm-eval-framework

![CI](https://github.com/JainMayankA/llm-eval-framework/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)

A small framework for evaluating LLM outputs across factuality, safety, latency, and cost. It includes a CLI, FastAPI service, markdown/CSV reporting, mock adapters for local runs, provider adapters, and an automated red-team prompt suite.

## Why this exists

A model that scores well on one benchmark can still hallucinate, leak PII, ignore prompt-injection boundaries, or be too slow or expensive to ship. This runs the same prompts through one or more models and scores them on factuality, safety, latency, and cost, so the comparison is repeatable instead of ad hoc.

## Architecture

```text
llm-eval-framework/
|-- evaluator/
|   |-- engine.py       # Orchestrates model runs and surfaces suite failures
|   |-- scorers.py      # Heuristic, LLM-judge, moderation, and latency scorers
|   |-- models.py       # ModelAdapter ABC plus OpenAI, Anthropic, and mock adapters
|   `-- reporter.py     # Markdown/CSV report generation and composite scoring
|-- agents/
|   `-- red_team.py     # Generates adversarial prompts and reports pass rates
|-- api/
|   `-- server.py       # FastAPI REST API with typed request schemas
|-- benchmarks/
|   |-- demo_prompts.json
|   `-- sample_prompts.json
`-- scripts/
    `-- run_eval.py     # CLI entry point
```

## Evaluation Dimensions

| Dimension | Method | Score range |
|-----------|--------|-------------|
| Factuality | Token-overlap F1 by default, optional LLM judge | 0.0 to 1.0 |
| Safety | Local rules by default, optional OpenAI Moderation | 0.0 to 1.0 |
| Latency | Measured wall-clock latency, normalized to tiers | 0.25 to 1.0 |
| Cost | Token usage multiplied by provider pricing table | USD |

The `best_model()` function computes a weighted composite:

```text
factuality 40% + safety 30% + latency 20% + cost 10%
```

## Red-Team Categories

- `prompt_injection`: attempts to override system instructions.
- `jailbreak`: roleplay and fictional framing attacks.
- `data_extraction`: probing for hidden prompt contents.
- `hallucination_inducing`: prompts referencing fabricated sources.
- `bias_probing`: demographic and political bias elicitation.

## Quickstart

### Windows / PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

python scripts/run_eval.py --mock
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python scripts/run_eval.py --mock
```

## Ollama (free, local — no API key)

The fastest way to see a real model evaluated. [Install Ollama](https://ollama.com/download), pull a model, and run:

```bash
ollama pull llama3.2
ollama serve          # starts the server at localhost:11434
```

Then:

```bash
# Evaluate llama3.2 across all 50 benchmark prompts
python scripts/run_eval.py --ollama --prompts benchmarks/sample_prompts.json

# Compare two local models
python scripts/run_eval.py --ollama llama3.2 mistral --prompts benchmarks/sample_prompts.json

# Run red-team attacks against a local model
python scripts/run_eval.py --ollama --redteam

# Use Ollama as the factuality judge in the scorer comparison
python scripts/scorer_demo.py --ollama

# Start the REST API backed by Ollama (no keys needed)
EVAL_OLLAMA_MODELS=llama3.2 uvicorn api.server:app --host 127.0.0.1 --port 8000
```

Ollama cost is always $0.00 — `estimate_cost()` returns 0.0 for local inference. Latency depends on your hardware; expect 500ms–5s per prompt on a modern CPU without a GPU.

> The Ollama adapter uses the OpenAI-compatible endpoint at `localhost:11434/v1`, so any model available via `ollama pull` works: `llama3.2`, `mistral`, `gemma3`, `phi4`, `deepseek-r1`, etc.

## Real Provider Demo

Copy `.env.example` to `.env`, then set:

```text
OPENAI_API_KEY=sk-...
EVAL_OPENAI_MODELS=gpt-4o-mini
EVAL_FACTUALITY_SCORER=heuristic
EVAL_SAFETY_SCORER=heuristic
```

Run:

```bash
python scripts/run_eval.py --models gpt-4o-mini --prompts benchmarks/demo_prompts.json --output report.md
```

For a stronger factuality pass, use a separate LLM judge:

```text
EVAL_FACTUALITY_SCORER=llm-judge
EVAL_JUDGE_MODEL=gpt-4o-mini
```

Or from the CLI:

```bash
python scripts/run_eval.py --models gpt-4o-mini --prompts benchmarks/demo_prompts.json --factuality-scorer llm-judge
```

For a stronger safety pass in the API, use OpenAI Moderation:

```text
EVAL_SAFETY_SCORER=openai-moderation
EVAL_MODERATION_MODEL=omni-moderation-latest
```

The moderation scorer still runs local prompt-injection checks, because moderation APIs classify harmful content better than instruction-boundary attacks.

## CLI Examples

```bash
# Run the full sample benchmark with mock models
python scripts/run_eval.py --mock --prompts benchmarks/sample_prompts.json

# Run red-team sweep with a real model
python scripts/run_eval.py --models gpt-4o-mini --redteam

# Compare real OpenAI models
python scripts/run_eval.py --models gpt-4o-mini gpt-4o --prompts benchmarks/demo_prompts.json
```

If any prompt/model case fails inside a suite, the engine raises `EvalSuiteError` with the failed cases and partial results. This prevents reports from silently omitting failed benchmark rows.

## REST API

Start locally:

```bash
uvicorn api.server:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

Useful flow:

1. Run `GET /health`.
2. Run `POST /eval/demo` with `{}`.
3. Copy the returned `job_id`.
4. Run `GET /eval/{job_id}`.
5. Run `GET /eval/{job_id}/report`.

### Typed Prompt Schema

`POST /eval/run` validates each prompt with this schema:

```json
{
  "id": "q1",
  "prompt": "What is 2+2?",
  "ground_truth": "4",
  "category": "math",
  "metadata": {}
}
```

`id` and `prompt` are required non-empty strings. Unknown fields are rejected so benchmark payloads stay explicit.

Example request:

```bash
curl -X POST http://localhost:8000/eval/run \
  -H "Content-Type: application/json" \
  -d '{"prompts": [{"id": "q1", "prompt": "What is 2+2?", "ground_truth": "4"}]}'
```

Other endpoints:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/models
curl http://localhost:8000/eval/{job_id}
curl http://localhost:8000/eval/{job_id}/report

curl -X POST http://localhost:8000/redteam/run \
  -H "Content-Type: application/json" \
  -d '{"categories": ["prompt_injection", "jailbreak"]}'
```

If a background eval job hits failed cases, the job status becomes `failed` and includes:

```json
{
  "status": "failed",
  "results": [],
  "failures": [
    {"model": "openai/gpt-4o-mini", "prompt_id": "q1", "error": "..."}
  ]
}
```

## Docker

```bash
docker compose up --build
```

The container exposes:

```text
http://localhost:8000
```

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | empty | Enables OpenAI adapters and judge/moderation options |
| `ANTHROPIC_API_KEY` | empty | Enables Anthropic adapter usage when instantiated |
| `EVAL_OLLAMA_MODELS` | empty | Comma-separated Ollama models for the API (e.g. `llama3.2,mistral`) |
| `EVAL_OPENAI_MODELS` | `gpt-4o-mini` | Comma-separated OpenAI models for the API |
| `EVAL_FACTUALITY_SCORER` | `heuristic` | `heuristic` or `llm-judge` |
| `EVAL_JUDGE_MODEL` | `gpt-4o-mini` | Judge model for factuality scoring |
| `EVAL_JUDGE_MAX_TOKENS` | `256` | Max tokens for judge responses |
| `EVAL_SAFETY_SCORER` | `heuristic` | `heuristic` or `openai-moderation` |
| `EVAL_MODERATION_MODEL` | `omni-moderation-latest` | Moderation model name |
| `EVAL_PROVIDER_TIMEOUT_SECONDS` | `30` | Provider call timeout |
| `EVAL_PROVIDER_MAX_RETRIES` | `2` | Provider retry count |
| `EVAL_USE_MOCK` | `false` | Force API mock mode even when keys are present |

Provider adapters use the shared timeout and retry controls. OpenAI and Anthropic clients both disable SDK retries and apply explicit retry/backoff in the adapter.

## Extending

### Add a model provider

Implement `ModelAdapter` in `evaluator/models.py`:

```python
class MyAdapter(ModelAdapter):
    name = "my/model"

    def complete(self, prompt: str) -> tuple[str, Usage]:
        ...

    def estimate_cost(self, usage: Usage) -> float:
        ...
```

Then instantiate it from `scripts/run_eval.py` or `api/server.py`.

### Add or change scoring

Scorers live in `evaluator/scorers.py`. Keep deterministic scorers available for unit tests and local smoke runs. For production-like evaluations:

- Use `LLMJudgeFactualityScorer` for model-graded factuality.
- Use `OpenAIModerationSafetyScorer` for safety classification.
- Add semantic similarity scoring for non-exact answers.
- Add category-level metrics for coding, reasoning, and hallucination prompts.

### Add benchmarks

Add a JSON file under `benchmarks/`:

```json
[
  {
    "id": "my_prompt_001",
    "category": "factual",
    "prompt": "What is 2+2?",
    "ground_truth": "4"
  }
]
```

Run it with:

```bash
python scripts/run_eval.py --models gpt-4o-mini --prompts benchmarks/my_prompts.json
```

## Sample Output

### Model comparison report

Running the framework against two mock models across 50 prompts (5 categories):

```
| Model        | Prompts | Factuality | Safety | Latency p50 | Latency p95 | Cost/1k reqs | Flagged |
|--------------|---------|------------|--------|-------------|-------------|--------------|---------|
| mock/model-a |      50 |      0.016 |  1.000 |         10ms|         10ms|       $0.0000|       0 |
| mock/model-b |      50 |      0.017 |  1.000 |         10ms|         11ms|       $0.0000|       0 |
```

With real providers (representative numbers — actual results vary by prompt set):

```
| Model              | Prompts | Factuality | Safety | Latency p50 | Latency p95 | Cost/1k reqs | Flagged |
|--------------------|---------|------------|--------|-------------|-------------|--------------|---------|
| openai/gpt-4o      |      50 |      0.820 |  1.000 |       1180ms|       2340ms|       $0.2300|       0 |
| openai/gpt-4o-mini |      50 |      0.743 |  1.000 |        620ms|       1050ms|       $0.0180|       0 |
```

> Best model (composite score): **openai/gpt-4o-mini** — factuality is 10% lower but costs 93% less and is 47% faster. Whether that tradeoff is acceptable depends on your SLA.

The full per-prompt breakdown is saved to `benchmarks/sample_report.md` (mock run). Generate your own:

```bash
python scripts/run_eval.py --mock --output benchmarks/sample_report.md
```

### Red-team summary

```
Red-team report: 34 attacks
  mock/model-a: 100.0% pass rate
  mock/model-b: 100.0% pass rate
  No failures detected.
```

> Mock models always pass because they return a fixed safe string. Run with real models to get meaningful safety signal:
> ```bash
> python scripts/run_eval.py --models gpt-4o-mini --redteam
> ```

### Scorer comparison: heuristic vs LLM-judge

The heuristic scorer (token-overlap F1) only partly handles negation and under-scores short correct answers. Run the built-in demo to see it:

```bash
python scripts/scorer_demo.py
```

```
Scorer comparison — heuristic token-F1  vs  LLM-Judge (simulated)
==========================================================================================
ID                 Category          Heuristic    LLM-Judge (simulated)    Delta
------------------------------------------------------------------------------------------
negation_001       negation              0.700                    0.000    -0.70 <-- heuristic misled
negation_002       negation              0.571                    0.000    -0.57 <-- heuristic misled
negation_003       negation              0.706                    0.000    -0.71 <-- heuristic misled
negation_004       negation              0.222                    0.000    -0.22
negation_005       partial_credit        0.250                    0.850    +0.60 <-- heuristic misled
negation_006       partial_credit        0.769                    0.500    -0.27
------------------------------------------------------------------------------------------
```

Negation cases (001–004): the heuristic still scores factually wrong answers well above 0 because they reuse most of the ground truth's words. The 0.3 penalty only fires when the wrong answer itself contains a negation word (case 001), so the rest slip through. The LLM-judge returns 0.0.

Partial-credit cases (005–006): the heuristic penalises short correct answers and rewards verbose but partly wrong ones. The LLM-judge gives appropriate partial credit.

To run with a real OpenAI judge:

```bash
OPENAI_API_KEY=sk-... python scripts/scorer_demo.py --live
```

The edge-case prompts are in `benchmarks/negation_edge_cases.json` — extend them to cover your domain's failure modes.

## Run Tests

```bash
pytest tests/ -v
ruff check evaluator/ agents/ api/ scripts/
```

## Known Limitations

Out of scope for now:

- Jobs live in an in-process dict (`api/server.py`), so they don't survive a restart and the API won't run across more than one worker.
- The API has no auth or rate limiting. `/eval/run` can spend real money against your OpenAI key, so don't expose it publicly as-is.
- The heuristic factuality scorer is rough: it under-scores short correct answers and only partly handles negation (see `scripts/scorer_demo.py`). Use `EVAL_FACTUALITY_SCORER=llm-judge` when it matters.
- The heuristic safety scorer only catches obvious prompt-injection patterns, not general harmful content. Use `EVAL_SAFETY_SCORER=openai-moderation` for that.
- Latency is a single wall-clock measurement per prompt — fine for comparing models, not a load test. Results aren't persisted anywhere.
