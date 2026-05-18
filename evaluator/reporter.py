"""
Generates comparison reports from eval results.
Outputs: markdown table, CSV, and per-model summary stats.
"""

import csv
import io
import statistics
from collections import defaultdict
from evaluator.engine import EvalResult


def _group_by_model(results: list[EvalResult]) -> dict[str, list[EvalResult]]:
    grouped = defaultdict(list)
    for r in results:
        grouped[r.model].append(r)
    return dict(grouped)


def model_stats(results: list[EvalResult]) -> dict:
    if not results:
        return {}
    return {
        "count": len(results),
        "factuality_avg": round(statistics.mean(r.factuality_score for r in results), 3),
        "factuality_p50": round(statistics.median(r.factuality_score for r in results), 3),
        "safety_avg": round(statistics.mean(r.safety_score for r in results), 3),
        "latency_p50_ms": round(statistics.median(r.latency_ms for r in results), 1),
        "latency_p95_ms": round(sorted(r.latency_ms for r in results)[int(len(results) * 0.95)], 1),
        "latency_score_avg": round(statistics.mean(r.latency_score for r in results), 3),
        "cost_total_usd": round(sum(r.cost_usd for r in results), 6),
        "cost_per_1k_usd": round(sum(r.cost_usd for r in results) / len(results) * 1000, 4),
        "flagged_count": sum(1 for r in results if r.flagged_categories),
    }


def to_markdown(results: list[EvalResult]) -> str:
    grouped = _group_by_model(results)
    winner = best_model(results)

    max_cost_per_1k = max(
        (model_stats(res)["cost_per_1k_usd"] for res in grouped.values()),
        default=0.0,
    )

    def _composite(s: dict) -> float:
        cost_score = (
            1 - (s["cost_per_1k_usd"] / max_cost_per_1k) if max_cost_per_1k > 0 else 1.0
        )
        return round(
            0.4 * s["factuality_avg"]
            + 0.3 * s["safety_avg"]
            + 0.2 * s["latency_score_avg"]
            + 0.1 * cost_score,
            4,
        )

    lines = [
        "# LLM Evaluation Report\n",
        "## Model comparison\n",
        "| Model | Prompts | Factuality | Safety | Latency p50 | Latency p95 | Cost/1k reqs | Flagged | Score |",
        "|-------|---------|-----------|--------|-------------|-------------|-------------|---------|-------|",
    ]

    for model, res in sorted(grouped.items()):
        s = model_stats(res)
        label = f"**{model}**" if model == winner else model
        lines.append(
            f"| {label} | {s['count']} | {s['factuality_avg']:.3f} | {s['safety_avg']:.3f} "
            f"| {s['latency_p50_ms']:.0f}ms | {s['latency_p95_ms']:.0f}ms "
            f"| ${s['cost_per_1k_usd']:.4f} | {s['flagged_count']} | {_composite(s):.4f} |"
        )

    lines.append("\n## Per-prompt breakdown\n")
    lines.append("| Model | Prompt ID | Factuality | Safety | Latency (ms) | Flagged |")
    lines.append("|-------|-----------|-----------|--------|-------------|---------|")
    for r in results:
        flagged = ", ".join(r.flagged_categories) or "-"
        lines.append(
            f"| {r.model} | {r.prompt_id} | {r.factuality_score:.3f} "
            f"| {r.safety_score:.3f} | {r.latency_ms:.0f} | {flagged} |"
        )

    return "\n".join(lines)


def to_csv(results: list[EvalResult]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "model", "prompt_id", "factuality_score", "safety_score",
        "latency_ms", "cost_usd", "flagged_categories"
    ])
    for r in results:
        writer.writerow([
            r.model, r.prompt_id, r.factuality_score, r.safety_score,
            r.latency_ms, r.cost_usd, "|".join(r.flagged_categories)
        ])
    return output.getvalue()


def best_model(results: list[EvalResult], weight_factuality=0.4,
               weight_safety=0.3, weight_latency=0.2, weight_cost=0.1) -> str:
    """Returns the model with the highest weighted composite score.

    Latency uses the tier-normalized `latency_score` produced by
    `LatencyScorer` in the engine — same 0.25-1.0 range documented in the
    README. Cost is normalized against the most-expensive model in the run
    so the cheapest gets 1.0 and the priciest gets 0.0; for a single-model
    run cost contribution collapses to 0 by construction.
    """
    grouped = _group_by_model(results)
    if not grouped:
        return ""

    scores = {}
    max_cost_per_1k = max(
        (model_stats(res)["cost_per_1k_usd"] for res in grouped.values()),
        default=0.0,
    )

    for model, res in grouped.items():
        s = model_stats(res)
        latency_score = s["latency_score_avg"]
        cost_score = (
            1 - (s["cost_per_1k_usd"] / max_cost_per_1k)
            if max_cost_per_1k > 0
            else 1.0
        )
        composite = (
            weight_factuality * s["factuality_avg"]
            + weight_safety * s["safety_avg"]
            + weight_latency * latency_score
            + weight_cost * cost_score
        )
        scores[model] = round(composite, 4)

    return max(scores, key=scores.get)
