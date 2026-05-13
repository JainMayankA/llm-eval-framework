#!/usr/bin/env python3
"""
Demonstrates the gap between heuristic and LLM-judge factuality scoring.

Runs both scorers against negation and partial-credit cases from
benchmarks/negation_edge_cases.json.

Without an API key the LLM-judge column uses a local MockJudge that
returns pre-set realistic scores so the comparison is always runnable.
With a real OPENAI_API_KEY, pass --live to call GPT-4o-mini as the judge.

Usage:
    python scripts/scorer_demo.py              # always runs, no key needed
    python scripts/scorer_demo.py --live       # calls real judge (requires OPENAI_API_KEY)
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluator.scorers import FactualityScorer, LLMJudgeFactualityScorer

BENCHMARK_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "benchmarks", "negation_edge_cases.json",
)

# Pre-set realistic scores a real LLM judge would return for each case.
# Keys match the `id` field in negation_edge_cases.json.
MOCK_JUDGE_SCORES = {
    "negation_001": 0.00,  # wrong answer (negation)
    "negation_002": 0.00,  # factually inverted
    "negation_003": 0.00,  # Nobel Prize misconception
    "negation_004": 0.00,  # cold-weather myth
    "negation_005": 0.85,  # short but correct
    "negation_006": 0.50,  # mostly correct, adds false detail
}


class MockJudge:
    """Stand-in for a real LLM judge. Returns pre-set realistic scores."""

    name = "mock/judge"

    def complete(self, prompt: str):
        from evaluator.models import Usage
        for case_id, score in MOCK_JUDGE_SCORES.items():
            if case_id in prompt:
                return json.dumps({"score": score, "reason": "pre-set demo score"}), Usage(0, 0)
        return json.dumps({"score": 0.5, "reason": "unknown case"}), Usage(0, 0)


def _build_judge_prompt(case: dict, response: str) -> str:
    return (
        f"Case ID: {case['id']}\n"
        f"Prompt: {case['prompt']}\n"
        f"Ground truth: {case['ground_truth']}\n"
        f"Response: {response}"
    )


def run(live: bool = False, ollama_model: str | None = None):
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        cases = json.load(f)

    heuristic = FactualityScorer()

    if live:
        if not os.getenv("OPENAI_API_KEY"):
            print("ERROR: --live requires OPENAI_API_KEY to be set.")
            sys.exit(1)
        from evaluator.models import OpenAIAdapter
        judge_adapter = OpenAIAdapter("gpt-4o-mini", max_tokens=256)
        llm_judge = LLMJudgeFactualityScorer(judge_adapter)
        judge_label = "LLM-Judge (gpt-4o-mini)"
    elif ollama_model:
        from evaluator.models import OllamaAdapter
        judge_adapter = OllamaAdapter(ollama_model, max_tokens=256)
        llm_judge = LLMJudgeFactualityScorer(judge_adapter)
        judge_label = f"LLM-Judge (ollama/{ollama_model})"
    else:
        llm_judge = LLMJudgeFactualityScorer(MockJudge())
        judge_label = "LLM-Judge (simulated)"

    print(f"\nScorer comparison — heuristic token-F1  vs  {judge_label}")
    print("=" * 90)
    header = f"{'ID':<18} {'Category':<16} {'Heuristic':>10} {judge_label:>24} {'Delta':>8}"
    print(header)
    print("-" * 90)

    for case in cases:
        response = case["adversarial_response"]
        ground_truth = case["ground_truth"]
        prompt = case["prompt"]

        h_score = heuristic.score(response, ground_truth, prompt)

        judge_prompt = _build_judge_prompt(case, response)
        j_score = llm_judge.score(response, ground_truth, judge_prompt)

        delta = j_score - h_score
        delta_str = f"{delta:+.2f}"
        flag = " <-- heuristic misled" if abs(delta) >= 0.3 else ""

        print(
            f"{case['id']:<18} {case['category']:<16} {h_score:>10.3f} {j_score:>24.3f} "
            f"{delta_str:>8}{flag}"
        )

    print("-" * 90)
    print(
        "\nNegation cases: heuristic over-scores wrong answers due to word overlap.\n"
        "Partial-credit cases: heuristic under-scores short correct answers.\n"
        "LLM-judge handles both correctly.\n"
        "\nTo run with a real judge:\n"
        "  OPENAI_API_KEY=sk-... python scripts/scorer_demo.py --live\n"
        "  python scripts/scorer_demo.py --ollama              # uses llama3.2\n"
        "  python scripts/scorer_demo.py --ollama mistral      # uses mistral\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heuristic vs LLM-judge scorer comparison")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--live", action="store_true",
                       help="Call real OpenAI gpt-4o-mini as judge (requires OPENAI_API_KEY)")
    group.add_argument("--ollama", metavar="MODEL", nargs="?", const="llama3.2",
                       help="Call a local Ollama model as judge (default: llama3.2). "
                            "Requires Ollama running at localhost:11434.")
    args = parser.parse_args()
    run(live=args.live, ollama_model=args.ollama)
