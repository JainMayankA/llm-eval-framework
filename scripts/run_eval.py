#!/usr/bin/env python3
"""
CLI entry point for running eval suites.

Usage:
    # Run eval on sample prompts with mock models (no API keys needed)
    python scripts/run_eval.py --mock

    # Run with real OpenAI models
    OPENAI_API_KEY=sk-... python scripts/run_eval.py --models gpt-4o-mini gpt-3.5-turbo

    # Run red-team only
    python scripts/run_eval.py --mock --redteam

    # Save report
    python scripts/run_eval.py --mock --output report.md
"""

import argparse
import json
import logging
import sys
import os
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluator.engine import EvalEngine
from evaluator.models import MockAdapter, OllamaAdapter, OpenAIAdapter
from evaluator.scorers import LLMJudgeFactualityScorer, OpenAIModerationSafetyScorer
from evaluator.reporter import to_markdown, to_csv, best_model
from agents.red_team import RedTeamAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run LLM eval suite")
    parser.add_argument("--mock", action="store_true", help="Use mock adapters (no API keys)")
    parser.add_argument("--ollama", nargs="*", metavar="MODEL",
                        help="Run against local Ollama models (default: llama3.2). "
                             "Requires Ollama running at localhost:11434.")
    parser.add_argument("--models", nargs="+", help="OpenAI model names to evaluate")
    parser.add_argument("--prompts", default="benchmarks/sample_prompts.json")
    parser.add_argument("--redteam", action="store_true", help="Run red-team attacks")
    parser.add_argument("--output", help="Save markdown report to file")
    parser.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    parser.add_argument(
        "--factuality-scorer",
        choices=["heuristic", "llm-judge"],
        default=os.getenv("EVAL_FACTUALITY_SCORER", "heuristic"),
        help="Use token-overlap scoring or an OpenAI judge model",
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("EVAL_JUDGE_MODEL", "gpt-4o-mini"),
        help="OpenAI model used when --factuality-scorer=llm-judge",
    )
    parser.add_argument(
        "--safety-scorer",
        choices=["heuristic", "openai-moderation"],
        default=os.getenv("EVAL_SAFETY_SCORER", "heuristic"),
        help="Use local rules or OpenAI Moderation for safety scoring",
    )
    args = parser.parse_args()

    if args.mock:
        adapters = [
            MockAdapter("mock/model-a", "Paris is the capital of France."),
            MockAdapter("mock/model-b", "France's capital city is Paris, located in northern France."),
        ]
    elif args.ollama is not None:
        model_names = args.ollama if args.ollama else ["llama3.2"]
        adapters = [OllamaAdapter(m) for m in model_names]
    elif args.models:
        adapters = [OpenAIAdapter(m) for m in args.models]
    else:
        logger.error("Specify --mock, --ollama, or --models. Run with --help for usage.")
        sys.exit(1)

    factuality_scorer = None
    if args.factuality_scorer == "llm-judge":
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("Falling back to heuristic factuality: OPENAI_API_KEY is not set")
        else:
            judge = OpenAIAdapter(
                args.judge_model,
                max_tokens=int(os.getenv("EVAL_JUDGE_MAX_TOKENS", "256")),
            )
            factuality_scorer = LLMJudgeFactualityScorer(judge)

    safety_scorer = None
    if args.safety_scorer == "openai-moderation":
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("Falling back to heuristic safety: OPENAI_API_KEY is not set")
        else:
            safety_scorer = OpenAIModerationSafetyScorer(
                model=os.getenv("EVAL_MODERATION_MODEL", "omni-moderation-latest")
            )

    engine = EvalEngine(
        adapters,
        factuality_scorer=factuality_scorer,
        safety_scorer=safety_scorer,
    )

    with open(args.prompts, encoding="utf-8") as f:
        prompts = json.load(f)

    logger.info(f"Running {len(prompts)} prompts across {len(adapters)} models...")
    results = engine.run_suite(prompts)

    if args.format == "csv":
        report = to_csv(results)
    else:
        report = to_markdown(results)

    winner = best_model(results)
    logger.info(f"\nBest model (composite score): {winner}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Report saved to {args.output}")
    else:
        print("\n" + report)

    if args.redteam:
        logger.info("\nRunning red-team attacks...")
        agent = RedTeamAgent(engine)
        rt_report = agent.run()
        print("\n" + rt_report.summary())


if __name__ == "__main__":
    main()
