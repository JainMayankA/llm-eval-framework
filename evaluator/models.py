import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int


class ModelAdapter(ABC):
    name: str

    @abstractmethod
    def complete(self, prompt: str) -> tuple[str, Usage]:
        """Returns (response_text, usage)."""

    @abstractmethod
    def estimate_cost(self, usage: Usage) -> float:
        """Returns cost in USD."""


class OpenAIAdapter(ModelAdapter):
    # Pricing per 1M tokens. Keep this table in sync with provider docs for
    # cost-sensitive comparisons.
    PRICING = {
        "gpt-4o":           {"input": 5.0,   "output": 15.0},
        "gpt-4o-mini":      {"input": 0.15,  "output": 0.60},
        "gpt-3.5-turbo":    {"input": 0.50,  "output": 1.50},
    }

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 512,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        import openai
        self.timeout = timeout or float(os.getenv("EVAL_PROVIDER_TIMEOUT_SECONDS", "30"))
        self.max_retries = (
            max_retries
            if max_retries is not None
            else int(os.getenv("EVAL_PROVIDER_MAX_RETRIES", "2"))
        )
        self.client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=self.timeout,
            max_retries=0,
        )
        self.model = model
        self.name = f"openai/{model}"
        self.max_tokens = max_tokens

    def complete(self, prompt: str) -> tuple[str, Usage]:
        resp = None
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                time.sleep(min(2 ** attempt, 8))

        if resp is None:
            raise RuntimeError("OpenAI request failed") from last_error

        response_text = resp.choices[0].message.content or ""
        if resp.usage:
            prompt_tokens = resp.usage.prompt_tokens
            completion_tokens = resp.usage.completion_tokens
        else:
            prompt_tokens = len(prompt.split())
            completion_tokens = len(response_text.split())

        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return response_text, usage

    def estimate_cost(self, usage: Usage) -> float:
        p = self.PRICING.get(self.model, {"input": 1.0, "output": 3.0})
        return (usage.prompt_tokens * p["input"] + usage.completion_tokens * p["output"]) / 1_000_000


class AnthropicAdapter(ModelAdapter):
    PRICING = {
        "claude-3-haiku-20240307":  {"input": 0.25,  "output": 1.25},
        "claude-3-sonnet-20240229": {"input": 3.0,   "output": 15.0},
        "claude-3-opus-20240229":   {"input": 15.0,  "output": 75.0},
    }

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        max_tokens: int = 512,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        import anthropic
        self.timeout = timeout or float(os.getenv("EVAL_PROVIDER_TIMEOUT_SECONDS", "30"))
        self.max_retries = (
            max_retries
            if max_retries is not None
            else int(os.getenv("EVAL_PROVIDER_MAX_RETRIES", "2"))
        )
        self.client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            timeout=self.timeout,
            max_retries=0,
        )
        self.model = model
        self.name = f"anthropic/{model}"
        self.max_tokens = max_tokens

    def complete(self, prompt: str) -> tuple[str, Usage]:
        resp = None
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                time.sleep(min(2 ** attempt, 8))

        if resp is None:
            raise RuntimeError("Anthropic request failed") from last_error

        usage = Usage(
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
        )
        return resp.content[0].text, usage

    def estimate_cost(self, usage: Usage) -> float:
        p = self.PRICING.get(self.model, {"input": 3.0, "output": 15.0})
        return (usage.prompt_tokens * p["input"] + usage.completion_tokens * p["output"]) / 1_000_000


class OllamaAdapter(ModelAdapter):
    """
    Adapter for locally-running Ollama models (https://ollama.com).

    Ollama exposes an OpenAI-compatible REST API at localhost:11434/v1,
    so this reuses the openai client with a custom base_url — no API key
    or internet connection required.

    Prerequisites:
        1. Install Ollama: https://ollama.com/download
        2. Pull a model:  ollama pull llama3.2
        3. Start server:  ollama serve   (or it starts automatically on most installs)
    """

    def __init__(
        self,
        model: str = "llama3.2",
        max_tokens: int = 512,
        base_url: str = "http://localhost:11434/v1",
        timeout: Optional[float] = None,
    ):
        import openai
        self.model = model
        self.name = f"ollama/{model}"
        self.max_tokens = max_tokens
        self.timeout = timeout or float(os.getenv("EVAL_PROVIDER_TIMEOUT_SECONDS", "60"))
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key="ollama",  # Ollama accepts any non-empty string
            timeout=self.timeout,
            max_retries=0,
        )

    def complete(self, prompt: str) -> tuple[str, Usage]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
        )
        response_text = resp.choices[0].message.content or ""
        if resp.usage:
            usage = Usage(
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
            )
        else:
            usage = Usage(
                prompt_tokens=len(prompt.split()),
                completion_tokens=len(response_text.split()),
            )
        return response_text, usage

    def estimate_cost(self, usage: Usage) -> float:
        return 0.0  # local inference has no per-token billing


class MockAdapter(ModelAdapter):
    """Deterministic adapter for unit tests - no API key required."""

    def __init__(self, name: str = "mock/test", response: str = "Paris is the capital of France."):
        self.name = name
        self._response = response

    def complete(self, prompt: str) -> tuple[str, Usage]:
        time.sleep(0.01)  # simulate slight latency
        return self._response, Usage(prompt_tokens=20, completion_tokens=10)

    def estimate_cost(self, usage: Usage) -> float:
        return 0.0
