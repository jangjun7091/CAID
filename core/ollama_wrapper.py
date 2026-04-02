# Local LLM Integration (Ollama via OpenAI-compatible API)
"""
OllamaWrapper: full drop-in replacement for LLMWrapper that routes all LLM
calls to a local Ollama server via its OpenAI-compatible /v1 endpoint.
No Anthropic API key required.

Uses the `openai` Python library, which handles retries, timeouts, and proper
chat-completion message formatting automatically.

Environment variables (read by AgentOrchestrator.create()):
    LLM_BASE_URL   Base URL of the local inference server  (default: http://localhost:11434/v1)
    LLM_MODEL      Model tag to use                        (default: qwen2.5-coder:7b)

Implements the same public interface as LLMWrapper:
    complete()            — render a Jinja2 template, call the model, return raw text
    complete_structured() — render, call, parse JSON, validate via Pydantic
    complete_code()       — render, call, extract a Python code block
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Type, TypeVar

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class OllamaWrapper:
    """
    Calls a local Ollama server (or any OpenAI-compatible server) for all LLM
    operations in CAID.

    Args:
        model: Model tag for generation calls (e.g. ``"llama3"``).
        fast_model: Model tag for fast/cheap calls (``fast=True``).
                    Defaults to ``model`` — a single pulled model handles everything.
        base_url: Base URL of the inference server. Must include ``/v1``.
        api_key: API key sent to the server. Ollama accepts any non-empty string.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        fast_model: str | None = None,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.fast_model = fast_model or model
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        self._jinja = Environment(
            loader=FileSystemLoader(str(_PROMPTS_DIR)),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    # ------------------------------------------------------------------
    # Public interface — mirrors LLMWrapper exactly
    # ------------------------------------------------------------------

    def complete(
        self,
        template_name: str,
        variables: dict,
        *,
        system: str = "",
        max_tokens: int = 4096,
        fast: bool = False,
    ) -> str:
        """
        Render a Jinja2 prompt template and call the model.

        Args:
            template_name: Filename under ``prompts/``.
            variables: Values to inject into the template.
            system: Optional system prompt. Sent as the ``system`` role message.
            max_tokens: Maximum tokens to generate.
            fast: Use ``fast_model`` when True (same model by default).

        Returns:
            Raw text response from the model.
        """
        prompt = self._render(template_name, variables)
        model = self.fast_model if fast else self.model

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        logger.debug("OllamaWrapper.complete | model=%s | template=%s", model, template_name)

        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content or ""
        logger.debug("OllamaWrapper | response length=%d chars", len(text))
        return text

    def complete_structured(
        self,
        template_name: str,
        variables: dict,
        response_schema: Type[T],
        *,
        system: str = "",
        max_tokens: int = 4096,
        fast: bool = False,
    ) -> T:
        """
        Like ``complete()``, but parses the response as JSON and validates it
        against a Pydantic model.

        Returns:
            A validated instance of ``response_schema``.

        Raises:
            ValueError: If the response cannot be parsed or validated.
        """
        json_system = (
            (system + "\n\n" if system else "")
            + "Output ONLY valid JSON — no markdown fences, no explanation, "
            "no text before or after the JSON."
        )
        text = self.complete(
            template_name,
            variables,
            system=json_system,
            max_tokens=max_tokens,
            fast=fast,
        )

        text = _strip_fences(text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Llama3 occasionally truncates the response before the final
            # closing brace/bracket.  Try appending the missing delimiters.
            data = _repair_json(text)

        return response_schema.model_validate(data)

    def complete_code(
        self,
        template_name: str,
        variables: dict,
        *,
        system: str = "",
        max_tokens: int = 8192,
    ) -> str:
        """
        Like ``complete()``, but extracts a Python code block from the response.

        Returns:
            Pure Python source code string (no fences).

        Raises:
            ValueError: If no code block can be extracted.
        """
        text = self.complete(
            template_name,
            variables,
            system=system,
            max_tokens=max_tokens,
        )
        return _extract_code_block(text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _render(self, template_name: str, variables: dict) -> str:
        template = self._jinja.get_template(template_name)
        return template.render(**variables)


# ---------------------------------------------------------------------------
# Module-level helpers (shared with tests)
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    """Remove leading/trailing markdown fences and surrounding whitespace."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return text.strip()


def _repair_json(text: str) -> object:
    """
    Attempt to recover a valid JSON value from a malformed model response.

    Handles two distinct failure modes:

    * **Trailing text** (Qwen2.5-coder) — the model appends an explanation
      after the closing ``}`` or ``]``.  Fixed by trimming lines from the end
      until the remaining text parses cleanly.

    * **Truncation** (Llama3) — the model stops mid-object before writing the
      final closing delimiter.  Fixed by appending ``}`` / ``]`` variants.

    Raises:
        ValueError: If neither strategy yields valid JSON.
    """
    # Strategy 1 — strip trailing lines until the text is valid JSON.
    # Covers: JSON followed by explanation text, blank lines, or partial sentences.
    lines = text.splitlines()
    for end in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end]).rstrip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Strategy 2 — append missing closing delimiters.
    # Covers: response cut off before the final } or ].
    for suffix in ["}", "]", "\n}", "\n]", "}}", "\n}}"]:
        try:
            return json.loads(text + suffix)
        except json.JSONDecodeError:
            continue

    raise ValueError(
        f"Model returned malformed JSON that could not be repaired.\n\nRaw text:\n{text}"
    )


def _extract_code_block(text: str) -> str:
    """Extract content from the first ```python ... ``` block."""
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: if the whole response looks like code, return as-is
    if any(kw in text for kw in ("import ", "def ", "result =")):
        return text.strip()
    raise ValueError(
        "No Python code block found in model response.\n\nRaw response:\n" + text
    )
