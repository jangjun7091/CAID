# LLM Integration and Prompt Management
"""
Single gateway to all LLM calls in CAID.
- Manages prompt templates (Jinja2)
- Wraps the Anthropic SDK
- Supports structured (JSON/Pydantic) output
- Logs all prompts and completions
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TypeVar, Type

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class LLMWrapper:
    """
    Wraps the Anthropic Claude API with prompt templating and structured output.

    Args:
        api_key: Anthropic API key. Reads ANTHROPIC_API_KEY env var if None.
        model: Claude model ID for generation calls.
        fast_model: Claude model ID for fast/cheap calls (e.g., critique passes).
        max_retries: Number of retries on transient API errors.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        fast_model: str = "claude-haiku-4-5-20251001",
        max_retries: int = 3,
    ) -> None:
        import anthropic  # lazy import — not needed when using OllamaWrapper
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=max_retries)
        self.model = model
        self.fast_model = fast_model
        self._jinja = Environment(
            loader=FileSystemLoader(str(_PROMPTS_DIR)),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    # ------------------------------------------------------------------
    # Public API
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
        Render a Jinja2 prompt template and call Claude.

        Args:
            template_name: Filename under prompts/ (e.g., "designer_generate.jinja2").
            variables: Values to inject into the template.
            system: Optional system prompt.
            max_tokens: Maximum tokens in the response.
            fast: Use the faster/cheaper model if True.

        Returns:
            Raw text response from Claude.
        """
        prompt = self._render(template_name, variables)
        model = self.fast_model if fast else self.model

        logger.debug("LLM call | model=%s | template=%s", model, template_name)

        messages = [{"role": "user", "content": prompt}]
        kwargs: dict = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        text = response.content[0].text

        logger.debug("LLM response | output_tokens=%d", response.usage.output_tokens)
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
        Like complete(), but parses the response as JSON and validates it
        against a Pydantic model.

        The prompt template must instruct Claude to respond with raw JSON only.

        Returns:
            A validated instance of response_schema.

        Raises:
            ValueError: If the response cannot be parsed or validated.
        """
        json_system = (
            (system + "\n\n" if system else "")
            + "Respond with valid JSON only. No markdown fences, no explanation."
        )
        text = self.complete(
            template_name,
            variables,
            system=json_system,
            max_tokens=max_tokens,
            fast=fast,
        )

        # Strip accidental markdown fences
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Claude returned non-JSON: {exc}\n\nRaw text:\n{text}"
            ) from exc

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
        Like complete(), but extracts a Python code block from the response.

        The template should instruct Claude to wrap code in ```python fences.

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


def _extract_code_block(text: str) -> str:
    """Extract content from the first ```python ... ``` block."""
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: if the whole response looks like code, return as-is
    if any(kw in text for kw in ("import ", "def ", "result =")):
        return text.strip()
    raise ValueError(
        "No Python code block found in LLM response.\n\nRaw response:\n" + text
    )
