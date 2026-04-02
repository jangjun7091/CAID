"""
CAID command-line interface.

Usage:
    caid design "Design a mounting bracket for a 200g motor, Al6061, CNC milled"
    caid design "L-bracket, PA12, SLS" --max-iterations 3 --output ./my_parts
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_design(args: argparse.Namespace) -> int:
    from core.orchestrator import AgentOrchestrator

    output_dir = Path(args.output)
    use_local = args.local or os.environ.get("CAID_USE_LOCAL_LLM", "").lower() in ("1", "true", "yes")
    orchestrator = AgentOrchestrator.create(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        model=os.environ.get("CAID_MODEL", "claude-sonnet-4-6"),
        output_dir=output_dir,
        max_iterations=args.max_iterations,
        use_local_llm=use_local or None,   # None lets orchestrator read env var
        # CLI flags take priority; fall back to LLM_* env vars inside create()
        ollama_model=args.ollama_model or os.environ.get("CAID_OLLAMA_MODEL"),
        ollama_base_url=args.ollama_url or os.environ.get("CAID_OLLAMA_URL"),
    )

    print(f"\nCAID — Running design loop for:\n  \"{args.brief}\"\n")
    session = orchestrator.run(args.brief)

    print("\n" + "=" * 60)
    print(session.summary())
    print("=" * 60)

    if args.save:
        save_path = output_dir / "session.json"
        session.save(save_path)
        print(f"\nSession log saved to: {save_path}")

    return 0 if session.all_passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="caid",
        description="Computer AI Design — Agentic hardware design framework",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    design_parser = sub.add_parser("design", help="Run the design loop for a brief")
    design_parser.add_argument("brief", help="Natural-language design brief")
    design_parser.add_argument(
        "--max-iterations", type=int, default=3, metavar="N",
        help="Max Design-Critique-Refine cycles per component (default: 3)"
    )
    design_parser.add_argument(
        "--output", default="output", metavar="DIR",
        help="Output directory for STEP/STL files (default: ./output)"
    )
    design_parser.add_argument(
        "--save", action="store_true",
        help="Save session log to output/session.json"
    )
    design_parser.add_argument(
        "--local", action="store_true",
        help="Use local Ollama instead of the Anthropic API (no API key required)"
    )
    design_parser.add_argument(
        "--ollama-model", default=None, metavar="MODEL",
        help="Ollama model tag (default: llama3, or CAID_OLLAMA_MODEL env var)"
    )
    design_parser.add_argument(
        "--ollama-url", default=None, metavar="URL",
        help="Ollama server URL (default: http://localhost:11434, or CAID_OLLAMA_URL env var)"
    )

    args = parser.parse_args()
    _configure_logging(args.verbose)

    if args.command == "design":
        sys.exit(cmd_design(args))


if __name__ == "__main__":
    main()
