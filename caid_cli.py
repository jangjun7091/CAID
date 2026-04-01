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
    orchestrator = AgentOrchestrator.create(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        model=os.environ.get("CAID_MODEL", "claude-sonnet-4-6"),
        output_dir=output_dir,
        max_iterations=args.max_iterations,
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
        "--max-iterations", type=int, default=5, metavar="N",
        help="Max Design-Critique-Refine cycles per component (default: 5)"
    )
    design_parser.add_argument(
        "--output", default="output", metavar="DIR",
        help="Output directory for STEP/STL files (default: ./output)"
    )
    design_parser.add_argument(
        "--save", action="store_true",
        help="Save session log to output/session.json"
    )

    args = parser.parse_args()
    _configure_logging(args.verbose)

    if args.command == "design":
        sys.exit(cmd_design(args))


if __name__ == "__main__":
    main()
