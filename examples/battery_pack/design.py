"""
Battery Pack Module Design — CAID Phase 2 Example

Designs a 3-component lithium-ion battery module housing:
  1. base_tray       — structural base with cell retention features
  2. cover_plate     — top cover that snap-fits onto the base tray
  3. busbar_bracket  — internal bracket to route and support busbars

Material: Al6061, CNC milled
Safety factor: 2.5 (vibration + thermal cycling environment)

Run from the CAID root:
    python -m examples.battery_pack.design
    # or
    python examples/battery_pack/design.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

BRIEF = """
Design a lithium-ion battery module housing for an EV battery pack with three components:

1. base_tray: Rectangular tray (200 mm × 150 mm × 40 mm) to hold 12 cylindrical
   18650 cells (3×4 arrangement, 18.4 mm diameter, 65 mm tall). Walls must be
   at least 2 mm thick. Mounting holes (M5, 5.5 mm diameter) at all four corners,
   10 mm from each edge. Cell retention bosses (19 mm inner diameter, 3 mm tall)
   spaced at 21 mm pitch in a 3×4 grid.

2. cover_plate: Flat plate (200 mm × 150 mm × 3 mm) with four M5 clearance holes
   matching the base_tray corner holes. Two rectangular cutouts (40 mm × 10 mm)
   for busbar access, centered on the 200 mm edges.

3. busbar_bracket: L-shaped bracket (80 mm × 20 mm × 15 mm, 2.5 mm wall) that
   clips onto the inside wall of the base_tray to route busbars.

Material: Al6061, CNC milled.
Safety factor: 2.5.
Critical tolerance: M5 mounting hole pattern must have centre distance
150 mm ± 0.1 mm (cover to tray mating dimension).
"""

OUTPUT_DIR = Path(__file__).parent / "output"


def main() -> int:
    from core.orchestrator import AgentOrchestrator

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.")
        return 1

    print("\n" + "=" * 70)
    print("CAID — Battery Pack Module Design Example")
    print("=" * 70)
    print(f"Brief:\n{BRIEF.strip()}\n")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70 + "\n")

    orchestrator = AgentOrchestrator.create(
        api_key=api_key,
        output_dir=OUTPUT_DIR,
        max_iterations=5,
    )

    session = orchestrator.run(BRIEF)

    print("\n" + "=" * 70)
    print("SESSION RESULTS")
    print("=" * 70)
    print(session.summary())
    print("=" * 70)

    session.save(OUTPUT_DIR / "session.json")
    print(f"\nSession log saved to: {OUTPUT_DIR / 'session.json'}")

    if session.all_passed:
        print("\nAll components passed. STEP files:")
        for name, result in session.results.items():
            print(f"  {name}: {result.step_path}")
        return 0
    else:
        failed = [n for n, r in session.results.items() if not r.passed]
        print(f"\nFailed components: {failed}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
