"""
Test 3 (fixed): main_chassis_v1 + four M4 hex bolts in corner holes.

Fixes applied vs. the failed run:
  1. cq.Color("gray")  — "darkgray" is not a valid CadQuery color name.
  2. bolt_z = +5.0     — bolt-head origin placed at the hex-socket floor
                         (Z = +5.0 mm), not at Z = 0 which buried bolts inside
                         the chassis body and caused 46 814 mm³ interference.

Chassis geometry (from STEP analysis of output/main_chassis_v1.step):
  • Body:        120 × 80 × 14.8 mm, centred at origin (Z = −7.0 … +7.8 mm)
  • Hex sockets: across-corners 7.0 mm, depth 2.8 mm (Z = +5.0 … +7.8 mm)
  • Clearance holes: ∅4.5 mm, Z = −5.0 … +5.0 mm
  • Tap holes:   ∅4.0 mm, Z = −7.0 … −5.0 mm
  • Corner centres: (±52, ±32) mm

Run from the CAID root:
    python examples/test_chassis_bolt_assembly.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cadquery as cq

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# M4 hex bolt head — must fit in the 7.0 mm across-corners hex socket
BOLT_HEAD_ACROSS_CORNERS = 7.0   # mm  (circumscribed circle diameter for polygon(6, …))
BOLT_HEAD_HEIGHT         = 2.8   # mm  (exactly fills the 2.8 mm socket depth)
SHANK_R                  = 2.0   # mm  (M4 nominal; 4.5 mm clearance hole gives 0.25 mm gap)
BOLT_LENGTH              = 14.0  # mm  (shank: Z = +5.0 → −9.0; exits chassis at Z = −7.0)

# Hole centres from chassis STEP (circular-edge centres)
HOLE_XY = [
    ( 52.0,  32.0),
    (-52.0,  32.0),
    ( 52.0, -32.0),
    (-52.0, -32.0),
]

# Z origin of the bolt = floor of hex socket = top of clearance-hole section
BOLT_Z = 5.0   # mm


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def make_m4_bolt() -> cq.Workplane:
    """Single M4 hex bolt: hex head (7.0 mm ⊙, 2.8 mm tall) + plain shank."""
    head  = cq.Workplane("XY").polygon(6, BOLT_HEAD_ACROSS_CORNERS).extrude(BOLT_HEAD_HEIGHT)
    shank = cq.Workplane("XY").circle(SHANK_R).extrude(-BOLT_LENGTH)
    return head.union(shank)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_assembly() -> cq.Assembly:
    chassis_step = Path("output/main_chassis_v1.step")
    if not chassis_step.exists():
        raise FileNotFoundError(f"Chassis STEP not found: {chassis_step}")

    chassis = cq.importers.importStep(str(chassis_step))
    bolt    = make_m4_bolt()

    assy = cq.Assembly()
    assy.add(chassis, name="main_chassis_v1", color=cq.Color("gray"))

    for i, (x, y) in enumerate(HOLE_XY):
        assy.add(
            bolt,
            name=f"M4_bolt_{i + 1}",
            loc=cq.Location(cq.Vector(x, y, BOLT_Z)),
            color=cq.Color("gray"),   # FIX: "darkgray" is not valid; use "gray"
        )

    return assy


# ---------------------------------------------------------------------------
# Interference check
# ---------------------------------------------------------------------------

def check_interference(assy: cq.Assembly) -> None:
    """Compute and print bolt-vs-chassis intersection volumes."""
    from geometry.cadquery_ext import GeometryService
    geo = GeometryService(output_dir=Path("output"))

    chassis_step = Path("output/main_chassis_v1.step")
    bolt_step    = Path("output/M4_hex_bolt_x4.step")

    if not chassis_step.exists() or not bolt_step.exists():
        print("  [skip] STEP files not found for interference check.")
        return

    vol = geo.check_interference(chassis_step, bolt_step)
    status = "PASS" if vol < 1.0 else f"FAIL ({vol:.1f} mm³)"
    print(f"  Interference chassis ↔ bolts: {status}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("Building chassis + M4 bolt assembly …")
    assy = build_assembly()

    out = Path("output")
    out.mkdir(exist_ok=True)

    # Export bolt compound for interference checking
    bolt_compound = make_m4_bolt()
    bolt_assy = cq.Assembly()
    for i, (x, y) in enumerate(HOLE_XY):
        bolt_assy.add(
            make_m4_bolt(),
            name=f"M4_bolt_{i + 1}",
            loc=cq.Location(cq.Vector(x, y, BOLT_Z)),
        )
    bolt_solid = bolt_assy.toCompound()
    cq.exporters.export(bolt_solid, str(out / "M4_hex_bolt_x4.step"))
    cq.exporters.export(bolt_solid, str(out / "M4_hex_bolt_x4.stl"))
    print(f"  Bolt STEP: {out / 'M4_hex_bolt_x4.step'}")

    # Export full assembly
    assy_path = out / "chassis_bolt_assembly.step"
    assy.save(str(assy_path))
    print(f"  Assembly STEP: {assy_path}")

    print("Interference check:")
    check_interference(assy)

    return 0


if __name__ == "__main__":
    sys.exit(main())
