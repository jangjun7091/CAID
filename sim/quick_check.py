"""
quick_check: Rule-based pre-FEA structural viability filter.

Runs fast geometric checks against WorldModelContext constraints before
committing to a full FEA run. No external dependencies.

Checks:
  1. Minimum feature size  — smallest bounding-box dimension vs min_feature_size_mm
  2. Volume density        — part volume vs bounding-box volume (catches degenerate solids)
  3. Overhang heuristic    — height/footprint aspect ratio for additive processes (FDM, SLS)
"""

from __future__ import annotations

import math

from core.schema import (
    CheckCategory,
    Finding,
    GeometryResult,
    ManufacturingProcess,
    Severity,
    WorldModelContext,
)

# Additive processes where overhang angle matters
_ADDITIVE = {ManufacturingProcess.FDM, ManufacturingProcess.SLS}


def run_quick_check(
    geo: GeometryResult,
    ctx: WorldModelContext,
) -> list[Finding]:
    """
    Run rule-based pre-FEA checks on a GeometryResult.

    Args:
        geo: Output from GeometryService.execute_cadquery().
        ctx: WorldModelContext for the material + process combination.

    Returns:
        List of Finding objects. Empty list means all checks passed cleanly.
        Callers should treat any FAIL severity as a blocker for FEA.
    """
    findings: list[Finding] = []
    findings.extend(_check_minimum_feature(geo, ctx))
    findings.extend(_check_volume_density(geo))
    if ctx.process in _ADDITIVE:
        findings.extend(_check_overhang(geo, ctx))
    return findings


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_minimum_feature(geo: GeometryResult, ctx: WorldModelContext) -> list[Finding]:
    """Flag if the smallest bounding-box dimension is below min_feature_size_mm."""
    min_dim = min(geo.bounding_box_mm)
    threshold = ctx.min_feature_size_mm

    if threshold <= 0:
        return []

    if min_dim < threshold:
        return [Finding(
            category=CheckCategory.DFM,
            severity=Severity.FAIL,
            message=(
                f"Smallest bounding-box dimension {min_dim:.2f} mm is below the "
                f"minimum feature size {threshold:.2f} mm for "
                f"{ctx.process.value} / {ctx.material}."
            ),
            remediation=(
                f"Increase the thinnest dimension to at least {threshold:.2f} mm, "
                "or select a process with a smaller minimum feature size."
            ),
        )]

    if min_dim < threshold * 1.5:
        return [Finding(
            category=CheckCategory.DFM,
            severity=Severity.WARNING,
            message=(
                f"Smallest dimension {min_dim:.2f} mm is only "
                f"{min_dim / threshold:.1f}× the minimum feature size "
                f"({threshold:.2f} mm). May be marginal for manufacturing."
            ),
            remediation="Add 20–30 % margin to the thinnest feature to improve yield.",
        )]

    return []


def _check_volume_density(geo: GeometryResult) -> list[Finding]:
    """
    Detect degenerate solids: part volume should be at least 0.5 % of bounding-box volume.

    Very low density typically indicates a near-zero-thickness shell or a geometry
    error (e.g. extrusion of zero height).
    """
    bb_vol = (
        geo.bounding_box_mm[0]
        * geo.bounding_box_mm[1]
        * geo.bounding_box_mm[2]
    )
    if bb_vol <= 0 or geo.volume_mm3 <= 0:
        return []

    density = geo.volume_mm3 / bb_vol

    if density < 0.005:
        return [Finding(
            category=CheckCategory.DFM,
            severity=Severity.FAIL,
            message=(
                f"Part volume ({geo.volume_mm3:.1f} mm³) is only "
                f"{density * 100:.3f} % of its bounding box "
                f"({bb_vol:.1f} mm³). Likely a degenerate or zero-thickness solid."
            ),
            remediation=(
                "Verify the CadQuery script produces a valid closed solid. "
                "Check for missing extrusions or accidental cut-throughs."
            ),
        )]

    if density < 0.02:
        return [Finding(
            category=CheckCategory.DFM,
            severity=Severity.WARNING,
            message=(
                f"Part volume density is low ({density * 100:.1f} % of bounding box). "
                "The part may be structurally marginal."
            ),
            remediation=(
                "Review thin walls and ensure the part meets load-bearing requirements."
            ),
        )]

    return []


def _check_overhang(geo: GeometryResult, ctx: WorldModelContext) -> list[Finding]:
    """
    Heuristic overhang check for additive processes (FDM / SLS).

    Uses the build-height-to-footprint aspect ratio as a proxy for overhang risk.
    A proper check would analyse per-face normals; this is a fast pre-FEA filter.

    Logic: if the build height is so large relative to the minimum footprint side
    that a feature at the top of the part could only be supported by material
    steeper than max_overhang_deg, flag it.
    """
    bb_x, bb_y, bb_z = geo.bounding_box_mm
    max_angle = ctx.max_overhang_deg

    if bb_z <= 0 or max_angle <= 0 or max_angle >= 90:
        return []

    min_side = min(bb_x, bb_y)
    if min_side <= 0:
        return []

    # Maximum unsupported horizontal reach at the given overhang angle
    max_reach = bb_z * math.tan(math.radians(90.0 - max_angle))

    if min_side < max_reach * 0.3:
        return [Finding(
            category=CheckCategory.DFM,
            severity=Severity.WARNING,
            message=(
                f"Part aspect ratio (H={bb_z:.1f} mm, "
                f"min footprint side={min_side:.1f} mm) suggests potential "
                f"overhangs beyond {max_angle:.0f}° for {ctx.process.value}."
            ),
            remediation=(
                f"Add support structures, chamfer overhanging features to ≤{max_angle:.0f}°, "
                "or reorient the part on the build plate."
            ),
        )]

    return []
