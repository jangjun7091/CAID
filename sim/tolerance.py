# Tolerance Analysis and Interference Check
"""
Tolerance stack-up analysis for linear dimension chains.

Two methods:
- Worst-Case (WC): every dimension simultaneously at its worst extreme.
- Root Sum Square (RSS): statistical model, assuming each tolerance is ±Nσ
  of a normal distribution. Default sigma=3 gives 99.73% confidence.
"""

from __future__ import annotations

import math

from core.schema import DimensionWithTolerance, ToleranceResult


def run_tolerance_stack(
    chain: list[DimensionWithTolerance],
    sigma: float = 3.0,
) -> ToleranceResult:
    """
    Compute worst-case and RSS gap for a linear tolerance chain.

    Each DimensionWithTolerance contributes its nominal_mm to the gap.
    Pass negative nominal values for dimensions that close the gap
    (e.g., a shaft diameter mating into a bore).

    Args:
        chain: Ordered list of dimensions forming the stack-up.
        sigma: Standard deviations for RSS tolerance (default 3 → 99.73%).

    Returns:
        ToleranceResult with worst-case gap, RSS gap, and violation probability.
    """
    if not chain:
        return ToleranceResult(
            worst_case_gap_mm=0.0,
            rss_gap_mm=0.0,
            violation_probability=0.0,
            chain_summary="Empty chain.",
        )

    nominal_gap = sum(d.nominal_mm for d in chain)

    # Worst-case: every tolerance accumulates in the direction that minimises gap
    wc_total_minus = sum(d.minus_mm for d in chain)
    worst_case_gap_mm = nominal_gap - wc_total_minus

    # RSS: treat bilateral tolerance as ±σ where σ = bilateral_half / sigma
    bilateral_variances = [((d.plus_mm + d.minus_mm) / 2.0 / sigma) ** 2 for d in chain]
    rss_one_sigma = math.sqrt(sum(bilateral_variances))
    rss_gap_mm = nominal_gap - sigma * rss_one_sigma

    # Violation probability P(gap < 0) via standard normal CDF
    if rss_one_sigma > 0.0:
        z = nominal_gap / rss_one_sigma
        violation_probability = _normal_cdf(-z)
    else:
        violation_probability = 0.0 if nominal_gap >= 0.0 else 1.0

    lines = [f"Tolerance chain ({len(chain)} links):"]
    for d in chain:
        lines.append(
            f"  {d.description}: {d.nominal_mm:+.3f} mm "
            f"[+{d.plus_mm:.3f} / -{d.minus_mm:.3f}]"
        )
    lines += [
        f"Nominal gap:      {nominal_gap:+.4f} mm",
        f"Worst-case gap:   {worst_case_gap_mm:+.4f} mm",
        f"RSS gap ({sigma:.0f}σ):   {rss_gap_mm:+.4f} mm",
        f"P(interference):  {violation_probability:.4%}",
    ]

    return ToleranceResult(
        worst_case_gap_mm=worst_case_gap_mm,
        rss_gap_mm=rss_gap_mm,
        violation_probability=violation_probability,
        chain_summary="\n".join(lines),
    )


def _normal_cdf(x: float) -> float:
    """CDF of the standard normal distribution."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
