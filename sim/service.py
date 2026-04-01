"""
SimService: unified facade over FEA and tolerance stack-up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.schema import (
    BoundaryCondition,
    DimensionWithTolerance,
    FEAResult,
    Load,
    ToleranceResult,
    WorldModelContext,
)
from sim.fea_engine import run_fea
from sim.tolerance import run_tolerance_stack


class SimService:
    """
    Single entry point for all simulation operations.

    Args:
        work_dir: Scratch directory for FEA temp files.
                  Uses system temp dir if None.
    """

    def __init__(self, work_dir: Optional[Path] = None) -> None:
        self.work_dir = work_dir

    def run_fea(
        self,
        step_path: Path,
        ctx: WorldModelContext,
        loads: list[Load],
        boundary_conditions: list[BoundaryCondition],
        required_safety_factor: float = 2.0,
    ) -> FEAResult:
        """
        Mesh the STEP file, run CalculiX, and return stress/displacement results.

        Falls back to an analytical estimate if CalculiX is not on PATH.
        """
        return run_fea(
            step_path=step_path,
            ctx=ctx,
            loads=loads,
            boundary_conditions=boundary_conditions,
            required_safety_factor=required_safety_factor,
            work_dir=self.work_dir,
        )

    def run_tolerance_stack(
        self,
        chain: list[DimensionWithTolerance],
        sigma: float = 3.0,
    ) -> ToleranceResult:
        """
        Compute worst-case and RSS gap for a linear tolerance chain.
        """
        return run_tolerance_stack(chain=chain, sigma=sigma)
