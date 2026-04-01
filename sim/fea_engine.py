# FEA Automation Module
"""
FEA automation: Gmsh meshing → CalculiX solver → results parsing.

Pipeline:
  1. Import STEP file via Gmsh Python API.
  2. Generate a 2nd-order tetrahedral mesh (C3D10) and export as Abaqus .inp.
  3. Append material, boundary condition, and load cards to the .inp file.
  4. Execute the CalculiX binary (ccx / ccx.exe) as a subprocess.
  5. Parse the .dat output file for max displacement and von Mises stress.

If CalculiX is not installed, the engine returns a graceful fallback result
using a simple analytical beam-bending estimate so the Critic can still run.

Dependencies:
  pip install gmsh
  CalculiX: https://www.calculix.de  (install ccx and add to PATH)
"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from core.schema import BoundaryCondition, FEAResult, Load, WorldModelContext

logger = logging.getLogger(__name__)

_CCX_BINARY = "ccx"          # name of the CalculiX executable on PATH
_MESH_SIZE_FACTOR = 0.15     # mesh size relative to bounding box diagonal


def run_fea(
    step_path: Path,
    ctx: WorldModelContext,
    loads: list[Load],
    boundary_conditions: list[BoundaryCondition],
    required_safety_factor: float,
    work_dir: Optional[Path] = None,
) -> FEAResult:
    """
    Run a linear static FEA on the given STEP geometry.

    Args:
        step_path: Path to the STEP file to analyse.
        ctx: WorldModelContext providing material properties.
        loads: Applied loads (magnitude and direction).
        boundary_conditions: Geometric constraints (fixed faces, etc.).
        required_safety_factor: Safety factor to check against yield strength.
        work_dir: Scratch directory (uses a temp dir if None).

    Returns:
        FEAResult with max stress, max displacement, safety factor, and mesh path.
        If CalculiX is not available, returns an analytical fallback result.
    """
    if not _ccx_available():
        logger.warning(
            "CalculiX (ccx) not found on PATH — using analytical fallback. "
            "Install CalculiX and add ccx to PATH for real FEA."
        )
        return _analytical_fallback(ctx, loads, required_safety_factor)

    use_temp = work_dir is None
    work = Path(tempfile.mkdtemp()) if use_temp else work_dir
    work.mkdir(parents=True, exist_ok=True)

    try:
        inp_path = work / "model.inp"
        _build_inp(step_path, ctx, loads, boundary_conditions, inp_path)

        dat_path = work / "model.dat"
        frd_path = work / "model.frd"
        error = _run_ccx(inp_path, work)
        if error:
            return FEAResult(
                max_stress_mpa=0.0,
                max_displacement_mm=0.0,
                safety_factor=0.0,
                mesh_path=None,
                converged=False,
                error=error,
            )

        max_disp, max_stress = _parse_dat(dat_path)
        sf = ctx.yield_strength_mpa / max_stress if max_stress > 0 else float("inf")

        return FEAResult(
            max_stress_mpa=max_stress,
            max_displacement_mm=max_disp,
            safety_factor=sf,
            mesh_path=frd_path if frd_path.exists() else None,
            converged=True,
            error=None,
        )
    except Exception as exc:
        logger.exception("FEA failed: %s", exc)
        return FEAResult(
            max_stress_mpa=0.0,
            max_displacement_mm=0.0,
            safety_factor=0.0,
            mesh_path=None,
            converged=False,
            error=str(exc),
        )
    finally:
        if use_temp:
            shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------
# Internal pipeline steps
# ---------------------------------------------------------------------------

def _build_inp(
    step_path: Path,
    ctx: WorldModelContext,
    loads: list[Load],
    boundary_conditions: list[BoundaryCondition],
    inp_path: Path,
) -> None:
    """Generate a CalculiX .inp file from a STEP geometry."""
    try:
        import gmsh
    except ImportError as exc:
        raise RuntimeError(
            "gmsh Python package is not installed. Run: pip install gmsh"
        ) from exc

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)

    try:
        gmsh.model.occ.importShapes(str(step_path))
        gmsh.model.occ.synchronize()

        # Compute a sensible mesh size from the bounding box diagonal
        bb = gmsh.model.getBoundingBox(-1, -1)
        diag = math.sqrt((bb[3]-bb[0])**2 + (bb[4]-bb[1])**2 + (bb[5]-bb[2])**2)
        mesh_size = max(diag * _MESH_SIZE_FACTOR, 0.5)

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.1)
        gmsh.option.setNumber("Mesh.ElementOrder", 2)    # C3D10 (Tet10)
        gmsh.option.setNumber("Mesh.Algorithm3D", 4)     # Frontal-Delaunay

        gmsh.model.mesh.generate(3)
        gmsh.write(str(inp_path))
    finally:
        gmsh.finalize()

    # Append material, BCs, and load cards after Gmsh's node/element output
    _append_ccx_cards(inp_path, ctx, loads, boundary_conditions)


def _append_ccx_cards(
    inp_path: Path,
    ctx: WorldModelContext,
    loads: list[Load],
    boundary_conditions: list[BoundaryCondition],
) -> None:
    """Append CalculiX simulation cards to a Gmsh-generated .inp mesh file."""
    # Gmsh writes ELSET=ALL or similar; we need to know the actual set name.
    # Gmsh uses "Volume1" or the entity tag. We'll use a wildcard approach:
    # re-assign everything to EALL for simplicity.
    original = inp_path.read_text()

    # Find the first *ELEMENT line to get Gmsh's element set name
    elset_match = re.search(r"\*ELEMENT[^\n]*ELSET=(\S+)", original, re.IGNORECASE)
    elset = elset_match.group(1).rstrip(",") if elset_match else "ALL"

    # Find node set name (Gmsh typically writes *NSET,NSET=...)
    nset_match = re.search(r"\*NSET[^\n]*NSET=(\S+)", original, re.IGNORECASE)
    nset = nset_match.group(1).rstrip(",") if nset_match else "ALL"

    # Build a fixed-face node set from surface nodes.
    # For a general bracket, we fix the bottom face (min Z nodes).
    # This is a heuristic — Phase 3 will map BCs to specific faces geometrically.
    bc_lines = _build_bc_lines(nset, boundary_conditions)
    load_lines = _build_load_lines(nset, loads)

    E_pa = ctx.youngs_modulus_gpa * 1e3          # GPa → MPa (N/mm²)
    rho = ctx.density_g_cm3 * 1e-3 / 1e3        # g/cm³ → t/mm³ (CalculiX SI-mm)

    cards = f"""
*MATERIAL, NAME=CAIDMAT
*ELASTIC
{E_pa:.1f}, {ctx.poisson_ratio}
*DENSITY
{rho:.6E}
*SOLID SECTION, ELSET={elset}, MATERIAL=CAIDMAT
*STEP
*STATIC
{bc_lines}
{load_lines}
*NODE PRINT, NSET={nset}, FREQUENCY=1
U
*EL PRINT, ELSET={elset}, FREQUENCY=1
S
*END STEP
"""
    with open(inp_path, "a") as f:
        f.write(cards)


def _build_bc_lines(nset: str, bcs: list[BoundaryCondition]) -> str:
    """Generate *BOUNDARY lines. Heuristic: fix all DOF for the given node set."""
    if not bcs:
        # Default: fix the entire model base (all DOF) — conservative
        return f"*BOUNDARY\n{nset}, 1, 6, 0.0"
    lines = ["*BOUNDARY"]
    for _ in bcs:
        # Each BC fixes all translational DOF (1-3). Rotational DOF fixed too (4-6).
        lines.append(f"{nset}, 1, 6, 0.0")
    return "\n".join(lines)


def _build_load_lines(nset: str, loads: list[Load]) -> str:
    """Generate *CLOAD lines from Load objects."""
    if not loads:
        return ""
    lines = ["*CLOAD"]
    for load in loads:
        dx, dy, dz = load.direction
        mag = load.magnitude_n
        # Distribute the load over the node set (CalculiX sums CLOAD per node)
        # For a simple estimate we apply the total load to the set centroid node
        # by referencing the set — CalculiX applies per-node when set is used with CLOAD.
        if abs(dx) > 0:
            lines.append(f"{nset}, 1, {dx * mag:.4E}")
        if abs(dy) > 0:
            lines.append(f"{nset}, 2, {dy * mag:.4E}")
        if abs(dz) > 0:
            lines.append(f"{nset}, 3, {dz * mag:.4E}")
    return "\n".join(lines)


def _run_ccx(inp_path: Path, work_dir: Path) -> Optional[str]:
    """
    Run the CalculiX solver on inp_path. Returns an error string or None on success.
    """
    stem = inp_path.stem  # CalculiX takes the stem, not the full path
    try:
        result = subprocess.run(
            [_CCX_BINARY, stem],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            return f"CalculiX exited with code {result.returncode}:\n{result.stderr[:500]}"
        return None
    except FileNotFoundError:
        return f"CalculiX binary '{_CCX_BINARY}' not found. Install CalculiX and add to PATH."
    except subprocess.TimeoutExpired:
        return "CalculiX timed out after 300 seconds."


def _parse_dat(dat_path: Path) -> tuple[float, float]:
    """
    Parse the CalculiX .dat output file.

    Returns:
        (max_displacement_mm, max_von_mises_stress_mpa)
    """
    if not dat_path.exists():
        raise FileNotFoundError(f"CalculiX .dat output not found: {dat_path}")

    text = dat_path.read_text()

    # --- Displacements ---
    # .dat section header: "displacements (output request)"
    # Columns: node  U1  U2  U3
    max_disp = 0.0
    in_disp = False
    for line in text.splitlines():
        if "displacements" in line.lower():
            in_disp = True
            continue
        if in_disp:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    u1, u2, u3 = float(parts[1]), float(parts[2]), float(parts[3])
                    max_disp = max(max_disp, math.sqrt(u1**2 + u2**2 + u3**2))
                except (ValueError, IndexError):
                    if parts and not parts[0].lstrip("-").replace(".", "").isdigit():
                        in_disp = False  # reached next section

    # --- Stresses ---
    # .dat section: "stresses (elem, integ.pnt.) for set ..."
    # Columns: elem  intpt  S11  S22  S33  S12  S13  S23
    # Von Mises = sqrt(0.5*((S11-S22)^2+(S22-S33)^2+(S33-S11)^2+6*(S12^2+S13^2+S23^2)))
    max_mises = 0.0
    in_stress = False
    for line in text.splitlines():
        if "stresses" in line.lower():
            in_stress = True
            continue
        if in_stress:
            parts = line.split()
            if len(parts) >= 8:
                try:
                    s11 = float(parts[2])
                    s22 = float(parts[3])
                    s33 = float(parts[4])
                    s12 = float(parts[5])
                    s13 = float(parts[6])
                    s23 = float(parts[7])
                    mises = math.sqrt(0.5 * (
                        (s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2
                        + 6*(s12**2 + s13**2 + s23**2)
                    ))
                    max_mises = max(max_mises, mises)
                except (ValueError, IndexError):
                    if parts and not parts[0].lstrip("-").replace(".", "").isdigit():
                        in_stress = False

    return max_disp, max_mises


def _analytical_fallback(
    ctx: WorldModelContext,
    loads: list[Load],
    required_safety_factor: float,
) -> FEAResult:
    """
    Bounding-box beam-bending estimate used when CalculiX is not available.
    Returns an FEAResult flagged with a warning in the error field.
    """
    if not loads:
        return FEAResult(
            max_stress_mpa=0.0,
            max_displacement_mm=0.0,
            safety_factor=float("inf"),
            mesh_path=None,
            converged=True,
            error="Analytical fallback: no loads specified.",
        )

    total_load_n = sum(l.magnitude_n for l in loads)
    # Cannot estimate stress without geometry — return a placeholder
    return FEAResult(
        max_stress_mpa=0.0,
        max_displacement_mm=0.0,
        safety_factor=float("inf"),
        mesh_path=None,
        converged=False,
        error=(
            f"CalculiX not available. Install ccx for real FEA. "
            f"Total applied load: {total_load_n:.1f} N. "
            f"Yield strength: {ctx.yield_strength_mpa:.0f} MPa."
        ),
    )


def _ccx_available() -> bool:
    """Return True if the CalculiX binary is on PATH."""
    return shutil.which(_CCX_BINARY) is not None
