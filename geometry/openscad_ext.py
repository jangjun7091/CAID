# Integration with the OpenSCAD Engine
"""
OpenSCAD executor: runs LLM-generated .scad code in a subprocess,
exports to STL and converts to STEP via CadQuery's OCC kernel.

OpenSCAD is an alternative code target for simpler geometry or users
who prefer its CSG syntax. All outputs are normalised to STEP so the
rest of the CAID pipeline (interference checks, FEA) is unaffected.

Requires:
  - openscad executable on PATH (https://openscad.org/downloads.html)
  - cadquery installed (for STL → STEP conversion)
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Optional

from core.schema import GeometryResult

logger = logging.getLogger(__name__)

_OPENSCAD_BINARY = "openscad"
_TIMEOUT_S = 120


def execute_openscad(code: str, component_name: str, output_dir: Path) -> GeometryResult:
    """
    Write .scad code to a temp file, render it to STL via OpenSCAD,
    then convert the STL to STEP using CadQuery's import utilities.

    The .scad code must produce a valid solid when rendered. It may use
    any OpenSCAD built-in modules.

    Args:
        code: OpenSCAD (.scad) source code.
        component_name: Base filename for outputs.
        output_dir: Directory to write STL and STEP files into.

    Returns:
        GeometryResult with paths and properties, or an error description.
    """
    if not _openscad_available():
        return _error_result(
            "OpenSCAD binary not found on PATH. "
            "Install OpenSCAD from https://openscad.org/downloads.html"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    stl_path = output_dir / f"{component_name}.stl"
    step_path = output_dir / f"{component_name}_from_scad.step"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".scad", delete=False, encoding="utf-8"
    ) as scad_file:
        scad_file.write(code)
        scad_path = Path(scad_file.name)

    try:
        # Step 1: render .scad → .stl
        error = _run_openscad(scad_path, stl_path)
        if error:
            return _error_result(error)

        # Step 2: convert .stl → .step via CadQuery
        error = _stl_to_step(stl_path, step_path)
        if error:
            # Return a partial result — STL exists but STEP conversion failed
            logger.warning("STL→STEP conversion failed: %s", error)
            return _error_result(
                f"OpenSCAD rendered successfully but STEP conversion failed: {error}"
            )

        # Step 3: compute geometry properties from the STEP file
        return _compute_properties(step_path, stl_path)

    finally:
        scad_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_openscad(scad_path: Path, stl_path: Path) -> Optional[str]:
    """Render a .scad file to .stl. Returns an error string or None on success."""
    try:
        result = subprocess.run(
            [_OPENSCAD_BINARY, "-o", str(stl_path), str(scad_path)],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
        )
        if result.returncode != 0:
            return (
                f"OpenSCAD exited with code {result.returncode}:\n"
                f"{result.stderr[:500]}"
            )
        if not stl_path.exists() or stl_path.stat().st_size == 0:
            return "OpenSCAD produced an empty or missing STL file."
        return None
    except subprocess.TimeoutExpired:
        return f"OpenSCAD timed out after {_TIMEOUT_S} seconds."
    except FileNotFoundError:
        return f"OpenSCAD binary '{_OPENSCAD_BINARY}' not found."


def _stl_to_step(stl_path: Path, step_path: Path) -> Optional[str]:
    """
    Convert STL to STEP using CadQuery's import utilities.

    CadQuery can import STL as a shell and convert to a solid via OCC.
    Note: the resulting STEP is a faceted approximation, not a true B-rep.
    For DFM and interference purposes this is sufficient.
    """
    try:
        import cadquery as cq

        # importers.importStep expects a STEP; we use the Shape import for STL
        shape = cq.Shape.importStl(str(stl_path))
        # Wrap in a Workplane and export to STEP
        result = cq.Workplane().newObject([shape])
        cq.exporters.export(result, str(step_path))
        return None
    except Exception:
        return traceback.format_exc()


def _compute_properties(step_path: Path, stl_path: Path) -> GeometryResult:
    """Compute bounding box and volume from a STEP file."""
    try:
        import cadquery as cq

        shape = cq.importers.importStep(str(step_path))
        bb = shape.val().BoundingBox()
        return GeometryResult(
            step_path=step_path,
            stl_path=stl_path,
            volume_mm3=shape.val().Volume(),
            surface_area_mm2=shape.val().Area(),
            mass_g=0.0,  # filled by caller
            bounding_box_mm=(
                bb.xmax - bb.xmin,
                bb.ymax - bb.ymin,
                bb.zmax - bb.zmin,
            ),
            error=None,
        )
    except Exception:
        return _error_result(
            f"Could not compute geometry properties: {traceback.format_exc()}"
        )


def _error_result(error: str) -> GeometryResult:
    return GeometryResult(
        step_path=None,
        stl_path=None,
        volume_mm3=0.0,
        surface_area_mm2=0.0,
        mass_g=0.0,
        bounding_box_mm=(0.0, 0.0, 0.0),
        error=error,
    )


def _openscad_available() -> bool:
    return shutil.which(_OPENSCAD_BINARY) is not None
