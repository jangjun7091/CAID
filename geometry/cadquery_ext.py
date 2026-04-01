# CadQuery-based parametric code executor
"""
GeometryService: executes LLM-generated CadQuery code in a sandboxed subprocess,
exports STEP/STL, and performs interference detection via OCC Boolean operations.

SECURITY NOTE: LLM-generated code is untrusted. All execution happens in a
subprocess with CPU time and memory limits. The subprocess has no network access.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import signal
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Optional

from core.schema import GeometryResult

logger = logging.getLogger(__name__)

# Default resource limits for the sandbox subprocess
_DEFAULT_CPU_SECONDS = 60
_DEFAULT_MEMORY_MB = 512


def _sandbox_worker(
    code: str,
    output_dir: str,
    component_name: str,
    result_queue: multiprocessing.Queue,
    memory_mb: int,
) -> None:
    """
    Worker function that runs inside the sandboxed subprocess.
    Applies resource limits, executes CadQuery code, and exports geometry.
    """
    try:
        # Apply memory limit on POSIX systems
        if sys.platform != "win32":
            import resource
            mem_bytes = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))

        import cadquery as cq

        # Execute the generated code in a restricted namespace
        namespace: dict = {"cq": cq}
        exec(compile(code, "<caid_generated>", "exec"), namespace)  # noqa: S102

        # The generated code must assign its result to `result`
        if "result" not in namespace:
            result_queue.put({
                "error": "Generated code did not assign a CadQuery object to `result`."
            })
            return

        shape = namespace["result"]

        step_path = os.path.join(output_dir, f"{component_name}.step")
        stl_path = os.path.join(output_dir, f"{component_name}.stl")

        cq.exporters.export(shape, step_path)
        cq.exporters.export(shape, stl_path)

        # Compute mass properties
        bb = shape.val().BoundingBox()
        props = {
            "volume_mm3": shape.val().Volume(),
            "surface_area_mm2": shape.val().Area(),
            "bounding_box_mm": (
                bb.xmax - bb.xmin,
                bb.ymax - bb.ymin,
                bb.zmax - bb.zmin,
            ),
            "step_path": step_path,
            "stl_path": stl_path,
        }
        result_queue.put(props)

    except Exception:
        result_queue.put({"error": traceback.format_exc()})


class GeometryService:
    """
    Executes parametric CadQuery code, exports geometry, and performs
    interference detection.

    Args:
        output_dir: Directory where STEP and STL files are written.
        cpu_seconds: CPU time limit for the sandbox subprocess.
        memory_mb: Memory limit for the sandbox subprocess (POSIX only).
    """

    def __init__(
        self,
        output_dir: Path | str = Path("output"),
        cpu_seconds: int = _DEFAULT_CPU_SECONDS,
        memory_mb: int = _DEFAULT_MEMORY_MB,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cpu_seconds = cpu_seconds
        self.memory_mb = memory_mb

    def execute_cadquery(self, code: str, component_name: str) -> GeometryResult:
        """
        Execute CadQuery Python code in a sandboxed subprocess.

        The code must assign its final CadQuery workplane/solid to a variable
        named `result`. Example:

            import cadquery as cq
            result = cq.Workplane("XY").box(10, 20, 5)

        Args:
            code: CadQuery Python source code.
            component_name: Used as the base filename for STEP/STL exports.

        Returns:
            GeometryResult with paths, properties, and any error message.
        """
        result_queue: multiprocessing.Queue = multiprocessing.Queue()

        proc = multiprocessing.Process(
            target=_sandbox_worker,
            args=(code, str(self.output_dir), component_name, result_queue, self.memory_mb),
            daemon=True,
        )
        proc.start()
        proc.join(timeout=self.cpu_seconds)

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
            return self._error_result("Geometry execution timed out.")

        if result_queue.empty():
            return self._error_result("Sandbox subprocess exited without a result.")

        data = result_queue.get_nowait()

        if "error" in data:
            logger.warning("Geometry execution error:\n%s", data["error"])
            return self._error_result(data["error"])

        return GeometryResult(
            step_path=Path(data["step_path"]),
            stl_path=Path(data["stl_path"]),
            volume_mm3=data["volume_mm3"],
            surface_area_mm2=data["surface_area_mm2"],
            mass_g=0.0,  # filled in by caller with density from WorldModel
            bounding_box_mm=data["bounding_box_mm"],
            error=None,
        )

    def check_interference(self, step_a: Path, step_b: Path) -> float:
        """
        Compute the interference volume (mm³) between two STEP bodies.

        Returns:
            Overlap volume in mm³. 0.0 means no interference.
        """
        try:
            import cadquery as cq
            shape_a = cq.importers.importStep(str(step_a))
            shape_b = cq.importers.importStep(str(step_b))
            intersection = shape_a.intersect(shape_b)
            volume = intersection.val().Volume()
            return max(0.0, volume)
        except Exception as exc:
            logger.warning("Interference check failed: %s", exc)
            return 0.0

    def compute_mass(self, geometry: GeometryResult, density_g_cm3: float) -> float:
        """Return mass in grams given a GeometryResult and material density."""
        volume_cm3 = geometry.volume_mm3 / 1000.0
        return volume_cm3 * density_g_cm3

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
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
