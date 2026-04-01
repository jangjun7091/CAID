# Physics-informed research space for deformation models
"""
WorldModel: deterministic, curated knowledge base of material properties
and manufacturing process DFM rules.

This is NOT an LLM. All constraint values come from versioned YAML files,
providing a grounding layer that prevents LLM hallucination of invalid limits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from core.schema import ManufacturingProcess, WorldModelContext

_DATA_DIR = Path(__file__).parent.parent / "data" / "world_model"


class WorldModel:
    """
    Loads material and process rule tables from YAML and answers constraint queries.

    Usage:
        wm = WorldModel()
        ctx = wm.query("Al6061", ManufacturingProcess.CNC)
        print(ctx.min_wall_thickness_mm)  # -> 0.8
    """

    def __init__(self, data_dir: Path = _DATA_DIR) -> None:
        with open(data_dir / "materials.yaml") as f:
            self._materials: dict = yaml.safe_load(f)
        with open(data_dir / "process_rules.yaml") as f:
            self._processes: dict = yaml.safe_load(f)

    def query(self, material: str, process: ManufacturingProcess) -> WorldModelContext:
        """
        Return all constraint bounds for a material + process combination.

        Args:
            material: Material key (e.g., "Al6061", "PA12").
            process: Manufacturing process enum value.

        Returns:
            WorldModelContext with all numeric bounds populated.

        Raises:
            KeyError: If the material or process is not in the knowledge base.
        """
        mat = self._get_material(material)
        proc = self._get_process(process)

        return WorldModelContext(
            material=material,
            process=process,
            min_wall_thickness_mm=proc["min_wall_thickness_mm"],
            min_feature_size_mm=proc["min_feature_size_mm"],
            draft_angle_deg=proc["draft_angle_deg"],
            max_overhang_deg=proc["max_overhang_deg"],
            typical_surface_roughness_ra=proc["typical_surface_roughness_ra"],
            youngs_modulus_gpa=mat["youngs_modulus_gpa"],
            yield_strength_mpa=mat["yield_strength_mpa"],
            density_g_cm3=mat["density_g_cm3"],
            poisson_ratio=mat["poisson_ratio"],
        )

    def list_materials(self) -> list[str]:
        """Return all material keys in the knowledge base."""
        return list(self._materials.keys())

    def list_processes(self) -> list[str]:
        """Return all process keys in the knowledge base."""
        return list(self._processes.keys())

    def get_process_notes(self, process: ManufacturingProcess) -> str:
        """Return human-readable DFM notes for a process."""
        return self._get_process(process).get("notes", "")

    def get_material_class(self, material: str) -> str:
        """Return the material class string (e.g., 'aluminum_alloy')."""
        return self._get_material(material)["class"]

    def validate_process_compatibility(
        self, material: str, process: ManufacturingProcess
    ) -> tuple[bool, str]:
        """
        Check if a material is compatible with a manufacturing process.

        Returns:
            (is_compatible, reason_if_not)
        """
        mat = self._get_material(material)
        compatible = mat.get("compatible_processes", [])
        if process.value in compatible:
            return True, ""
        return (
            False,
            f"{material} is not compatible with {process.value}. "
            f"Compatible processes: {compatible}",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_material(self, material: str) -> dict:
        if material not in self._materials:
            available = list(self._materials.keys())
            raise KeyError(
                f"Unknown material '{material}'. Available: {available}"
            )
        return self._materials[material]

    def _get_process(self, process: ManufacturingProcess) -> dict:
        key = process.value
        if key not in self._processes:
            available = list(self._processes.keys())
            raise KeyError(
                f"Unknown process '{key}'. Available: {available}"
            )
        return self._processes[key]
