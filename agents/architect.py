# Architext Agent
"""
ArchitectAgent: parses a natural-language design brief into a structured
DesignSpec and decomposes it into a WorkPlan of DesignTasks.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from core.llm_wrapper import LLMWrapper
from core.schema import (
    BoundaryCondition,
    DesignSpec,
    DesignTask,
    DimensionWithTolerance,
    Load,
    ManufacturingContext,
    ManufacturingProcess,
    WorkPlan,
)
from core.world_model import WorldModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM output
# ---------------------------------------------------------------------------

class _LoadSchema(BaseModel):
    magnitude_n: float
    direction: list[float]
    location_description: str


class _BCSchema(BaseModel):
    description: str


class _DimSchema(BaseModel):
    nominal_mm: float
    plus_mm: float
    minus_mm: float
    description: str


class _DesignSpecSchema(BaseModel):
    components: list[str]
    material: str
    process: str
    loads: list[_LoadSchema]
    boundary_conditions: list[_BCSchema]
    safety_factor: float
    tolerance_critical_dims: list[_DimSchema]
    notes: str


# ---------------------------------------------------------------------------
# ArchitectAgent
# ---------------------------------------------------------------------------

class ArchitectAgent:
    """
    Converts a free-text design brief into a structured WorkPlan.

    Args:
        llm: LLMWrapper instance.
        world_model: WorldModel instance for constraint validation.
    """

    def __init__(self, llm: LLMWrapper, world_model: WorldModel) -> None:
        self._llm = llm
        self._wm = world_model

    def parse_brief(self, brief: str) -> DesignSpec:
        """
        Parse a natural-language design brief into a DesignSpec.

        Uses Claude with structured JSON output to extract all fields.
        Validates material/process compatibility against the WorldModel.

        Args:
            brief: Free-text description of the design task.

        Returns:
            A validated DesignSpec.

        Raises:
            ValueError: If the extracted material/process is unknown or incompatible.
        """
        logger.info("Parsing design brief...")

        raw: _DesignSpecSchema = self._llm.complete_structured(
            "architect_parse.jinja2",
            {
                "brief": brief,
                "materials": self._wm.list_materials(),
                "processes": self._wm.list_processes(),
            },
            _DesignSpecSchema,
        )

        process = ManufacturingProcess(raw.process)

        # Validate material/process compatibility
        ok, reason = self._wm.validate_process_compatibility(raw.material, process)
        if not ok:
            logger.warning("Compatibility issue: %s", reason)

        loads = tuple(
            Load(
                magnitude_n=l.magnitude_n,
                direction=tuple(l.direction),
                location_description=l.location_description,
            )
            for l in raw.loads
        )
        bcs = tuple(BoundaryCondition(description=bc.description) for bc in raw.boundary_conditions)
        dims = tuple(
            DimensionWithTolerance(
                nominal_mm=d.nominal_mm,
                plus_mm=d.plus_mm,
                minus_mm=d.minus_mm,
                description=d.description,
            )
            for d in raw.tolerance_critical_dims
        )

        spec = DesignSpec(
            brief=brief,
            components=tuple(raw.components),
            material=raw.material,
            process=process,
            loads=loads,
            boundary_conditions=bcs,
            safety_factor=raw.safety_factor,
            tolerance_critical_dims=dims,
            notes=raw.notes,
        )

        logger.info(
            "Parsed spec: %d component(s), material=%s, process=%s",
            len(spec.components),
            spec.material,
            spec.process.value,
        )
        return spec

    def decompose(self, spec: DesignSpec) -> WorkPlan:
        """
        Decompose a DesignSpec into individual DesignTasks (one per component).

        Args:
            spec: The validated DesignSpec.

        Returns:
            A WorkPlan with one DesignTask per component.
        """
        tasks = tuple(
            DesignTask(
                component_name=name,
                spec=spec,
                acceptance_criteria=(
                    f"Geometry executes without error",
                    f"All wall thicknesses >= {self._wm.query(spec.material, spec.process).min_wall_thickness_mm} mm",
                    f"No DFM FAIL findings",
                ),
            )
            for name in spec.components
        )
        return WorkPlan(tasks=tasks, mating_constraints=())

    def select_manufacturing_context(self, spec: DesignSpec) -> ManufacturingContext:
        """Return a ManufacturingContext for the given spec."""
        mat_class = self._wm.get_material_class(spec.material)
        return ManufacturingContext(
            process=spec.process,
            machine_description=f"Standard {spec.process.value} setup",
            material_class=mat_class,
            post_processes=(),
        )
