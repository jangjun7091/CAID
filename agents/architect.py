# Architext Agent
"""
ArchitectAgent: parses a natural-language design brief into a structured
DesignSpec and decomposes it into a WorkPlan of DesignTasks.

Phase 3 addition: before decomposing, searches the part library for existing
parts that match each component. Matches are appended to DesignSpec.notes so
the DesignerAgent can reuse or adapt them instead of generating from scratch.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Optional

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
    MateConstraint,
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


class _MateConstraintSchema(BaseModel):
    part_a: str
    part_b: str
    constraint_type: str
    description: str


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

    def __init__(
        self,
        llm: LLMWrapper,
        world_model: WorldModel,
        repository=None,   # Optional[PartRepository] — imported lazily to avoid circular deps
    ) -> None:
        self._llm = llm
        self._wm = world_model
        self._repository = repository

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
        Decompose a DesignSpec into individual DesignTasks and extract
        assembly mate constraints via LLM (for multi-component assemblies).

        Args:
            spec: The validated DesignSpec.

        Returns:
            A WorkPlan with one DesignTask per component and mate constraints.
        """
        ctx = self._wm.query(spec.material, spec.process)

        # Phase 3: search for reusable library parts before generating
        spec = self._inject_library_hints(spec)

        tasks = tuple(
            DesignTask(
                component_name=name,
                spec=spec,
                acceptance_criteria=(
                    "Geometry executes without error",
                    f"All wall thicknesses >= {ctx.min_wall_thickness_mm} mm",
                    "No DFM FAIL findings",
                ),
            )
            for name in spec.components
        )

        mate_constraints: tuple[MateConstraint, ...]
        if len(spec.components) > 1:
            mate_constraints = self._extract_mate_constraints(spec)
        else:
            mate_constraints = ()

        return WorkPlan(tasks=tasks, mating_constraints=mate_constraints)

    def search_library(self, query: str, top_k: int = 3) -> list:
        """
        Search the part library for existing parts matching a query string.

        Returns a list of (PartRecord, score) pairs, or [] if no repository
        is configured or the library is empty.
        """
        if self._repository is None:
            return []
        from library.search import PartSearchIndex
        index = PartSearchIndex(self._repository)
        return index.search(query, top_k=top_k)

    def _inject_library_hints(self, spec: DesignSpec) -> DesignSpec:
        """
        Search the part library for each component in the spec.

        If any matches are found, the top candidates are appended to
        DesignSpec.notes so the DesignerAgent can reference or adapt them.
        Returns a new DesignSpec (frozen dataclass replacement); unchanged if
        no repository is configured or no matches found.
        """
        if self._repository is None:
            return spec

        hints: list[str] = []
        for name in spec.components:
            query = f"{name} {spec.material} {spec.process.value}"
            matches = self.search_library(query, top_k=2)
            if matches:
                parts_text = "; ".join(
                    f"'{rec.name}' (score={score:.2f}): {rec.description[:80]}"
                    for rec, score in matches
                )
                hints.append(f"Library matches for '{name}': {parts_text}")

        if not hints:
            return spec

        extra = "\n\nPart library suggestions (reuse or adapt if appropriate):\n" + "\n".join(hints)
        updated_notes = spec.notes + extra
        logger.info("Injected %d library hint(s) into DesignSpec.notes.", len(hints))
        return dataclasses.replace(spec, notes=updated_notes)

    def _extract_mate_constraints(self, spec: DesignSpec) -> tuple[MateConstraint, ...]:
        """
        Ask Claude to identify how the components in the spec mate together.

        Returns a tuple of MateConstraint objects. Falls back to an empty
        tuple if the LLM returns unparseable output.
        """
        from pydantic import TypeAdapter

        logger.info("Extracting assembly mate constraints for %d components...", len(spec.components))

        try:
            adapter = TypeAdapter(list[_MateConstraintSchema])
            raw_text = self._llm.complete(
                "architect_assembly.jinja2",
                {
                    "brief": spec.brief,
                    "components": list(spec.components),
                },
                system="Respond with valid JSON only. No markdown fences.",
                fast=True,
            )
            raw_text = raw_text.strip()
            if raw_text.startswith("```"):
                lines = raw_text.splitlines()
                raw_text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            import json
            items = json.loads(raw_text)
            schemas = adapter.validate_python(items)

            constraints = tuple(
                MateConstraint(
                    part_a=s.part_a,
                    part_b=s.part_b,
                    constraint_type=s.constraint_type,
                    description=s.description,
                )
                for s in schemas
                if s.part_a in spec.components and s.part_b in spec.components
            )
            logger.info("Extracted %d mate constraint(s).", len(constraints))
            return constraints

        except Exception as exc:
            logger.warning("Could not extract mate constraints: %s", exc)
            return ()

    def select_manufacturing_context(self, spec: DesignSpec) -> ManufacturingContext:
        """Return a ManufacturingContext for the given spec."""
        mat_class = self._wm.get_material_class(spec.material)
        return ManufacturingContext(
            process=spec.process,
            machine_description=f"Standard {spec.process.value} setup",
            material_class=mat_class,
            post_processes=(),
        )
