# Code Generation
"""
DesignerAgent: generates and refines parametric CadQuery code for a component.
"""

from __future__ import annotations

import logging

from core.llm_wrapper import LLMWrapper
from core.schema import (
    CritiqueReport,
    DesignArtifact,
    DesignTask,
    GeometryResult,
)
from core.world_model import WorldModel
from geometry.cadquery_ext import GeometryService

logger = logging.getLogger(__name__)

_MAX_CODE_TOKENS = 8192


class DesignerAgent:
    """
    Generates parametric CadQuery code for a single DesignTask and
    refines it based on CriticAgent feedback.

    Args:
        llm: LLMWrapper (or OllamaWrapper) instance.
        world_model: WorldModel for constraint lookup.
        geometry: GeometryService for sandboxed code execution.
    """

    def __init__(
        self,
        llm: LLMWrapper,
        world_model: WorldModel,
        geometry: GeometryService,
    ) -> None:
        self._llm = llm
        self._wm = world_model
        self._geo = geometry

    def generate(self, task: DesignTask) -> DesignArtifact:
        """
        Generate CadQuery code for the component described in task.

        Looks up WorldModel constraints, prompts Claude, executes the code
        in the geometry sandbox, and returns a DesignArtifact.

        Args:
            task: The DesignTask specifying the component.

        Returns:
            DesignArtifact at iteration=0 (first attempt).
        """
        ctx = self._wm.query(task.spec.material, task.spec.process)
        process_notes = self._wm.get_process_notes(task.spec.process)
        mat_class = self._wm.get_material_class(task.spec.material)

        logger.info("Generating code for '%s'...", task.component_name)

        code = self._llm.complete_code(
            "designer_generate.jinja2",
            {
                "component_name": task.component_name,
                "brief": task.spec.brief,
                "material": task.spec.material,
                "material_class": mat_class,
                "process": task.spec.process.value,
                "ctx": ctx,
                "process_notes": process_notes,
                "critique": None,
                "execution_error": None,
            },
            max_tokens=_MAX_CODE_TOKENS,
        )

        return self._execute_and_wrap(code, task, iteration=0)

    def refine(self, artifact: DesignArtifact, critique: CritiqueReport) -> DesignArtifact:
        """
        Refine previously generated code based on CriticAgent findings.

        Args:
            artifact: The DesignArtifact from the previous iteration.
            critique: The CritiqueReport containing FAIL and WARNING findings.

        Returns:
            A new DesignArtifact at artifact.iteration + 1.
        """
        task_name = artifact.component_name
        spec = critique.artifact.params.get("_spec")
        if spec is None:
            raise ValueError("DesignArtifact params must include '_spec' key (set by generate/refine).")

        ctx = self._wm.query(spec.material, spec.process)
        process_notes = self._wm.get_process_notes(spec.process)
        mat_class = self._wm.get_material_class(spec.material)

        logger.info(
            "Refining '%s' (iteration %d -> %d), %d failure(s)...",
            task_name,
            artifact.iteration,
            artifact.iteration + 1,
            len(critique.failures),
        )

        # Format findings for the template
        findings_for_template = [
            {
                "severity": f.severity.value,
                "category": f.category.value,
                "message": f.message,
                "remediation": f.remediation,
            }
            for f in critique.failures + critique.warnings
        ]

        # Surface the raw execution error separately so the template can
        # highlight it above the generic findings list. This gives the LLM
        # the exact exception message rather than a paraphrased version.
        execution_error: str | None = None
        if artifact.geometry is not None and not artifact.geometry.success:
            execution_error = artifact.geometry.error

        code = self._llm.complete_code(
            "designer_generate.jinja2",
            {
                "component_name": task_name,
                "brief": spec.brief,
                "material": spec.material,
                "material_class": mat_class,
                "process": spec.process.value,
                "ctx": ctx,
                "process_notes": process_notes,
                "critique": findings_for_template,
                "execution_error": execution_error,
            },
            max_tokens=_MAX_CODE_TOKENS,
        )

        return self._execute_and_wrap(code, None, iteration=artifact.iteration + 1, spec=spec, component_name=task_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_and_wrap(
        self,
        code: str,
        task: DesignTask | None,
        iteration: int,
        spec=None,
        component_name: str | None = None,
    ) -> DesignArtifact:
        """Execute code in the geometry sandbox and wrap the result."""
        name = component_name or (task.component_name if task else "unknown")
        _spec = spec or (task.spec if task else None)

        geo_result: GeometryResult = self._geo.execute_cadquery(code, name)

        # Compute mass if geometry succeeded
        if geo_result.success and _spec:
            ctx = self._wm.query(_spec.material, _spec.process)
            mass = self._geo.compute_mass(geo_result, ctx.density_g_cm3)
            # GeometryResult is frozen, so rebuild with mass filled in
            geo_result = GeometryResult(
                step_path=geo_result.step_path,
                stl_path=geo_result.stl_path,
                volume_mm3=geo_result.volume_mm3,
                surface_area_mm2=geo_result.surface_area_mm2,
                mass_g=mass,
                bounding_box_mm=geo_result.bounding_box_mm,
                error=None,
            )

        if not geo_result.success:
            logger.warning("Geometry execution failed: %s", geo_result.error)

        return DesignArtifact(
            component_name=name,
            code=code,
            params={"_spec": _spec},
            geometry=geo_result,
            iteration=iteration,
        )
