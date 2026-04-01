# Review of Manufacturability and Physical Precision
"""
CriticAgent: reviews a DesignArtifact for DFM violations, physics/FEA failures,
and assembly interference. Produces a CritiqueReport with actionable findings.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from pydantic import BaseModel

from core.llm_wrapper import LLMWrapper
from core.schema import (
    Assembly,
    CheckCategory,
    CritiqueReport,
    DesignArtifact,
    Finding,
    Severity,
)
from core.world_model import WorldModel
from geometry.cadquery_ext import GeometryService

logger = logging.getLogger(__name__)

_INTERFERENCE_THRESHOLD_MM3 = 0.01  # volumes below this are floating point noise

# Map known CadQuery error substrings to actionable remediation text.
_CADQUERY_ERROR_REMEDIATIONS: list[tuple[str, str]] = [
    (
        "Selected faces must be co-planar",
        (
            "The call to .workplane() received multiple non-coplanar faces. "
            "Do NOT use '|Z', '|X', or '|Y' face selectors before .workplane() — "
            "they can match several faces at different heights. "
            "Use directional selectors ('>Z', '<Z', '>X', etc.) which always return "
            "a single face, or start a fresh cq.Workplane('XY', origin=(x, y, z)) "
            "with explicit coordinates instead of chaining through a face selector."
        ),
    ),
    (
        "No pending wires present",
        (
            "A .extrude(), .cutBlind(), or similar operation was called with no "
            "pending 2-D wire on the stack. Ensure .circle(), .rect(), .polygon(), "
            "or .polyline() is called before the extrude/cut."
        ),
    ),
    (
        "Workplane must be initialized",
        (
            "The CadQuery Workplane was used before a base plane was set. "
            "Always start the chain with cq.Workplane('XY') (or 'XZ'/'YZ')."
        ),
    ),
    (
        "Edge not found",
        (
            "An edge selector returned no results. Check that the selector string "
            "matches existing edges after all boolean and fillet operations."
        ),
    ),
]


def _execution_remediation(error: str) -> str:
    """Return a specific remediation string for a known CadQuery execution error."""
    for substring, remediation in _CADQUERY_ERROR_REMEDIATIONS:
        if substring in error:
            return remediation
    return (
        "Fix the error shown above. Read the full traceback to identify the "
        "exact line number and CadQuery call that failed, then rewrite that "
        "operation using safe alternatives."
    )


# ---------------------------------------------------------------------------
# Pydantic schema for LLM-returned DFM findings
# ---------------------------------------------------------------------------

class _FindingSchema(BaseModel):
    category: str
    severity: str
    message: str
    remediation: str


# ---------------------------------------------------------------------------
# CriticAgent
# ---------------------------------------------------------------------------

class CriticAgent:
    """
    Runs three independent checks on a DesignArtifact:
      1. DFM check — wall thickness, draft angle, overhangs (LLM-assisted + rules)
      2. Physics check — safety factor vs. yield strength (Phase 2: FEA)
      3. Interference check — geometric Boolean intersection (Phase 2: assembly)

    Args:
        llm: LLMWrapper instance.
        world_model: WorldModel for DFM rule lookup.
        geometry: GeometryService for interference detection.
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

    def critique(
        self,
        artifact: DesignArtifact,
        assembly: Optional[Assembly] = None,
    ) -> CritiqueReport:
        """
        Run all available checks and aggregate into a CritiqueReport.

        Args:
            artifact: The DesignArtifact to review.
            assembly: Optional Assembly for interference checks (Phase 2).

        Returns:
            CritiqueReport with all findings.
        """
        findings: list[Finding] = []

        # Check 1: geometry must have executed successfully
        if artifact.geometry is None or not artifact.geometry.success:
            error = artifact.geometry.error if artifact.geometry else "No geometry result"
            findings.append(Finding(
                category=CheckCategory.DFM,
                severity=Severity.FAIL,
                message=f"Code execution failed: {error}",
                remediation=_execution_remediation(error),
            ))
            return CritiqueReport(artifact=artifact, findings=tuple(findings))

        # Check 2: DFM rules
        findings.extend(self._dfm_check(artifact))

        # Check 3: interference (only when assembly provided)
        if assembly is not None:
            findings.extend(self._interference_check(artifact, assembly))

        # Check 4: physics (placeholder — FEA integrated in Phase 2)
        findings.extend(self._physics_check(artifact))

        passed = sum(1 for f in findings if f.severity == Severity.FAIL)
        logger.info(
            "Critique for '%s': %d finding(s), %d FAIL(s)",
            artifact.component_name,
            len(findings),
            passed,
        )

        return CritiqueReport(artifact=artifact, findings=tuple(findings))

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _dfm_check(self, artifact: DesignArtifact) -> list[Finding]:
        """
        LLM-assisted DFM review using the process rules as grounding context.
        The LLM is given the code and numeric bounds; it identifies violations.
        """
        spec = artifact.params.get("_spec")
        if spec is None:
            return []

        ctx = self._wm.query(spec.material, spec.process)
        process_notes = self._wm.get_process_notes(spec.process)
        geo = artifact.geometry

        raw_text = self._llm.complete(
            "critic_dfm.jinja2",
            {
                "component_name": artifact.component_name,
                "material": spec.material,
                "process": spec.process.value,
                "ctx": ctx,
                "process_notes": process_notes,
                "volume_mm3": geo.volume_mm3,
                "surface_area_mm2": geo.surface_area_mm2,
                "bbox": geo.bounding_box_mm,
                "code": artifact.code,
            },
            fast=True,   # use fast model for critique passes
        )

        return self._parse_findings(raw_text, CheckCategory.DFM)

    def _interference_check(
        self, artifact: DesignArtifact, assembly: Assembly
    ) -> list[Finding]:
        """
        Check interference between this artifact and all others in the assembly.
        Uses geometric Boolean intersection — not an LLM judgment.
        """
        findings: list[Finding] = []

        if artifact.geometry is None or not artifact.geometry.step_path:
            return findings

        for other in assembly.artifacts:
            if other.component_name == artifact.component_name:
                continue
            if other.geometry is None or not other.geometry.step_path:
                continue

            overlap_mm3 = self._geo.check_interference(
                artifact.geometry.step_path,
                other.geometry.step_path,
            )

            if overlap_mm3 > _INTERFERENCE_THRESHOLD_MM3:
                findings.append(Finding(
                    category=CheckCategory.INTERFERENCE,
                    severity=Severity.FAIL,
                    message=(
                        f"'{artifact.component_name}' overlaps '{other.component_name}' "
                        f"by {overlap_mm3:.3f} mm³."
                    ),
                    remediation=(
                        f"Reduce the size of one part or adjust the mating offset "
                        f"to eliminate the {overlap_mm3:.3f} mm³ interference volume."
                    ),
                ))

        return findings

    def _physics_check(self, artifact: DesignArtifact) -> list[Finding]:
        """
        Phase 1: simple analytical safety factor estimate using bounding-box volume
        and applied loads from the spec. Full FEA is integrated in Phase 2.
        """
        spec = artifact.params.get("_spec")
        if spec is None or not spec.loads:
            return []

        ctx = self._wm.query(spec.material, spec.process)
        geo = artifact.geometry

        # Rough cross-sectional area estimate from bounding box (smallest face)
        bb = geo.bounding_box_mm
        min_cross_section_mm2 = min(bb[0] * bb[1], bb[0] * bb[2], bb[1] * bb[2])

        if min_cross_section_mm2 < 1.0:
            return []  # degenerate geometry — caught by DFM check

        total_load_n = sum(l.magnitude_n for l in spec.loads)
        stress_mpa = total_load_n / min_cross_section_mm2  # N/mm² = MPa

        actual_sf = ctx.yield_strength_mpa / stress_mpa if stress_mpa > 0 else float("inf")

        if actual_sf < spec.safety_factor:
            return [Finding(
                category=CheckCategory.PHYSICS,
                severity=Severity.FAIL,
                message=(
                    f"Estimated safety factor {actual_sf:.2f} is below the required "
                    f"{spec.safety_factor:.1f}. Applied load: {total_load_n:.0f} N, "
                    f"estimated cross-section: {min_cross_section_mm2:.1f} mm²."
                ),
                remediation=(
                    f"Increase the minimum cross-sectional area to at least "
                    f"{total_load_n * spec.safety_factor / ctx.yield_strength_mpa:.1f} mm² "
                    f"or reduce the applied load."
                ),
            )]

        return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_findings(raw_text: str, default_category: CheckCategory) -> list[Finding]:
        """Parse a JSON array of findings from an LLM response."""
        raw_text = raw_text.strip()
        # Strip markdown fences if present
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            raw_text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            items = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.warning("Could not parse DFM findings JSON: %s", raw_text[:200])
            return []

        findings = []
        for item in items:
            try:
                schema = _FindingSchema.model_validate(item)
                findings.append(Finding(
                    category=CheckCategory(schema.category),
                    severity=Severity(schema.severity),
                    message=schema.message,
                    remediation=schema.remediation,
                ))
            except Exception as exc:
                logger.warning("Skipping malformed finding: %s — %s", item, exc)

        return findings
