"""
AgentOrchestrator: sequences ArchitectAgent -> DesignerAgent -> CriticAgent
in a Design-Critique-Refine loop until all components pass or max iterations
is reached.

Phase 2 additions:
  - Multi-component assembly loop: after all components pass individually,
    run a second pass of interference checks across the full assembly.
  - SimService wired into CriticAgent for FEA physics checks.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from agents.architect import ArchitectAgent
from agents.critic import CriticAgent
from agents.designer import DesignerAgent
from core.llm_wrapper import LLMWrapper
from core.ollama_wrapper import OllamaWrapper
from core.schema import (
    Assembly,
    CritiqueReport,
    DesignArtifact,
    DesignSpec,
    PartKind,
    WorkPlan,
)
from core.session import DesignSession
from core.world_model import WorldModel
from geometry.cadquery_ext import GeometryService
from library.repository import PartRecord, PartRepository
from sim.service import SimService

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ITERATIONS = 3


class AgentOrchestrator:
    """
    Drives the full CAID design loop for one user brief.

    Usage:
        orchestrator = AgentOrchestrator.create(api_key="sk-...")
        session = orchestrator.run("Design a 2-part motor housing, Al6061, CNC")
        print(session.summary())

    Args:
        architect: ArchitectAgent instance.
        designer: DesignerAgent instance.
        critic: CriticAgent instance.
        max_iterations: Maximum Design-Critique-Refine cycles per component.
        output_dir: Directory for STEP/STL outputs.
    """

    def __init__(
        self,
        architect: ArchitectAgent,
        designer: DesignerAgent,
        critic: CriticAgent,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
        output_dir: Path = Path("output"),
        repository: Optional[PartRepository] = None,
    ) -> None:
        self._architect = architect
        self._designer = designer
        self._critic = critic
        self.max_iterations = max_iterations
        self.output_dir = Path(output_dir)
        self._repository = repository

    @classmethod
    def create(
        cls,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        output_dir: Path | str = Path("output"),
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
        repository: Optional[PartRepository] = None,
        use_local_llm: bool | None = None,
        ollama_model: str | None = None,
        ollama_base_url: str | None = None,
    ) -> "AgentOrchestrator":
        """
        Factory method that wires all dependencies together.

        Local Ollama is enabled when ``use_local_llm=True`` or when the
        ``CAID_USE_LOCAL_LLM`` env var is set to ``1``/``true``/``yes``.
        Model and base URL fall back to ``LLM_MODEL`` / ``LLM_BASE_URL`` env
        vars so the ``.env`` file is the single place to configure local inference.

        Args:
            api_key: Anthropic API key. Ignored when local LLM is active.
            model: Claude model ID. Ignored when local LLM is active.
            output_dir: Directory for STEP/STL outputs.
            max_iterations: Max refine cycles per component.
            repository: Optional PartRepository; successful parts are auto-saved.
            use_local_llm: Route all LLM calls to a local Ollama server.
                           Defaults to the ``CAID_USE_LOCAL_LLM`` env var.
            ollama_model: Ollama model tag. Defaults to ``LLM_MODEL`` env var,
                          then ``"llama3"``.
            ollama_base_url: Ollama server base URL. Defaults to ``LLM_BASE_URL``
                             env var, then ``"http://localhost:11434/v1"``.
        """
        # Resolve local-LLM flag from argument, then env var
        if use_local_llm is None:
            use_local_llm = os.environ.get("CAID_USE_LOCAL_LLM", "").lower() in (
                "1", "true", "yes"
            )

        # Resolve model / URL from arguments, then dedicated env vars
        resolved_model = ollama_model or os.environ.get("LLM_MODEL", "llama3")
        resolved_url = ollama_base_url or os.environ.get(
            "LLM_BASE_URL", "http://localhost:11434/v1"
        )

        output_dir = Path(output_dir)
        world_model = WorldModel()
        geometry = GeometryService(output_dir=output_dir)
        sim = SimService()

        if use_local_llm:
            llm = OllamaWrapper(model=resolved_model, base_url=resolved_url)
        else:
            llm = LLMWrapper(api_key=api_key, model=model)

        architect = ArchitectAgent(llm=llm, world_model=world_model, repository=repository)
        designer = DesignerAgent(llm=llm, world_model=world_model, geometry=geometry)
        critic = CriticAgent(llm=llm, world_model=world_model, geometry=geometry, sim=sim)

        return cls(
            architect=architect,
            designer=designer,
            critic=critic,
            max_iterations=max_iterations,
            output_dir=output_dir,
            repository=repository,
        )

    def run(self, brief: str) -> DesignSession:
        """
        Execute the full design loop for the given brief.

        Phase 2 flow:
          1. ArchitectAgent parses the brief → DesignSpec + WorkPlan
             (with LLM-extracted mate constraints for multi-part assemblies).
          2. Per-component Design-Critique-Refine loop (individual checks only).
          3. Assembly-level interference pass: build the full Assembly from
             all final artifacts and re-run CriticAgent with the assembly
             context. Refine any component with interference FAILs.

        Args:
            brief: Natural-language design request.

        Returns:
            DesignSession containing all artifacts, critiques, and session log.
        """
        session = DesignSession(brief=brief)

        # Step 1: Parse brief
        logger.info("=== CAID: Parsing brief ===")
        spec: DesignSpec = self._architect.parse_brief(brief)
        plan: WorkPlan = self._architect.decompose(spec)
        session.set_plan(spec, plan)

        logger.info(
            "WorkPlan: %d component(s) — %s",
            len(plan.tasks),
            [t.component_name for t in plan.tasks],
        )
        if plan.mating_constraints:
            logger.info(
                "Mate constraints: %d — %s",
                len(plan.mating_constraints),
                [(c.part_a, c.constraint_type, c.part_b) for c in plan.mating_constraints],
            )

        # Step 2: Per-component Design-Critique-Refine loop
        final_artifacts: dict[str, DesignArtifact] = {}

        for task in plan.tasks:
            artifact, critique = self._component_loop(task, session)
            final_artifacts[task.component_name] = artifact
            session.finalize_component(task.component_name, artifact, critique)
            self._maybe_save_part(artifact, task.spec)

        # Step 3: Assembly-level interference pass (Phase 2)
        if len(final_artifacts) > 1:
            logger.info("=== CAID: Assembly interference pass ===")
            assembly = Assembly(
                artifacts=tuple(final_artifacts.values()),
                constraints=plan.mating_constraints,
            )
            final_artifacts = self._assembly_pass(
                final_artifacts, assembly, plan, session
            )

        session.complete()
        logger.info("=== CAID: Design complete. ===")
        return session

    # ------------------------------------------------------------------
    # Internal loop helpers
    # ------------------------------------------------------------------

    def _maybe_save_part(self, artifact: DesignArtifact, spec: DesignSpec) -> None:
        """Save a successfully designed artifact to the part repository as CUSTOM."""
        if self._repository is None:
            return
        if artifact.geometry is None or not artifact.geometry.success:
            return
        geo = artifact.geometry
        record = PartRecord(
            name=artifact.component_name,
            description=spec.brief[:200],
            kind=PartKind.CUSTOM,
            tags=[spec.material.lower(), spec.process.value.lower(), "custom"],
            parameters={
                "material": spec.material,
                "process": spec.process.value,
                "bounding_box_mm": list(geo.bounding_box_mm),
                "volume_mm3": geo.volume_mm3,
                "mass_g": geo.mass_g,
            },
            cadquery_code=artifact.code,
            step_path=str(geo.step_path) if geo.step_path else None,
            stl_path=str(geo.stl_path) if geo.stl_path else None,
            iso_standard=None,
        )
        part_id = self._repository.save(record)
        logger.info("Saved '%s' to part repository (id=%s).", artifact.component_name, part_id)

    def _component_loop(
        self, task, session: DesignSession
    ) -> tuple[DesignArtifact, CritiqueReport]:
        """Run the Design-Critique-Refine loop for a single component."""
        logger.info("=== Designing: %s ===", task.component_name)

        artifact: DesignArtifact = self._designer.generate(task)
        critique: CritiqueReport = self._critic.critique(artifact)
        session.add_iteration(task.component_name, artifact, critique)

        for i in range(1, self.max_iterations):
            if critique.passed:
                logger.info(
                    "'%s' PASSED at iteration %d.",
                    task.component_name,
                    artifact.iteration,
                )
                break

            logger.info(
                "'%s' %d FAIL(s) — refining (%d/%d)...",
                task.component_name,
                len(critique.failures),
                i,
                self.max_iterations - 1,
            )
            artifact = self._designer.refine(artifact, critique)
            critique = self._critic.critique(artifact)
            session.add_iteration(task.component_name, artifact, critique)
        else:
            if not critique.passed:
                logger.warning(
                    "'%s' still has failures after %d iterations.",
                    task.component_name,
                    self.max_iterations,
                )

        return artifact, critique

    def _assembly_pass(
        self,
        final_artifacts: dict[str, DesignArtifact],
        assembly: Assembly,
        plan: WorkPlan,
        session: DesignSession,
    ) -> dict[str, DesignArtifact]:
        """
        Run interference checks across the full assembly.
        Refine any component that has interference FAILs.
        """
        for task in plan.tasks:
            name = task.component_name
            artifact = final_artifacts[name]

            # Skip components whose geometry failed
            if artifact.geometry is None or not artifact.geometry.success:
                continue

            critique = self._critic.critique(artifact, assembly=assembly)

            interference_fails = [
                f for f in critique.failures
                if f.category.value == "INTERFERENCE"
            ]
            if not interference_fails:
                continue

            logger.info(
                "Assembly interference found in '%s' — refining (%d fail(s))...",
                name,
                len(interference_fails),
            )

            for i in range(1, self.max_iterations):
                artifact = self._designer.refine(artifact, critique)
                # Rebuild assembly with the updated artifact
                updated_arts = {
                    **{k: v for k, v in final_artifacts.items() if k != name},
                    name: artifact,
                }
                assembly = Assembly(
                    artifacts=tuple(updated_arts.values()),
                    constraints=assembly.constraints,
                )
                critique = self._critic.critique(artifact, assembly=assembly)
                session.add_iteration(name, artifact, critique)

                if critique.passed:
                    break
                if not any(
                    f.category.value == "INTERFERENCE" for f in critique.failures
                ):
                    break

            final_artifacts[name] = artifact
            session.finalize_component(name, artifact, critique)

        return final_artifacts
