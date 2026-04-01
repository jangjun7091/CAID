"""
AgentOrchestrator: sequences ArchitectAgent -> DesignerAgent -> CriticAgent
in a Design-Critique-Refine loop until all components pass or max iterations
is reached.
"""

from __future__ import annotations

import logging
from pathlib import Path

from agents.architect import ArchitectAgent
from agents.critic import CriticAgent
from agents.designer import DesignerAgent
from core.llm_wrapper import LLMWrapper
from core.schema import CritiqueReport, DesignArtifact, DesignSpec, WorkPlan
from core.session import DesignSession
from core.world_model import WorldModel
from geometry.cadquery_ext import GeometryService

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ITERATIONS = 5


class AgentOrchestrator:
    """
    Drives the full CAID design loop for one user brief.

    Usage:
        orchestrator = AgentOrchestrator.create(api_key="sk-...")
        session = orchestrator.run("Design a mounting bracket for a 200g motor, Al6061, CNC")
        print(session.final_artifacts)

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
    ) -> None:
        self._architect = architect
        self._designer = designer
        self._critic = critic
        self.max_iterations = max_iterations
        self.output_dir = Path(output_dir)

    @classmethod
    def create(
        cls,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        output_dir: Path | str = Path("output"),
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    ) -> "AgentOrchestrator":
        """
        Factory method that wires all dependencies together.

        Args:
            api_key: Anthropic API key (reads ANTHROPIC_API_KEY env var if None).
            model: Claude model ID for generation.
            output_dir: Directory for STEP/STL outputs.
            max_iterations: Max refine cycles per component.
        """
        output_dir = Path(output_dir)
        llm = LLMWrapper(api_key=api_key, model=model)
        world_model = WorldModel()
        geometry = GeometryService(output_dir=output_dir)

        architect = ArchitectAgent(llm=llm, world_model=world_model)
        designer = DesignerAgent(llm=llm, world_model=world_model, geometry=geometry)
        critic = CriticAgent(llm=llm, world_model=world_model, geometry=geometry)

        return cls(
            architect=architect,
            designer=designer,
            critic=critic,
            max_iterations=max_iterations,
            output_dir=output_dir,
        )

    def run(self, brief: str) -> DesignSession:
        """
        Execute the full design loop for the given brief.

        Steps:
          1. ArchitectAgent parses the brief into a DesignSpec + WorkPlan.
          2. For each component in the WorkPlan:
             a. DesignerAgent generates CadQuery code.
             b. CriticAgent reviews the artifact.
             c. If any FAIL findings exist, DesignerAgent refines.
             d. Repeat up to max_iterations times.
          3. Return a DesignSession with all artifacts and reports.

        Args:
            brief: Natural-language design request.

        Returns:
            DesignSession containing final artifacts and critique reports.
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

        # Step 2: Design-Critique-Refine loop per component
        for task in plan.tasks:
            logger.info("=== Designing: %s ===", task.component_name)

            artifact: DesignArtifact = self._designer.generate(task)
            critique: CritiqueReport = self._critic.critique(artifact)
            session.add_iteration(task.component_name, artifact, critique)

            for i in range(1, self.max_iterations):
                if critique.passed:
                    logger.info(
                        "'%s' PASSED all checks at iteration %d.",
                        task.component_name,
                        artifact.iteration,
                    )
                    break

                logger.info(
                    "'%s' has %d FAIL(s) — refining (attempt %d/%d)...",
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

            session.finalize_component(task.component_name, artifact, critique)

        session.complete()
        logger.info("=== CAID: Design complete. ===")
        return session
