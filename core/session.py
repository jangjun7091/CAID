"""
DesignSession: immutable log of all agent inputs, outputs, and artifact paths
for one CAID design run. Enables rollback, auditability, and resumability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.schema import CritiqueReport, DesignArtifact, DesignSpec, WorkPlan


@dataclass
class ComponentResult:
    """Final state for one component after the Design-Critique-Refine loop."""
    component_name: str
    final_artifact: DesignArtifact
    final_critique: CritiqueReport
    passed: bool

    @property
    def step_path(self) -> Optional[Path]:
        if self.final_artifact.geometry:
            return self.final_artifact.geometry.step_path
        return None

    @property
    def stl_path(self) -> Optional[Path]:
        if self.final_artifact.geometry:
            return self.final_artifact.geometry.stl_path
        return None


@dataclass
class IterationRecord:
    """One Design-Critique pass for a component."""
    component_name: str
    artifact: DesignArtifact
    critique: CritiqueReport
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DesignSession:
    """
    Mutable log built up during an orchestrator run.
    Provides a structured summary and JSON serialization when complete.

    Args:
        brief: The original user brief.
    """

    def __init__(self, brief: str) -> None:
        self.brief = brief
        self.started_at = datetime.now(timezone.utc)
        self.completed_at: Optional[datetime] = None
        self._spec: Optional[DesignSpec] = None
        self._plan: Optional[WorkPlan] = None
        self._iterations: list[IterationRecord] = []
        self._results: dict[str, ComponentResult] = {}

    # ------------------------------------------------------------------
    # Called by orchestrator during the run
    # ------------------------------------------------------------------

    def set_plan(self, spec: DesignSpec, plan: WorkPlan) -> None:
        self._spec = spec
        self._plan = plan

    def add_iteration(
        self,
        component_name: str,
        artifact: DesignArtifact,
        critique: CritiqueReport,
    ) -> None:
        self._iterations.append(IterationRecord(
            component_name=component_name,
            artifact=artifact,
            critique=critique,
        ))

    def finalize_component(
        self,
        component_name: str,
        artifact: DesignArtifact,
        critique: CritiqueReport,
    ) -> None:
        self._results[component_name] = ComponentResult(
            component_name=component_name,
            final_artifact=artifact,
            final_critique=critique,
            passed=critique.passed,
        )

    def complete(self) -> None:
        self.completed_at = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Read accessors
    # ------------------------------------------------------------------

    @property
    def spec(self) -> Optional[DesignSpec]:
        return self._spec

    @property
    def plan(self) -> Optional[WorkPlan]:
        return self._plan

    @property
    def final_artifacts(self) -> dict[str, DesignArtifact]:
        return {name: r.final_artifact for name, r in self._results.items()}

    @property
    def results(self) -> dict[str, ComponentResult]:
        return dict(self._results)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self._results.values())

    @property
    def iterations_for(self) -> dict[str, list[IterationRecord]]:
        out: dict[str, list[IterationRecord]] = {}
        for rec in self._iterations:
            out.setdefault(rec.component_name, []).append(rec)
        return out

    def summary(self) -> str:
        """Return a human-readable summary of the session."""
        lines = [
            f"Brief: {self.brief}",
            f"Started: {self.started_at.isoformat()}",
            f"Completed: {self.completed_at.isoformat() if self.completed_at else 'in progress'}",
            f"Components: {len(self._results)}",
            f"Overall: {'PASSED' if self.all_passed else 'FAILED'}",
            "",
        ]
        for name, result in self._results.items():
            iter_count = len(self.iterations_for.get(name, []))
            status = "PASS" if result.passed else "FAIL"
            step = str(result.step_path) if result.step_path else "no geometry"
            lines.append(f"  [{status}] {name} — {iter_count} iteration(s) — {step}")
            for finding in result.final_critique.findings:
                lines.append(f"         [{finding.severity.value}] {finding.message}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize the session to a JSON-compatible dict."""
        return {
            "brief": self.brief,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "all_passed": self.all_passed,
            "components": {
                name: {
                    "passed": r.passed,
                    "iterations": len(self.iterations_for.get(name, [])),
                    "step_path": str(r.step_path) if r.step_path else None,
                    "stl_path": str(r.stl_path) if r.stl_path else None,
                    "findings": [
                        {
                            "category": f.category.value,
                            "severity": f.severity.value,
                            "message": f.message,
                            "remediation": f.remediation,
                        }
                        for f in r.final_critique.findings
                    ],
                }
                for name, r in self._results.items()
            },
        }

    def save(self, path: Path) -> None:
        """Write the session log to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
