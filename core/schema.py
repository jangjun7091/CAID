"""
Shared dataclasses for CAID. All agents communicate exclusively through these types.
All dataclasses are frozen (immutable) to prevent accidental state mutation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


class CheckCategory(str, Enum):
    DFM = "DFM"
    PHYSICS = "PHYSICS"
    INTERFERENCE = "INTERFERENCE"


class ManufacturingProcess(str, Enum):
    FDM = "FDM"
    SLS = "SLS"
    CNC = "CNC"
    INJECTION_MOLDING = "INJECTION_MOLDING"
    SHEET_METAL = "SHEET_METAL"


class PartKind(str, Enum):
    CUSTOM = "CUSTOM"      # AI-generated, project-specific
    STANDARD = "STANDARD"  # ISO standard catalog part


# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Load:
    """A mechanical load applied to a component."""
    magnitude_n: float          # Newtons
    direction: tuple[float, float, float]  # unit vector (x, y, z)
    location_description: str   # human-readable, e.g. "top face centroid"


@dataclass(frozen=True)
class BoundaryCondition:
    """A fixed constraint applied to geometry."""
    description: str            # e.g. "bottom face fixed in all DOF"


@dataclass(frozen=True)
class DimensionWithTolerance:
    """One link in a tolerance stack-up chain."""
    nominal_mm: float
    plus_mm: float
    minus_mm: float
    description: str


@dataclass(frozen=True)
class MateConstraint:
    """How two components are assembled together."""
    part_a: str                 # component name
    part_b: str                 # component name
    constraint_type: str        # e.g. "planar", "cylindrical", "fixed"
    description: str


# ---------------------------------------------------------------------------
# Knowledge / context types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorldModelContext:
    """Numeric bounds returned by WorldModel.query() for a material+process."""
    material: str
    process: ManufacturingProcess
    min_wall_thickness_mm: float
    min_feature_size_mm: float
    draft_angle_deg: float
    max_overhang_deg: float          # for additive processes
    typical_surface_roughness_ra: float   # µm
    youngs_modulus_gpa: float
    yield_strength_mpa: float
    density_g_cm3: float
    poisson_ratio: float


@dataclass(frozen=True)
class ManufacturingContext:
    """Selected process and machine constraints for this design."""
    process: ManufacturingProcess
    machine_description: str
    material_class: str              # e.g. "aluminum alloy", "thermoplastic"
    post_processes: tuple[str, ...]  # e.g. ("anodizing", "thread tapping")


# ---------------------------------------------------------------------------
# Design planning types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DesignSpec:
    """
    Structured design intent parsed from a natural language brief by ArchitectAgent.
    """
    brief: str                              # original user text
    components: tuple[str, ...]             # list of component names to design
    material: str                           # e.g. "Al6061"
    process: ManufacturingProcess
    loads: tuple[Load, ...]
    boundary_conditions: tuple[BoundaryCondition, ...]
    safety_factor: float
    tolerance_critical_dims: tuple[DimensionWithTolerance, ...]
    notes: str                              # any extra constraints


@dataclass(frozen=True)
class DesignTask:
    """A single component task derived from a DesignSpec."""
    component_name: str
    spec: DesignSpec
    acceptance_criteria: tuple[str, ...]    # plain-text list of pass conditions


@dataclass(frozen=True)
class WorkPlan:
    """Ordered list of DesignTasks produced by ArchitectAgent.decompose()."""
    tasks: tuple[DesignTask, ...]
    mating_constraints: tuple[MateConstraint, ...]


# ---------------------------------------------------------------------------
# Artifact types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeometryResult:
    """Output of GeometryService after executing parametric code."""
    step_path: Optional[Path]
    stl_path: Optional[Path]
    volume_mm3: float
    surface_area_mm2: float
    mass_g: float
    bounding_box_mm: tuple[float, float, float]   # x, y, z extents
    error: Optional[str]                           # None if successful

    @property
    def success(self) -> bool:
        return self.error is None and self.step_path is not None


@dataclass(frozen=True)
class DesignArtifact:
    """The output of one DesignerAgent generation or refinement pass."""
    component_name: str
    code: str                           # CadQuery Python source
    params: dict                        # parameter dict used in generation
    geometry: Optional[GeometryResult]  # None if code failed to execute
    iteration: int                      # 0 = first attempt, 1+ = refinements


@dataclass(frozen=True)
class Assembly:
    """A collection of DesignArtifacts with assembly constraints."""
    artifacts: tuple[DesignArtifact, ...]
    constraints: tuple[MateConstraint, ...]


# ---------------------------------------------------------------------------
# Critique types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Finding:
    """A single check result from CriticAgent."""
    category: CheckCategory
    severity: Severity
    message: str                    # what is wrong
    remediation: str                # what the Designer should do


@dataclass(frozen=True)
class CritiqueReport:
    """Aggregated findings from all CriticAgent checks."""
    artifact: DesignArtifact
    findings: tuple[Finding, ...]

    @property
    def passed(self) -> bool:
        return all(f.severity != Severity.FAIL for f in self.findings)

    @property
    def failures(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == Severity.FAIL)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == Severity.WARNING)


# ---------------------------------------------------------------------------
# Simulation result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FEAResult:
    """Output of SimService.run_fea()."""
    max_stress_mpa: float
    max_displacement_mm: float
    safety_factor: float
    mesh_path: Optional[Path]
    converged: bool
    error: Optional[str]


@dataclass(frozen=True)
class ToleranceResult:
    """Output of SimService.run_tolerance_stack()."""
    worst_case_gap_mm: float
    rss_gap_mm: float
    violation_probability: float    # 0.0–1.0
    chain_summary: str
