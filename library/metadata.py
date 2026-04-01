"""
PartMetadataExtractor: uses Claude to extract functional annotations from CadQuery code.

Extracted annotations:
  - mounting_holes: fastener holes (diameter, count, pattern)
  - mating_axes: assembly contact faces and axes
  - key_dimensions: named dimension variables from the code
  - feature_summary: one-sentence functional description
  - material_hint: material string found in the code, if any
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic import BaseModel

from core.llm_wrapper import LLMWrapper

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MountingHole:
    """A fastener hole pattern found in the part geometry."""
    diameter_mm: float
    count: int
    pattern: str        # e.g. "4× bolt circle", "2× linear array"


@dataclass(frozen=True)
class PartMetadata:
    """Functional annotations extracted from CadQuery source by Claude."""
    mounting_holes: tuple[MountingHole, ...]
    mating_axes: tuple[str, ...]        # e.g. ("Z-axis bore", "bottom face flange")
    key_dimensions: dict                # {"length_mm": 100, "bore_diameter_mm": 12, …}
    feature_summary: str                # one-sentence human-readable description
    material_hint: str                  # e.g. "Al6061" or "" if not found in code


# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM output
# ---------------------------------------------------------------------------

class _MountingHoleSchema(BaseModel):
    diameter_mm: float
    count: int
    pattern: str


class _PartMetadataSchema(BaseModel):
    mounting_holes: list[_MountingHoleSchema]
    mating_axes: list[str]
    key_dimensions: dict
    feature_summary: str
    material_hint: str


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class PartMetadataExtractor:
    """
    Extracts functional annotations from CadQuery source using Claude (fast model).

    Args:
        llm: LLMWrapper instance shared with the rest of the system.
    """

    def __init__(self, llm: LLMWrapper) -> None:
        self._llm = llm

    def extract(self, cadquery_code: str, part_name: str) -> PartMetadata:
        """
        Analyse CadQuery code and return structured functional metadata.

        Falls back to an empty PartMetadata on any LLM or parse failure so
        that callers never crash due to metadata extraction.

        Args:
            cadquery_code: Executable CadQuery Python source.
            part_name:     Human-readable name used as context for the LLM.

        Returns:
            PartMetadata with all fields populated (possibly empty tuples/dicts).
        """
        try:
            raw: _PartMetadataSchema = self._llm.complete_structured(
                "metadata_extract.jinja2",
                {"code": cadquery_code, "part_name": part_name},
                _PartMetadataSchema,
                fast=True,
                max_tokens=1024,
            )
            metadata = PartMetadata(
                mounting_holes=tuple(
                    MountingHole(
                        diameter_mm=h.diameter_mm,
                        count=h.count,
                        pattern=h.pattern,
                    )
                    for h in raw.mounting_holes
                ),
                mating_axes=tuple(raw.mating_axes),
                key_dimensions=raw.key_dimensions,
                feature_summary=raw.feature_summary,
                material_hint=raw.material_hint,
            )
            logger.debug(
                "Extracted metadata for '%s': %d holes, %d axes, summary=%r",
                part_name,
                len(metadata.mounting_holes),
                len(metadata.mating_axes),
                metadata.feature_summary[:60],
            )
            return metadata

        except Exception as exc:
            logger.warning("Metadata extraction failed for '%s': %s", part_name, exc)
            return PartMetadata(
                mounting_holes=(),
                mating_axes=(),
                key_dimensions={},
                feature_summary="",
                material_hint="",
            )
