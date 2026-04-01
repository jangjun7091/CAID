"""
StandardPartsCatalog: generates parametric CadQuery code for ISO standard fasteners.

Primary backend:  cq_warehouse (gumyr/cq_warehouse) — accurate ISO geometry with
                  real thread profiles when installed.
Fallback backend: pure CadQuery geometric approximations — correct outer envelope
                  and bounding box; no thread detail. Suitable for interference
                  checking, clearance analysis, and assembly visualization.

Supported standards:
  ISO 4762 — Socket Head Cap Screw (M2–M12)
  ISO 4032 — Hex Nut              (M2–M12)
  ISO 7089 — Plain Washer         (M2–M12)
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Optional

# Lazy availability check — performed once on first call.
_CQ_WAREHOUSE_AVAILABLE: bool | None = None


def _has_cq_warehouse() -> bool:
    global _CQ_WAREHOUSE_AVAILABLE
    if _CQ_WAREHOUSE_AVAILABLE is None:
        try:
            importlib.import_module("cq_warehouse.fastener")
            _CQ_WAREHOUSE_AVAILABLE = True
        except ImportError:
            _CQ_WAREHOUSE_AVAILABLE = False
    return _CQ_WAREHOUSE_AVAILABLE


# ---------------------------------------------------------------------------
# Dimension tables (source: ISO standards)
# ---------------------------------------------------------------------------

# ISO 4762 Socket Head Cap Screw: (head_dia_mm, head_height_mm, socket_width_mm, nominal_dia_mm)
_SHCS_DIMS: dict[str, tuple[float, float, float, float]] = {
    "M2":  (3.8,  2.0, 1.5, 2.0),
    "M3":  (5.5,  3.0, 2.5, 3.0),
    "M4":  (7.0,  4.0, 3.0, 4.0),
    "M5":  (8.5,  5.0, 4.0, 5.0),
    "M6":  (10.0, 6.0, 5.0, 6.0),
    "M8":  (13.0, 8.0, 6.0, 8.0),
    "M10": (16.0, 10.0, 8.0, 10.0),
    "M12": (18.0, 12.0, 10.0, 12.0),
}

# ISO 4032 Hex Nut: (across_corners_mm, thickness_mm, thread_dia_mm)
_HEX_NUT_DIMS: dict[str, tuple[float, float, float]] = {
    "M2":  (4.62,  1.6,  2.0),
    "M3":  (6.35,  2.4,  3.0),
    "M4":  (8.08,  3.2,  4.0),
    "M5":  (9.24,  4.7,  5.0),
    "M6":  (11.55, 5.2,  6.0),
    "M8":  (15.01, 6.8,  8.0),
    "M10": (18.48, 8.4,  10.0),
    "M12": (21.10, 10.8, 12.0),
}

# ISO 7089 Plain Washer: (inner_dia_mm, outer_dia_mm, thickness_mm)
_WASHER_DIMS: dict[str, tuple[float, float, float]] = {
    "M2":  (2.4, 6.0,  0.5),
    "M3":  (3.2, 7.0,  0.5),
    "M4":  (4.3, 9.0,  0.8),
    "M5":  (5.3, 10.0, 1.0),
    "M6":  (6.4, 12.0, 1.6),
    "M8":  (8.4, 16.0, 1.6),
    "M10": (10.5, 20.0, 2.0),
    "M12": (13.0, 24.0, 2.5),
}

# Standard metric thread pitches (coarse series)
_METRIC_PITCHES: dict[str, str] = {
    "M2":  "0.4",
    "M3":  "0.5",
    "M4":  "0.7",
    "M5":  "0.8",
    "M6":  "1.0",
    "M8":  "1.25",
    "M10": "1.5",
    "M12": "1.75",
}


# ---------------------------------------------------------------------------
# Spec dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ISOPartSpec:
    """Specification for an ISO standard part request."""
    standard: str               # e.g. "ISO 4762"
    size: str                   # e.g. "M3" or "M3-0.5" (pitch suffix is stripped)
    length_mm: Optional[float] = None   # required for bolts/screws
    simple: bool = True         # True = geometric approx; False = threaded (cq_warehouse)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

class StandardPartsCatalog:
    """
    Generates CadQuery code strings for ISO standard fasteners.

    Usage::

        catalog = StandardPartsCatalog()
        spec = ISOPartSpec(standard="ISO 4762", size="M3", length_mm=16)
        code, description = catalog.get_code(spec)
        # Pass `code` to GeometryService.execute_cadquery() to produce STEP/STL.
    """

    SUPPORTED_STANDARDS: dict[str, str] = {
        "ISO 4762": "Socket Head Cap Screw",
        "ISO 4032": "Hex Nut",
        "ISO 7089": "Plain Washer",
    }

    def available_standards(self) -> dict[str, str]:
        """Return {standard_code: human_name} for all supported ISO standards."""
        return dict(self.SUPPORTED_STANDARDS)

    def available_sizes(self, standard: str) -> list[str]:
        """Return metric sizes supported for the given standard."""
        _dim_map = {
            "ISO 4762": _SHCS_DIMS,
            "ISO 4032": _HEX_NUT_DIMS,
            "ISO 7089": _WASHER_DIMS,
        }
        dims = _dim_map.get(standard)
        if dims is None:
            raise ValueError(
                f"Unknown standard {standard!r}. Supported: {list(self.SUPPORTED_STANDARDS)}"
            )
        return list(dims.keys())

    def get_code(self, spec: ISOPartSpec) -> tuple[str, str]:
        """
        Return (cadquery_code, description) for the given ISOPartSpec.

        The returned code assigns its solid to a variable named ``result``
        and is directly executable by GeometryService.execute_cadquery().

        When cq_warehouse is installed and spec.simple=False, the code imports
        cq_warehouse to produce accurate ISO geometry with thread profiles.
        Otherwise a pure CadQuery geometric approximation is returned.

        Raises:
            ValueError: If the standard or size is not supported.
        """
        size = _normalize_size(spec.size)

        if spec.standard == "ISO 4762":
            if spec.length_mm is None:
                raise ValueError("length_mm is required for ISO 4762 (screw).")
            return self._shcs_code(size, spec.length_mm, spec.simple)

        if spec.standard == "ISO 4032":
            return self._hex_nut_code(size, spec.simple)

        if spec.standard == "ISO 7089":
            return self._washer_code(size)

        raise ValueError(
            f"Unsupported standard {spec.standard!r}. "
            f"Supported: {list(self.SUPPORTED_STANDARDS)}"
        )

    def get_part_name(self, spec: ISOPartSpec) -> str:
        """
        Return a canonical filesystem-safe part name.
        Examples: 'ISO4762_M3x16', 'ISO4032_M6', 'ISO7089_M4'
        """
        size = _normalize_size(spec.size)
        tag = spec.standard.replace(" ", "")
        if spec.length_mm is not None:
            return f"{tag}_{size}x{int(spec.length_mm)}"
        return f"{tag}_{size}"

    def backend(self) -> str:
        """Return 'cq_warehouse' if available, else 'fallback'."""
        return "cq_warehouse" if _has_cq_warehouse() else "fallback"

    # ------------------------------------------------------------------
    # ISO 4762: Socket Head Cap Screw
    # ------------------------------------------------------------------

    def _shcs_code(self, size: str, length_mm: float, simple: bool) -> tuple[str, str]:
        if _has_cq_warehouse() and not simple:
            pitch = _METRIC_PITCHES.get(size, "1.0")
            cq_size = f"{size}-{pitch}"
            code = (
                f'from cq_warehouse.fastener import SocketHeadCapScrew\n'
                f'result = SocketHeadCapScrew(\n'
                f'    size="{cq_size}",\n'
                f'    fastener_type="iso4762",\n'
                f'    length={length_mm},\n'
                f'    simple=False,\n'
                f')\n'
            )
            return code, (
                f"{size} Socket Head Cap Screw (ISO 4762), L={length_mm} mm "
                f"[cq_warehouse — threaded]"
            )

        if size not in _SHCS_DIMS:
            raise ValueError(
                f"Size {size!r} not supported for ISO 4762. "
                f"Supported: {list(_SHCS_DIMS)}"
            )
        head_dia, head_h, socket_w, nom_dia = _SHCS_DIMS[size]
        # polygon(6, d) uses circumradius; socket_w is across-flats → circumradius = s/cos(30°)
        socket_cr = round(socket_w / 0.866, 4)
        socket_depth = round(head_h * 0.6, 4)

        code = f"""\
import cadquery as cq

head_dia    = {head_dia}
head_h      = {head_h}
socket_cr   = {socket_cr}   # hex socket circumradius
socket_depth = {socket_depth}
shank_r     = {nom_dia / 2.0}
length      = {length_mm}

# Cylindrical head
head = cq.Workplane("XY").circle(head_dia / 2.0).extrude(head_h)

# Hex socket recess (subtracted from top of head)
socket_recess = (
    cq.Workplane("XY")
    .workplane(offset=head_h - socket_depth)
    .polygon(6, socket_cr)
    .extrude(socket_depth)
)
head = head.cut(socket_recess)

# Threaded shank (approximated as plain cylinder — correct envelope for interference checks)
shank = cq.Workplane("XY").circle(shank_r).extrude(-length)

result = head.union(shank)
"""
        return code, (
            f"{size} Socket Head Cap Screw (ISO 4762), L={length_mm} mm "
            f"[geometric approximation — no thread detail]"
        )

    # ------------------------------------------------------------------
    # ISO 4032: Hex Nut
    # ------------------------------------------------------------------

    def _hex_nut_code(self, size: str, simple: bool) -> tuple[str, str]:
        if _has_cq_warehouse() and not simple:
            pitch = _METRIC_PITCHES.get(size, "1.0")
            cq_size = f"{size}-{pitch}"
            code = (
                f'from cq_warehouse.fastener import HexNut\n'
                f'result = HexNut(\n'
                f'    size="{cq_size}",\n'
                f'    fastener_type="iso4032",\n'
                f'    simple=False,\n'
                f')\n'
            )
            return code, f"{size} Hex Nut (ISO 4032) [cq_warehouse — threaded]"

        if size not in _HEX_NUT_DIMS:
            raise ValueError(
                f"Size {size!r} not supported for ISO 4032. "
                f"Supported: {list(_HEX_NUT_DIMS)}"
            )
        across_corners, thickness, thread_dia = _HEX_NUT_DIMS[size]
        inner_r = thread_dia / 2.0

        code = f"""\
import cadquery as cq

across_corners = {across_corners}   # circumscribed circle diameter
thickness      = {thickness}
inner_r        = {inner_r}

# Hex body
body = cq.Workplane("XY").polygon(6, across_corners).extrude(thickness)

# Central hole (thread approximated as plain cylinder)
hole = cq.Workplane("XY").circle(inner_r).extrude(thickness)

result = body.cut(hole)
"""
        return code, (
            f"{size} Hex Nut (ISO 4032), thickness={thickness} mm "
            f"[geometric approximation — no thread detail]"
        )

    # ------------------------------------------------------------------
    # ISO 7089: Plain Washer
    # ------------------------------------------------------------------

    def _washer_code(self, size: str) -> tuple[str, str]:
        if size not in _WASHER_DIMS:
            raise ValueError(
                f"Size {size!r} not supported for ISO 7089. "
                f"Supported: {list(_WASHER_DIMS)}"
            )
        inner_dia, outer_dia, thickness = _WASHER_DIMS[size]

        code = f"""\
import cadquery as cq

inner_r   = {inner_dia / 2.0}
outer_r   = {outer_dia / 2.0}
thickness = {thickness}

outer = cq.Workplane("XY").circle(outer_r).extrude(thickness)
hole  = cq.Workplane("XY").circle(inner_r).extrude(thickness)

result = outer.cut(hole)
"""
        return code, (
            f"{size} Plain Washer (ISO 7089), "
            f"OD={outer_dia} mm, ID={inner_dia} mm, t={thickness} mm"
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _normalize_size(size: str) -> str:
    """
    Accept 'M3', 'M3-0.5', 'm3' → 'M3'.
    Strips pitch suffix and uppercases.
    """
    s = size.strip().upper().split("-")[0]
    if not s.startswith("M") or not s[1:].isdigit():
        raise ValueError(
            f"Expected a metric size like 'M3' or 'M3-0.5', got {size!r}"
        )
    return s
