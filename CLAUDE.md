# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install package + dev dependencies
pip install -e ".[dev]"

# Geometry and simulation (optional, install separately)
pip install cadquery gmsh
# CalculiX FEA binary: install and add `ccx` to PATH
# ISO fasteners (optional, fallback geometry used if absent):
pip install git+https://github.com/gumyr/cq_warehouse.git

# Run tests (no test files exist yet — tests/ is a stub)
pytest

# Run CLI
caid design "Design brief here" --max-iterations 3 --output ./output --save

# Start REST API
python -m uvicorn api.routes:app --reload
```

Environment: copy `.env.example` → `.env` and set `ANTHROPIC_API_KEY`. Optional overrides: `CAID_MODEL`, `CAID_FAST_MODEL`, `CAID_OUTPUT_DIR`, `CAID_MAX_ITERATIONS`.

## Architecture

CAID converts a natural-language design brief into parametric 3D CAD (CadQuery → STEP/STL) via a multi-agent loop. The high-level flow:

```
Brief → ArchitectAgent → WorkPlan → [DesignerAgent → CriticAgent] × N → Assembly pass → DesignSession
```

### Core layer (`core/`)

- **`orchestrator.py`** — drives the full design loop; coordinates agents; handles per-component refine cycles and an assembly-level interference pass.
- **`schema.py`** — all shared dataclasses (`DesignSpec`, `WorkPlan`, `DesignTask`, `DesignArtifact`, `Assembly`, `CritiqueReport`, `Finding`). These are frozen; treat them as the canonical data contract between modules.
- **`llm_wrapper.py`** — single gateway to Anthropic Claude. Handles Jinja2 prompt rendering, JSON extraction, and code-fence extraction. Two model slots: default (`claude-sonnet-4-6`) and fast/cheap (`claude-haiku-4-5-20251001`).
- **`world_model.py`** — deterministic knowledge base of material properties and DFM process rules loaded from `data/world_model/*.yaml`. Injected into prompts to prevent hallucination; source of truth for constraint validation.
- **`session.py`** — mutable log (`DesignSession`) of all iterations, artifacts, and critiques; serializable to JSON.

### Agent layer (`agents/`)

- **`architect.py`** — parses brief → `DesignSpec`, decomposes → `WorkPlan`, extracts mating constraints for assemblies. In Phase 3+, searches the part library and injects reuse hints into `DesignSpec.notes`.
- **`designer.py`** — generates CadQuery Python code from a `DesignTask`. Executes it in a sandboxed subprocess via `GeometryService`. On critique failure, refines code using the `CritiqueReport` and error tracebacks.
- **`critic.py`** — runs three independent checks per artifact: DFM (LLM + rule-based), Physics (FEA via `SimService`, with analytical fallback), Interference (Boolean geometry intersection). Returns structured `CritiqueReport`.

### Geometry & simulation (`geometry/`, `sim/`)

- **`geometry/cadquery_ext.py`** (`GeometryService`) — executes untrusted LLM-generated CadQuery code in an isolated subprocess with CPU/memory limits (default 60s / 512 MB). Exports STEP/STL, computes mass properties. Also does Boolean interference detection between parts.
- **`sim/service.py`** (`SimService`) — unified facade called by `critic.py`; delegates to `fea_engine` and `tolerance` and returns typed results (`FEAResult`, `ToleranceResult`).
- **`sim/fea_engine.py`** — Gmsh meshing → CalculiX solve → parses `.dat` for max stress / displacement / safety factor. Falls back to an analytical bounding-box estimate if `ccx` is absent.
- **`sim/quick_check.py`** — fast rule-based pre-FEA filter (min feature size, density, overhang angle).
- **`sim/tolerance.py`** — worst-case and RSS tolerance stack-up with violation probability.

### Part library (`library/`) — Phase 3, actively developed

- **`repository.py`** — SQLite-backed store for AI-generated (`CUSTOM`) and ISO catalog (`STANDARD`) parts.
- **`search.py`** — pure-Python TF-IDF cosine similarity search over the repository.
- **`catalog.py`** — generates ISO 4762/4032/7089 fasteners (M2–M12); uses `cq_warehouse` if available, otherwise falls back to approximate CadQuery geometry.
- **`metadata.py`** — uses the fast Claude model to extract functional annotations (mounting holes, mating axes, feature summary) from CadQuery code.

### API & prompts (`api/`, `prompts/`)

- **`api/routes.py`** — FastAPI. Design submissions run in background tasks; sessions stored in-memory. Library endpoints mirror the part repository.
- **`prompts/*.jinja2`** — Jinja2 templates for each LLM call. Changing these is the primary lever for tuning agent behavior without touching Python code.

### Data files (`data/world_model/`)

`materials.yaml` and `process_rules.yaml` are the authoritative source for material properties and DFM bounds. `WorldModel` loads them at startup; all agents and the critic reference them through `WorldModel`, not by reading YAML directly.

## Key design decisions

- **Sandboxed code execution**: all LLM-generated CadQuery runs in a subprocess with resource limits — never `exec()` inline.
- **Frozen dataclasses**: `schema.py` types are immutable by design; new fields need explicit schema changes.
- **Graceful degradation**: FEA falls back to analytical estimates; ISO fastener geometry falls back to approximations; CalculiX is optional.
- **Prompt templates are code**: The `.jinja2` files control what context each agent receives (world model values, prior critiques, library hints). Keep them in sync with the schema fields they reference.
