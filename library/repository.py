"""
PartRepository: SQLite-backed store for CAID custom-designed and ISO catalog parts.

Parts are distinguished by PartKind (CUSTOM vs STANDARD) so callers can filter
generated designs separately from off-the-shelf ISO hardware.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from core.schema import PartKind

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS parts (
    id             TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    description    TEXT NOT NULL,
    kind           TEXT NOT NULL,
    tags           TEXT NOT NULL,        -- JSON array
    parameters     TEXT NOT NULL,        -- JSON object
    cadquery_code  TEXT NOT NULL,
    step_path      TEXT,
    stl_path       TEXT,
    iso_standard   TEXT,                 -- e.g. "ISO 4762", NULL for custom
    created_at     TEXT NOT NULL
)
"""

_CREATE_INDEX = "CREATE INDEX IF NOT EXISTS idx_parts_kind ON parts(kind)"


@dataclass
class PartRecord:
    """One part stored in the repository — either AI-generated or ISO catalog."""
    name: str
    description: str
    kind: PartKind
    tags: list[str]
    parameters: dict           # geometry dims for custom; {size, length, standard} for catalog
    cadquery_code: str         # executable CadQuery source (empty string if STEP sourced directly)
    step_path: str | None
    stl_path: str | None
    iso_standard: str | None   # "ISO 4762" etc.; None for CUSTOM parts
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


class PartRepository:
    """
    Lightweight SQLite-backed store for part records.

    Args:
        db_path: Path to the SQLite database file. Parent directory is created
                 automatically. Defaults to output/parts.db.
    """

    def __init__(self, db_path: Path = Path("output/parts.db")) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, part: PartRecord) -> str:
        """Persist a PartRecord (INSERT OR REPLACE). Returns the part id."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO parts
                  (id, name, description, kind, tags, parameters,
                   cadquery_code, step_path, stl_path, iso_standard, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    part.id,
                    part.name,
                    part.description,
                    part.kind.value,
                    json.dumps(part.tags),
                    json.dumps(part.parameters),
                    part.cadquery_code,
                    part.step_path,
                    part.stl_path,
                    part.iso_standard,
                    part.created_at,
                ),
            )
        return part.id

    def get(self, part_id: str) -> PartRecord | None:
        """Retrieve a part by id. Returns None if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM parts WHERE id = ?", (part_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def search(
        self,
        query: str = "",
        tags: list[str] | None = None,
        kind: PartKind | None = None,
    ) -> list[PartRecord]:
        """
        Search parts by free-text and/or tags, optionally filtered by kind.

        Args:
            query: Substring match against name, description, and tags JSON.
            tags:  Exact tag strings that must all be present.
            kind:  If provided, restrict results to CUSTOM or STANDARD.

        Returns:
            List of PartRecord sorted by creation time descending.
        """
        clauses: list[str] = []
        params: list = []

        if query:
            clauses.append("(name LIKE ? OR description LIKE ? OR tags LIKE ?)")
            like = f"%{query}%"
            params.extend([like, like, like])

        if tags:
            for tag in tags:
                clauses.append("tags LIKE ?")
                params.append(f'%"{tag}"%')

        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind.value)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM parts {where} ORDER BY created_at DESC"

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [self._row_to_record(r) for r in rows]

    def list_all(self, kind: PartKind | None = None) -> list[PartRecord]:
        """Return all parts, optionally filtered by PartKind."""
        return self.search(kind=kind)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_INDEX)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Open a connection, yield it, then always close it (important on Windows)."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> PartRecord:
        return PartRecord(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            kind=PartKind(row["kind"]),
            tags=json.loads(row["tags"]),
            parameters=json.loads(row["parameters"]),
            cadquery_code=row["cadquery_code"],
            step_path=row["step_path"],
            stl_path=row["stl_path"],
            iso_standard=row["iso_standard"],
            created_at=row["created_at"],
        )
