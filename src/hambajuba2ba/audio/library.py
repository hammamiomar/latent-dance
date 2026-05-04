"""Persistent song library index.

The library stores metadata in SQLite and keeps heavy song artifacts on disk:
mix audio, separated stems, and feature NPZ files.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Iterable, Iterator

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SongRecord:
    song_id: str
    content_hash: str
    filename: str | None
    source_type: str
    source_uri: str | None
    duration: float
    bpm: float
    stems: tuple[str, ...]
    feature_version: int
    feature_level: str
    coupling_stems: str
    sample_rate: int
    mix_filename: str | None
    created_at: float
    updated_at: float


class SongLibrary:
    """SQLite table of contents for cached song artifacts."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "songs.sqlite"
        self._lock = threading.Lock()
        self._init_db()

    def song_dir(self, content_hash: str) -> Path:
        return self.root / content_hash

    def upsert_song(
        self,
        *,
        content_hash: str,
        filename: str | None,
        source_type: str,
        source_uri: str | None,
        duration: float,
        bpm: float,
        stems: Iterable[str],
        feature_version: int,
        feature_level: str,
        coupling_stems: str,
        sample_rate: int,
        mix_path: Path | str | None,
    ) -> SongRecord:
        now = time.time()
        song_id = content_hash
        stem_tuple = tuple(stems)
        mix_filename = Path(mix_path).name if mix_path else None

        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "select created_at from songs where song_id = ?",
                (song_id,),
            ).fetchone()
            created_at = float(existing["created_at"]) if existing else now
            conn.execute(
                """
                insert into songs (
                    song_id, content_hash, filename, source_type, source_uri,
                    duration, bpm, stems_json, feature_version, feature_level,
                    coupling_stems, sample_rate, mix_filename, created_at, updated_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(song_id) do update set
                    filename = excluded.filename,
                    source_type = excluded.source_type,
                    source_uri = excluded.source_uri,
                    duration = excluded.duration,
                    bpm = excluded.bpm,
                    stems_json = excluded.stems_json,
                    feature_version = excluded.feature_version,
                    feature_level = excluded.feature_level,
                    coupling_stems = excluded.coupling_stems,
                    sample_rate = excluded.sample_rate,
                    mix_filename = excluded.mix_filename,
                    updated_at = excluded.updated_at
                """,
                (
                    song_id,
                    content_hash,
                    filename,
                    source_type,
                    source_uri,
                    float(duration),
                    float(bpm),
                    json.dumps(list(stem_tuple)),
                    int(feature_version),
                    feature_level,
                    coupling_stems,
                    int(sample_rate),
                    mix_filename,
                    created_at,
                    now,
                ),
            )

        record = self.get_song(song_id)
        if record is None:
            raise RuntimeError(f"Failed to upsert song library row: {song_id}")
        return record

    def list_songs(self) -> list[SongRecord]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "select * from songs order by updated_at desc, created_at desc"
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get_song(self, song_id: str) -> SongRecord | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "select * from songs where song_id = ?",
                (song_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def delete_song(self, song_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cursor = conn.execute("delete from songs where song_id = ?", (song_id,))
        return cursor.rowcount > 0

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                create table if not exists metadata (
                    key text primary key,
                    value text not null
                )
                """
            )
            conn.execute(
                """
                create table if not exists songs (
                    song_id text primary key,
                    content_hash text not null unique,
                    filename text,
                    source_type text not null,
                    source_uri text,
                    duration real not null,
                    bpm real not null,
                    stems_json text not null,
                    feature_version integer not null,
                    feature_level text not null,
                    coupling_stems text not null,
                    sample_rate integer not null,
                    mix_filename text,
                    created_at real not null,
                    updated_at real not null
                )
                """
            )
            conn.execute(
                "insert or replace into metadata (key, value) values (?, ?)",
                ("schema_version", str(SCHEMA_VERSION)),
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma busy_timeout = 30000")
        conn.execute("pragma foreign_keys = on")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> SongRecord:
        return SongRecord(
            song_id=str(row["song_id"]),
            content_hash=str(row["content_hash"]),
            filename=row["filename"],
            source_type=str(row["source_type"]),
            source_uri=row["source_uri"],
            duration=float(row["duration"]),
            bpm=float(row["bpm"]),
            stems=tuple(json.loads(row["stems_json"])),
            feature_version=int(row["feature_version"]),
            feature_level=str(row["feature_level"]),
            coupling_stems=str(row["coupling_stems"]),
            sample_rate=int(row["sample_rate"]),
            mix_filename=row["mix_filename"],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )
