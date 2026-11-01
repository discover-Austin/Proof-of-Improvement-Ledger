from __future__ import annotations

import json
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

from .models import (
    ConsensusDecision,
    LearningBlock,
    LearningEvent,
    PolicySnapshot,
    ProposedPolicyDiff,
    TrainingCandidate,
    ValidatorResult,
)


class LearningLedger:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn


    @contextmanager
    def transaction(self):
        conn = self.connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS learning_events (
                    id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    source_text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS training_candidates (
                    id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    instruction TEXT NOT NULL,
                    chosen TEXT NOT NULL,
                    rejected TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    FOREIGN KEY(event_id) REFERENCES learning_events(id)
                );

                CREATE TABLE IF NOT EXISTS policy_snapshots (
                    id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL UNIQUE,
                    weights_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    content_hash TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS proposed_policy_diffs (
                    id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL,
                    base_policy_id TEXT NOT NULL,
                    delta_json TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    FOREIGN KEY(candidate_id) REFERENCES training_candidates(id),
                    FOREIGN KEY(base_policy_id) REFERENCES policy_snapshots(id)
                );

                CREATE TABLE IF NOT EXISTS validator_results (
                    id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL,
                    validator_name TEXT NOT NULL,
                    passed INTEGER NOT NULL,
                    score REAL NOT NULL,
                    reason TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    FOREIGN KEY(candidate_id) REFERENCES training_candidates(id)
                );

                CREATE TABLE IF NOT EXISTS learning_blocks (
                    id TEXT PRIMARY KEY,
                    parent_hash TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    base_policy_id TEXT NOT NULL,
                    resulting_policy_id TEXT,
                    decision TEXT NOT NULL,
                    consensus_score REAL NOT NULL,
                    validator_result_ids_json TEXT NOT NULL,
                    memory_diff_json TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    block_hash TEXT NOT NULL UNIQUE,
                    FOREIGN KEY(candidate_id) REFERENCES training_candidates(id),
                    FOREIGN KEY(base_policy_id) REFERENCES policy_snapshots(id),
                    FOREIGN KEY(resulting_policy_id) REFERENCES policy_snapshots(id)
                );

                CREATE INDEX IF NOT EXISTS idx_candidates_status
                    ON training_candidates(status, created_at_ms);

                CREATE INDEX IF NOT EXISTS idx_validator_candidate
                    ON validator_results(candidate_id);

                CREATE INDEX IF NOT EXISTS idx_blocks_created
                    ON learning_blocks(created_at_ms);
                """
            )
            if self.get_active_policy(conn=conn) is None:
                genesis = PolicySnapshot.create(
                    version=0,
                    weights={
                        "root_cause_first": 0.5,
                        "full_replacement_code": 0.5,
                        "avoid_placeholders": 0.5,
                        "preserve_context": 0.5,
                        "scope_control": 0.5,
                        "evidence_required": 0.5,
                    },
                    metadata={"kind": "genesis"},
                )
                self.save_policy(genesis, conn=conn)

    def save_event(self, event: LearningEvent, conn: sqlite3.Connection | None = None) -> None:
        owns_conn = conn is None
        conn = conn or self.connect()
        try:
            conn.execute(
                """
                INSERT INTO learning_events
                    (id, event_type, source_text, metadata_json, created_at_ms)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.event_type,
                    event.source_text,
                    json.dumps(event.metadata, sort_keys=True),
                    event.created_at_ms,
                ),
            )
            if owns_conn:
                conn.commit()
        finally:
            if owns_conn:
                conn.close()

    def save_candidate(self, candidate: TrainingCandidate, conn: sqlite3.Connection | None = None) -> None:
        owns_conn = conn is None
        conn = conn or self.connect()
        try:
            conn.execute(
                """
                INSERT INTO training_candidates
                    (id, event_id, instruction, chosen, rejected, tags_json, status, created_at_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.id,
                    candidate.event_id,
                    candidate.instruction,
                    candidate.chosen,
                    candidate.rejected,
                    json.dumps(list(candidate.tags), sort_keys=True),
                    candidate.status,
                    candidate.created_at_ms,
                ),
            )
            if owns_conn:
                conn.commit()
        finally:
            if owns_conn:
                conn.close()

    def save_policy(self, policy: PolicySnapshot, conn: sqlite3.Connection | None = None) -> None:
        owns_conn = conn is None
        conn = conn or self.connect()
        try:
            conn.execute(
                """
                INSERT INTO policy_snapshots
                    (id, version, weights_json, metadata_json, created_at_ms, content_hash)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    policy.id,
                    policy.version,
                    json.dumps(policy.weights, sort_keys=True),
                    json.dumps(policy.metadata, sort_keys=True),
                    policy.created_at_ms,
                    policy.content_hash(),
                ),
            )
            if owns_conn:
                conn.commit()
        finally:
            if owns_conn:
                conn.close()

    def save_policy_diff(self, diff: ProposedPolicyDiff, conn: sqlite3.Connection | None = None) -> None:
        owns_conn = conn is None
        conn = conn or self.connect()
        try:
            conn.execute(
                """
                INSERT INTO proposed_policy_diffs
                    (id, candidate_id, base_policy_id, delta_json, rationale, created_at_ms)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    diff.id,
                    diff.candidate_id,
                    diff.base_policy_id,
                    json.dumps(diff.delta, sort_keys=True),
                    diff.rationale,
                    diff.created_at_ms,
                ),
            )
            if owns_conn:
                conn.commit()
        finally:
            if owns_conn:
                conn.close()

    def save_validator_result(self, result: ValidatorResult, conn: sqlite3.Connection | None = None) -> None:
        owns_conn = conn is None
        conn = conn or self.connect()
        try:
            conn.execute(
                """
                INSERT INTO validator_results
                    (id, candidate_id, validator_name, passed, score, reason, details_json, created_at_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.id,
                    result.candidate_id,
                    result.validator_name,
                    int(result.passed),
                    result.score,
                    result.reason,
                    json.dumps(result.details, sort_keys=True),
                    result.created_at_ms,
                ),
            )
            if owns_conn:
                conn.commit()
        finally:
            if owns_conn:
                conn.close()

    def save_learning_block(self, block: LearningBlock, conn: sqlite3.Connection | None = None) -> None:
        owns_conn = conn is None
        conn = conn or self.connect()
        try:
            conn.execute(
                """
                INSERT INTO learning_blocks
                    (
                        id, parent_hash, candidate_id, base_policy_id, resulting_policy_id,
                        decision, consensus_score, validator_result_ids_json,
                        memory_diff_json, created_at_ms, block_hash
                    )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    block.id,
                    block.parent_hash,
                    block.candidate_id,
                    block.base_policy_id,
                    block.resulting_policy_id,
                    block.decision.value,
                    block.consensus_score,
                    json.dumps(list(block.validator_result_ids), sort_keys=True),
                    json.dumps(block.memory_diff, sort_keys=True),
                    block.created_at_ms,
                    block.block_hash,
                ),
            )
            if owns_conn:
                conn.commit()
        finally:
            if owns_conn:
                conn.close()

    def mark_candidate(self, candidate_id: str, status: str, conn: sqlite3.Connection | None = None) -> None:
        owns_conn = conn is None
        conn = conn or self.connect()
        try:
            conn.execute(
                "UPDATE training_candidates SET status = ? WHERE id = ?",
                (status, candidate_id),
            )
            if owns_conn:
                conn.commit()
        finally:
            if owns_conn:
                conn.close()

    def get_active_policy(self, conn: sqlite3.Connection | None = None) -> PolicySnapshot | None:
        owns_conn = conn is None
        conn = conn or self.connect()
        try:
            row = conn.execute(
                """
                SELECT * FROM policy_snapshots
                ORDER BY version DESC
                LIMIT 1
                """
            ).fetchone()
            return self._policy_from_row(row) if row else None
        finally:
            if owns_conn:
                conn.close()

    def get_candidate(self, candidate_id: str, conn: sqlite3.Connection | None = None) -> TrainingCandidate | None:
        owns_conn = conn is None
        conn = conn or self.connect()
        try:
            row = conn.execute(
                "SELECT * FROM training_candidates WHERE id = ?",
                (candidate_id,),
            ).fetchone()
            return self._candidate_from_row(row) if row else None
        finally:
            if owns_conn:
                conn.close()

    def latest_pending_candidate(self, conn: sqlite3.Connection | None = None) -> TrainingCandidate | None:
        owns_conn = conn is None
        conn = conn or self.connect()
        try:
            row = conn.execute(
                """
                SELECT * FROM training_candidates
                WHERE status = 'pending'
                ORDER BY created_at_ms DESC
                LIMIT 1
                """
            ).fetchone()
            return self._candidate_from_row(row) if row else None
        finally:
            if owns_conn:
                conn.close()

    def validator_results_for_candidate(
        self,
        candidate_id: str,
        conn: sqlite3.Connection | None = None,
    ) -> list[ValidatorResult]:
        owns_conn = conn is None
        conn = conn or self.connect()
        try:
            rows = conn.execute(
                """
                SELECT * FROM validator_results
                WHERE candidate_id = ?
                ORDER BY created_at_ms ASC
                """,
                (candidate_id,),
            ).fetchall()
            return [self._validator_result_from_row(row) for row in rows]
        finally:
            if owns_conn:
                conn.close()

    def latest_block_hash(self, conn: sqlite3.Connection | None = None) -> str:
        owns_conn = conn is None
        conn = conn or self.connect()
        try:
            row = conn.execute(
                """
                SELECT block_hash FROM learning_blocks
                ORDER BY created_at_ms DESC
                LIMIT 1
                """
            ).fetchone()
            return row["block_hash"] if row else "GENESIS"
        finally:
            if owns_conn:
                conn.close()

    def status(self) -> dict[str, Any]:
        with closing(self.connect()) as conn:
            active = self.get_active_policy(conn=conn)
            counts = {}
            for table in [
                "learning_events",
                "training_candidates",
                "policy_snapshots",
                "validator_results",
                "learning_blocks",
            ]:
                counts[table] = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
            pending = conn.execute(
                "SELECT COUNT(*) AS n FROM training_candidates WHERE status = 'pending'"
            ).fetchone()["n"]
            return {
                "db_path": str(self.db_path),
                "active_policy_version": active.version if active else None,
                "active_policy_id": active.id if active else None,
                "active_policy_hash": active.content_hash() if active else None,
                "pending_candidates": pending,
                "counts": counts,
                "latest_block_hash": self.latest_block_hash(conn=conn),
            }

    def export(self) -> dict[str, Any]:
        with closing(self.connect()) as conn:
            return {
                "status": self.status(),
                "events": [dict(row) for row in conn.execute("SELECT * FROM learning_events")],
                "candidates": [dict(row) for row in conn.execute("SELECT * FROM training_candidates")],
                "policies": [dict(row) for row in conn.execute("SELECT * FROM policy_snapshots")],
                "diffs": [dict(row) for row in conn.execute("SELECT * FROM proposed_policy_diffs")],
                "validator_results": [dict(row) for row in conn.execute("SELECT * FROM validator_results")],
                "learning_blocks": [dict(row) for row in conn.execute("SELECT * FROM learning_blocks")],
            }

    def _candidate_from_row(self, row: sqlite3.Row) -> TrainingCandidate:
        return TrainingCandidate(
            id=row["id"],
            event_id=row["event_id"],
            instruction=row["instruction"],
            chosen=row["chosen"],
            rejected=row["rejected"],
            tags=tuple(json.loads(row["tags_json"])),
            status=row["status"],
            created_at_ms=int(row["created_at_ms"]),
        )

    def _policy_from_row(self, row: sqlite3.Row) -> PolicySnapshot:
        return PolicySnapshot(
            id=row["id"],
            version=int(row["version"]),
            weights={k: float(v) for k, v in json.loads(row["weights_json"]).items()},
            metadata=json.loads(row["metadata_json"]),
            created_at_ms=int(row["created_at_ms"]),
        )

    def _validator_result_from_row(self, row: sqlite3.Row) -> ValidatorResult:
        return ValidatorResult(
            id=row["id"],
            candidate_id=row["candidate_id"],
            validator_name=row["validator_name"],
            passed=bool(row["passed"]),
            score=float(row["score"]),
            reason=row["reason"],
            details=json.loads(row["details_json"]),
            created_at_ms=int(row["created_at_ms"]),
        )
