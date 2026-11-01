from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
import hashlib
import json
import time
import uuid


def now_ms() -> int:
    return int(time.time() * 1000)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class ConsensusDecision(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"


@dataclass(frozen=True)
class LearningEvent:
    id: str
    event_type: str
    source_text: str
    metadata: dict[str, Any]
    created_at_ms: int

    @classmethod
    def create(
        cls,
        event_type: str,
        source_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> "LearningEvent":
        clean_source = source_text.strip()
        if not clean_source:
            raise ValueError("source_text cannot be empty")
        return cls(
            id=new_id("evt"),
            event_type=event_type.strip() or "unspecified",
            source_text=clean_source,
            metadata=dict(metadata or {}),
            created_at_ms=now_ms(),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "event_type": self.event_type,
            "source_text": self.source_text,
            "metadata": self.metadata,
            "created_at_ms": self.created_at_ms,
        }


@dataclass(frozen=True)
class TrainingCandidate:
    id: str
    event_id: str
    instruction: str
    chosen: str
    rejected: str
    tags: tuple[str, ...]
    status: str
    created_at_ms: int

    @classmethod
    def create(
        cls,
        event_id: str,
        instruction: str,
        chosen: str,
        rejected: str = "",
        tags: list[str] | tuple[str, ...] | None = None,
    ) -> "TrainingCandidate":
        clean_instruction = instruction.strip()
        clean_chosen = chosen.strip()
        if not clean_instruction:
            raise ValueError("instruction cannot be empty")
        if not clean_chosen:
            raise ValueError("chosen cannot be empty")
        clean_tags = tuple(sorted({tag.strip() for tag in (tags or []) if tag.strip()}))
        return cls(
            id=new_id("cand"),
            event_id=event_id,
            instruction=clean_instruction,
            chosen=clean_chosen,
            rejected=rejected.strip(),
            tags=clean_tags,
            status="pending",
            created_at_ms=now_ms(),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "event_id": self.event_id,
            "instruction": self.instruction,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "tags": list(self.tags),
            "status": self.status,
            "created_at_ms": self.created_at_ms,
        }


@dataclass(frozen=True)
class PolicySnapshot:
    id: str
    version: int
    weights: dict[str, float]
    metadata: dict[str, Any]
    created_at_ms: int

    @classmethod
    def create(
        cls,
        version: int,
        weights: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "PolicySnapshot":
        return cls(
            id=new_id("policy"),
            version=version,
            weights={str(k): float(v) for k, v in (weights or {}).items()},
            metadata=dict(metadata or {}),
            created_at_ms=now_ms(),
        )

    def content_hash(self) -> str:
        return sha256_json(
            {
                "version": self.version,
                "weights": self.weights,
                "metadata": self.metadata,
            }
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "weights": self.weights,
            "metadata": self.metadata,
            "created_at_ms": self.created_at_ms,
            "content_hash": self.content_hash(),
        }


@dataclass(frozen=True)
class ProposedPolicyDiff:
    id: str
    candidate_id: str
    base_policy_id: str
    delta: dict[str, float]
    rationale: str
    created_at_ms: int

    @classmethod
    def create(
        cls,
        candidate_id: str,
        base_policy_id: str,
        delta: dict[str, float],
        rationale: str,
    ) -> "ProposedPolicyDiff":
        if not delta:
            raise ValueError("delta cannot be empty")
        clean_delta = {str(k): float(v) for k, v in delta.items() if float(v) != 0.0}
        if not clean_delta:
            raise ValueError("delta cannot contain only zero changes")
        return cls(
            id=new_id("diff"),
            candidate_id=candidate_id,
            base_policy_id=base_policy_id,
            delta=clean_delta,
            rationale=rationale.strip() or "Candidate-derived policy update.",
            created_at_ms=now_ms(),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "candidate_id": self.candidate_id,
            "base_policy_id": self.base_policy_id,
            "delta": self.delta,
            "rationale": self.rationale,
            "created_at_ms": self.created_at_ms,
        }


@dataclass(frozen=True)
class ValidatorResult:
    id: str
    candidate_id: str
    validator_name: str
    passed: bool
    score: float
    reason: str
    details: dict[str, Any]
    created_at_ms: int

    @classmethod
    def create(
        cls,
        candidate_id: str,
        validator_name: str,
        passed: bool,
        score: float,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> "ValidatorResult":
        bounded = max(0.0, min(1.0, float(score)))
        return cls(
            id=new_id("val"),
            candidate_id=candidate_id,
            validator_name=validator_name,
            passed=bool(passed),
            score=bounded,
            reason=reason.strip() or "No reason recorded.",
            details=dict(details or {}),
            created_at_ms=now_ms(),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "candidate_id": self.candidate_id,
            "validator_name": self.validator_name,
            "passed": self.passed,
            "score": self.score,
            "reason": self.reason,
            "details": self.details,
            "created_at_ms": self.created_at_ms,
        }


@dataclass(frozen=True)
class LearningBlock:
    id: str
    parent_hash: str
    candidate_id: str
    base_policy_id: str
    resulting_policy_id: str | None
    decision: ConsensusDecision
    consensus_score: float
    validator_result_ids: tuple[str, ...]
    memory_diff: dict[str, Any]
    created_at_ms: int
    block_hash: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            "id": self.id,
            "parent_hash": self.parent_hash,
            "candidate_id": self.candidate_id,
            "base_policy_id": self.base_policy_id,
            "resulting_policy_id": self.resulting_policy_id,
            "decision": self.decision.value,
            "consensus_score": self.consensus_score,
            "validator_result_ids": list(self.validator_result_ids),
            "memory_diff": self.memory_diff,
            "created_at_ms": self.created_at_ms,
        }
        object.__setattr__(self, "block_hash", sha256_json(payload))

    @classmethod
    def create(
        cls,
        parent_hash: str,
        candidate_id: str,
        base_policy_id: str,
        resulting_policy_id: str | None,
        decision: ConsensusDecision,
        consensus_score: float,
        validator_result_ids: list[str] | tuple[str, ...],
        memory_diff: dict[str, Any] | None = None,
    ) -> "LearningBlock":
        return cls(
            id=new_id("block"),
            parent_hash=parent_hash,
            candidate_id=candidate_id,
            base_policy_id=base_policy_id,
            resulting_policy_id=resulting_policy_id,
            decision=decision,
            consensus_score=max(0.0, min(1.0, float(consensus_score))),
            validator_result_ids=tuple(validator_result_ids),
            memory_diff=dict(memory_diff or {}),
            created_at_ms=now_ms(),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent_hash": self.parent_hash,
            "candidate_id": self.candidate_id,
            "base_policy_id": self.base_policy_id,
            "resulting_policy_id": self.resulting_policy_id,
            "decision": self.decision.value,
            "consensus_score": self.consensus_score,
            "validator_result_ids": list(self.validator_result_ids),
            "memory_diff": self.memory_diff,
            "created_at_ms": self.created_at_ms,
            "block_hash": self.block_hash,
        }
