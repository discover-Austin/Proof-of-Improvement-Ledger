from __future__ import annotations

from dataclasses import dataclass

from .models import PolicySnapshot, ProposedPolicyDiff, TrainingCandidate


@dataclass(frozen=True)
class DecoderPolicy:
    """
    A small mutable behavior layer used to demonstrate gated learning.

    In a real system, this role can be taken by a retrieval ranker, prompt compiler,
    tool policy, LoRA adapter pointer, model checkpoint pointer, or response decoder.
    """

    positive_markers: tuple[str, ...] = (
        "root cause",
        "full replacement",
        "full file",
        "no placeholders",
        "preserve",
        "context",
        "evidence",
        "tests pass",
        "validated",
        "source-backed",
    )

    negative_markers: tuple[str, ...] = (
        "todo",
        "placeholder",
        "stub",
        "fake",
        "vague",
        "partial",
        "ignore context",
        "unverified",
    )

    def propose_diff(
        self,
        candidate: TrainingCandidate,
        base_policy: PolicySnapshot,
    ) -> ProposedPolicyDiff:
        text = " ".join(
            [
                candidate.instruction.lower(),
                candidate.chosen.lower(),
                candidate.rejected.lower(),
                " ".join(candidate.tags).lower(),
            ]
        )

        delta: dict[str, float] = {}

        if "debug" in text or "traceback" in text or "root cause" in text:
            delta["root_cause_first"] = 0.08
        if "full replacement" in text or "full file" in text or "full function" in text:
            delta["full_replacement_code"] = 0.09
        if "placeholder" in text or "todo" in text or "stub" in text or "fake" in text:
            delta["avoid_placeholders"] = 0.10
        if "context" in text or "preserve" in text or "current state" in text:
            delta["preserve_context"] = 0.07
        if "scope" in text or "overfit" in text or "global" in text:
            delta["scope_control"] = 0.06
        if "evidence" in text or "validated" in text or "source" in text:
            delta["evidence_required"] = 0.06

        if not delta:
            delta["evidence_required"] = 0.03
            delta["scope_control"] = 0.03

        return ProposedPolicyDiff.create(
            candidate_id=candidate.id,
            base_policy_id=base_policy.id,
            delta=delta,
            rationale="Candidate text mapped into a small shadow policy update.",
        )

    def apply_diff(
        self,
        base_policy: PolicySnapshot,
        diff: ProposedPolicyDiff,
        metadata: dict[str, object] | None = None,
    ) -> PolicySnapshot:
        weights = dict(base_policy.weights)
        for key, change in diff.delta.items():
            before = weights.get(key, 0.5)
            weights[key] = max(0.0, min(1.0, before + change))

        next_metadata = dict(base_policy.metadata)
        next_metadata.update(metadata or {})
        next_metadata["parent_policy_id"] = base_policy.id
        next_metadata["applied_diff_id"] = diff.id

        return PolicySnapshot.create(
            version=base_policy.version + 1,
            weights=weights,
            metadata=next_metadata,
        )

    def render_contract(self, policy: PolicySnapshot) -> dict[str, bool]:
        weights = policy.weights
        return {
            "root_cause_first": weights.get("root_cause_first", 0.0) >= 0.55,
            "full_replacement_code": weights.get("full_replacement_code", 0.0) >= 0.55,
            "avoid_placeholders": weights.get("avoid_placeholders", 0.0) >= 0.55,
            "preserve_context": weights.get("preserve_context", 0.0) >= 0.55,
            "scope_control": weights.get("scope_control", 0.0) >= 0.55,
            "evidence_required": weights.get("evidence_required", 0.0) >= 0.55,
        }

    def score_candidate_text(self, candidate: TrainingCandidate) -> float:
        text = f"{candidate.instruction} {candidate.chosen}".lower()
        rejected = candidate.rejected.lower()

        positive_hits = sum(1 for marker in self.positive_markers if marker in text)
        negative_hits = sum(1 for marker in self.negative_markers if marker in text)
        rejected_negative_hits = sum(1 for marker in self.negative_markers if marker in rejected)

        raw = 0.5 + positive_hits * 0.06 - negative_hits * 0.10 + rejected_negative_hits * 0.04
        return max(0.0, min(1.0, raw))
