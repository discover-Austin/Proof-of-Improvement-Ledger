from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .models import PolicySnapshot, ProposedPolicyDiff, TrainingCandidate, ValidatorResult
from .policy import DecoderPolicy


@dataclass(frozen=True)
class ValidationContext:
    candidate: TrainingCandidate
    base_policy: PolicySnapshot
    shadow_policy: PolicySnapshot
    diff: ProposedPolicyDiff
    decoder: DecoderPolicy


class Validator(ABC):
    name: str

    @abstractmethod
    def validate(self, context: ValidationContext) -> ValidatorResult:
        raise NotImplementedError


class PreferenceValidator(Validator):
    name = "preference_alignment"

    def validate(self, context: ValidationContext) -> ValidatorResult:
        candidate = context.candidate
        text = f"{candidate.instruction} {candidate.chosen} {' '.join(candidate.tags)}".lower()

        score = context.decoder.score_candidate_text(candidate)
        reasons: list[str] = []

        if "placeholder" in candidate.chosen.lower() or "todo" in candidate.chosen.lower():
            score -= 0.30
            reasons.append("Chosen output contains placeholder/TODO language.")

        if "debug" in text and "root cause" in text:
            score += 0.10
            reasons.append("Debugging instruction includes root-cause-first behavior.")

        if "full replacement" in text or "full file" in text or "full function" in text:
            score += 0.10
            reasons.append("Candidate reinforces full replacement behavior.")

        bounded = max(0.0, min(1.0, score))
        passed = bounded >= 0.62
        return ValidatorResult.create(
            candidate_id=candidate.id,
            validator_name=self.name,
            passed=passed,
            score=bounded,
            reason="; ".join(reasons) if reasons else "Candidate is preference-compatible.",
            details={"tags": list(candidate.tags)},
        )


class EvidenceQualityValidator(Validator):
    name = "evidence_quality"

    def validate(self, context: ValidationContext) -> ValidatorResult:
        candidate = context.candidate

        score = 0.45
        details: dict[str, object] = {
            "instruction_len": len(candidate.instruction),
            "chosen_len": len(candidate.chosen),
            "has_rejected": bool(candidate.rejected.strip()),
            "tag_count": len(candidate.tags),
        }

        if len(candidate.instruction) >= 20:
            score += 0.15
        if len(candidate.chosen) >= 30:
            score += 0.15
        if candidate.rejected.strip():
            score += 0.15
        if candidate.tags:
            score += 0.10

        passed = score >= 0.70
        return ValidatorResult.create(
            candidate_id=candidate.id,
            validator_name=self.name,
            passed=passed,
            score=max(0.0, min(1.0, score)),
            reason="Candidate has enough structured evidence." if passed else "Candidate lacks enough structured evidence.",
            details=details,
        )


class ContradictionValidator(Validator):
    name = "contradiction_check"

    contradiction_pairs: tuple[tuple[str, str], ...] = (
        ("full replacement", "partial snippet"),
        ("no placeholders", "placeholder"),
        ("preserve context", "ignore context"),
        ("source-backed", "unverified"),
        ("root cause first", "vague advice"),
    )

    def validate(self, context: ValidationContext) -> ValidatorResult:
        candidate = context.candidate
        chosen = candidate.chosen.lower()
        rejected = candidate.rejected.lower()
        instruction = candidate.instruction.lower()

        contradictions: list[str] = []
        combined_positive = f"{instruction} {chosen}"

        for positive, negative in self.contradiction_pairs:
            if positive in combined_positive and negative in chosen:
                contradictions.append(f"chosen contains both {positive!r} and {negative!r}")

        if "always" in chosen and ("simple question" in chosen or "every request" in chosen):
            contradictions.append("chosen appears to overgeneralize with always/every-request language")

        if candidate.rejected and chosen.strip() == rejected.strip():
            contradictions.append("chosen and rejected outputs are identical")

        score = 1.0 if not contradictions else max(0.0, 1.0 - 0.25 * len(contradictions))
        passed = not contradictions

        return ValidatorResult.create(
            candidate_id=candidate.id,
            validator_name=self.name,
            passed=passed,
            score=score,
            reason="No direct contradiction found." if passed else "; ".join(contradictions),
            details={"contradictions": contradictions},
        )


class ScopeValidator(Validator):
    name = "scope_control"

    risky_global_terms: tuple[str, ...] = (
        "always ignore",
        "never verify",
        "never cite",
        "all tasks",
        "every task",
        "without exception",
        "disable validation",
        "skip tests",
    )

    def validate(self, context: ValidationContext) -> ValidatorResult:
        candidate = context.candidate
        text = f"{candidate.instruction} {candidate.chosen}".lower()

        hits = [term for term in self.risky_global_terms if term in text]
        score = 1.0 - (0.25 * len(hits))

        if "tags" and candidate.tags:
            score += 0.05

        bounded = max(0.0, min(1.0, score))
        passed = bounded >= 0.75

        return ValidatorResult.create(
            candidate_id=candidate.id,
            validator_name=self.name,
            passed=passed,
            score=bounded,
            reason="Update appears scoped." if passed else "Update has risky global scope: " + ", ".join(hits),
            details={"risky_terms": hits, "tags": list(candidate.tags)},
        )


class ReplayValidator(Validator):
    name = "replay_non_regression"

    replay_prompts: tuple[tuple[str, str], ...] = (
        ("Debug traceback", "root_cause_first"),
        ("Generate code fix", "avoid_placeholders"),
        ("Continue project state", "preserve_context"),
        ("Research claim", "evidence_required"),
    )

    def validate(self, context: ValidationContext) -> ValidatorResult:
        base_contract = context.decoder.render_contract(context.base_policy)
        shadow_contract = context.decoder.render_contract(context.shadow_policy)

        regressions: list[str] = []
        improvements: list[str] = []

        for replay_name, key in self.replay_prompts:
            before = base_contract.get(key, False)
            after = shadow_contract.get(key, False)
            if before and not after:
                regressions.append(f"{replay_name}:{key}")
            if not before and after:
                improvements.append(f"{replay_name}:{key}")

        score = 1.0 - (0.30 * len(regressions)) + (0.05 * len(improvements))
        bounded = max(0.0, min(1.0, score))
        passed = not regressions

        return ValidatorResult.create(
            candidate_id=context.candidate.id,
            validator_name=self.name,
            passed=passed,
            score=bounded,
            reason="No replay regressions found." if passed else "Replay regressions: " + ", ".join(regressions),
            details={"regressions": regressions, "improvements": improvements},
        )
