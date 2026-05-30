from __future__ import annotations

from dataclasses import dataclass

from .ledger import LearningLedger
from .models import ConsensusDecision, LearningBlock, PolicySnapshot, TrainingCandidate, ValidatorResult
from .policy import DecoderPolicy
from .validators import (
    ContradictionValidator,
    EvidenceQualityValidator,
    PreferenceValidator,
    ReplayValidator,
    ScopeValidator,
    ValidationContext,
    Validator,
)


@dataclass(frozen=True)
class ConsensusRule:
    required_validators: tuple[str, ...] = (
        "preference_alignment",
        "contradiction_check",
        "replay_non_regression",
    )
    minimum_average_score: float = 0.74
    minimum_pass_ratio: float = 0.80
    quarantine_average_score: float = 0.60


@dataclass(frozen=True)
class ConsensusReport:
    decision: ConsensusDecision
    consensus_score: float
    passed_ratio: float
    required_failures: tuple[str, ...]
    validator_results: tuple[ValidatorResult, ...]


class ConsensusEngine:
    def __init__(
        self,
        ledger: LearningLedger,
        decoder: DecoderPolicy | None = None,
        validators: list[Validator] | None = None,
        rule: ConsensusRule | None = None,
    ):
        self.ledger = ledger
        self.decoder = decoder or DecoderPolicy()
        self.validators = validators or [
            PreferenceValidator(),
            EvidenceQualityValidator(),
            ContradictionValidator(),
            ScopeValidator(),
            ReplayValidator(),
        ]
        self.rule = rule or ConsensusRule()

    def evaluate_candidate(self, candidate: TrainingCandidate) -> tuple[ConsensusReport, LearningBlock]:
        self.ledger.initialize()
        with self.ledger.connect() as conn:
            base_policy = self.ledger.get_active_policy(conn=conn)
            if base_policy is None:
                raise RuntimeError("ledger has no active policy after initialization")

            diff = self.decoder.propose_diff(candidate, base_policy)
            self.ledger.save_policy_diff(diff, conn=conn)

            shadow_policy = self.decoder.apply_diff(
                base_policy,
                diff,
                metadata={"status": "shadow", "candidate_id": candidate.id},
            )

            context = ValidationContext(
                candidate=candidate,
                base_policy=base_policy,
                shadow_policy=shadow_policy,
                diff=diff,
                decoder=self.decoder,
            )

            results = tuple(validator.validate(context) for validator in self.validators)
            for result in results:
                self.ledger.save_validator_result(result, conn=conn)

            report = self._decide(results)
            resulting_policy: PolicySnapshot | None = None

            if report.decision == ConsensusDecision.ACCEPTED:
                resulting_policy = self.decoder.apply_diff(
                    base_policy,
                    diff,
                    metadata={
                        "status": "active",
                        "candidate_id": candidate.id,
                        "accepted_by": "proof_of_improvement_consensus",
                    },
                )
                self.ledger.save_policy(resulting_policy, conn=conn)

            block = LearningBlock.create(
                parent_hash=self.ledger.latest_block_hash(conn=conn),
                candidate_id=candidate.id,
                base_policy_id=base_policy.id,
                resulting_policy_id=resulting_policy.id if resulting_policy else None,
                decision=report.decision,
                consensus_score=report.consensus_score,
                validator_result_ids=[result.id for result in results],
                memory_diff={
                    "policy_delta": diff.delta,
                    "rationale": diff.rationale,
                    "validator_names": [result.validator_name for result in results],
                },
            )

            self.ledger.save_learning_block(block, conn=conn)
            self.ledger.mark_candidate(candidate.id, report.decision.value, conn=conn)

            return report, block

    def evaluate_latest_pending(self) -> tuple[ConsensusReport, LearningBlock]:
        candidate = self.ledger.latest_pending_candidate()
        if candidate is None:
            raise RuntimeError("no pending candidate")
        return self.evaluate_candidate(candidate)

    def _decide(self, results: tuple[ValidatorResult, ...]) -> ConsensusReport:
        if not results:
            raise ValueError("at least one validator result is required")

        scores = [result.score for result in results]
        average_score = sum(scores) / len(scores)
        pass_ratio = sum(1 for result in results if result.passed) / len(results)

        result_by_name = {result.validator_name: result for result in results}
        required_failures = tuple(
            name
            for name in self.rule.required_validators
            if name not in result_by_name or not result_by_name[name].passed
        )

        if (
            not required_failures
            and average_score >= self.rule.minimum_average_score
            and pass_ratio >= self.rule.minimum_pass_ratio
        ):
            decision = ConsensusDecision.ACCEPTED
        elif average_score >= self.rule.quarantine_average_score:
            decision = ConsensusDecision.QUARANTINED
        else:
            decision = ConsensusDecision.REJECTED

        return ConsensusReport(
            decision=decision,
            consensus_score=average_score,
            passed_ratio=pass_ratio,
            required_failures=required_failures,
            validator_results=results,
        )
