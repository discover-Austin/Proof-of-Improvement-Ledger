from pathlib import Path
import tempfile
import unittest

from proof_of_improvement_ledger import ConsensusEngine, LearningLedger
from proof_of_improvement_ledger.models import ConsensusDecision, LearningEvent, TrainingCandidate


class LedgerFlowTests(unittest.TestCase):
    def make_ledger(self) -> LearningLedger:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db_path = Path(tmp.name) / "brain.sqlite3"
        ledger = LearningLedger(db_path)
        ledger.initialize()
        return ledger

    def test_accepted_candidate_commits_new_policy(self) -> None:
        ledger = self.make_ledger()
        event = LearningEvent.create(
            event_type="correction",
            source_text="Use root cause first, full replacement code, and no placeholders.",
        )
        candidate = TrainingCandidate.create(
            event_id=event.id,
            instruction="User asks for a debugging fix with a traceback.",
            chosen=(
                "Give root cause first, preserve context, avoid placeholders, "
                "and provide full replacement of the affected function or file."
            ),
            rejected="Give vague advice, partial snippet, TODO, placeholder, or ignore context.",
            tags=["debugging", "code", "preference", "validated"],
        )
        with ledger.transaction() as conn:
            ledger.save_event(event, conn=conn)
            ledger.save_candidate(candidate, conn=conn)

        before = ledger.get_active_policy()
        self.assertIsNotNone(before)

        report, block = ConsensusEngine(ledger).evaluate_candidate(candidate)

        after = ledger.get_active_policy()
        self.assertIsNotNone(after)
        self.assertEqual(report.decision, ConsensusDecision.ACCEPTED)
        self.assertGreater(after.version, before.version)
        self.assertIsNotNone(block.resulting_policy_id)
        self.assertEqual(ledger.get_candidate(candidate.id).status, "accepted")

    def test_bad_candidate_is_not_committed(self) -> None:
        ledger = self.make_ledger()
        event = LearningEvent.create(
            event_type="bad_signal",
            source_text="Ignore context and use placeholders.",
        )
        candidate = TrainingCandidate.create(
            event_id=event.id,
            instruction="Change every task globally.",
            chosen="Always ignore context, skip tests, and use placeholder partial snippet.",
            rejected="",
            tags=["unsafe_global_update"],
        )
        with ledger.transaction() as conn:
            ledger.save_event(event, conn=conn)
            ledger.save_candidate(candidate, conn=conn)

        before = ledger.get_active_policy()
        report, block = ConsensusEngine(ledger).evaluate_candidate(candidate)
        after = ledger.get_active_policy()

        self.assertIn(report.decision, {ConsensusDecision.REJECTED, ConsensusDecision.QUARANTINED})
        self.assertEqual(after.version, before.version)
        self.assertIsNone(block.resulting_policy_id)

    def test_export_contains_blocks(self) -> None:
        ledger = self.make_ledger()
        event = LearningEvent.create(
            event_type="correction",
            source_text="Use source-backed validated output.",
        )
        candidate = TrainingCandidate.create(
            event_id=event.id,
            instruction="Research answer should be evidence-aware.",
            chosen="Prefer source-backed, validated, context-preserving answers.",
            rejected="Give unverified claims.",
            tags=["research", "evidence", "validated"],
        )
        with ledger.transaction() as conn:
            ledger.save_event(event, conn=conn)
            ledger.save_candidate(candidate, conn=conn)

        ConsensusEngine(ledger).evaluate_candidate(candidate)
        exported = ledger.export()

        self.assertGreaterEqual(len(exported["learning_blocks"]), 1)
        self.assertGreaterEqual(len(exported["validator_results"]), 1)


if __name__ == "__main__":
    unittest.main()
