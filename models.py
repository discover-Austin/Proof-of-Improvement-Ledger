from pathlib import Path
import tempfile

from proof_of_improvement_ledger import ConsensusEngine, LearningLedger
from proof_of_improvement_ledger.models import LearningEvent, TrainingCandidate


def main() -> None:
    db_path = Path(tempfile.gettempdir()) / "proof_of_improvement_demo.sqlite3"
    if db_path.exists():
        db_path.unlink()

    ledger = LearningLedger(db_path)
    ledger.initialize()

    event = LearningEvent.create(
        event_type="correction",
        source_text="Debugging answers should give root cause first and full replacement code.",
    )

    candidate = TrainingCandidate.create(
        event_id=event.id,
        instruction="A user gives a traceback and asks for the fix.",
        chosen="State the root cause first, then provide full replacement of the affected function or file with no placeholders.",
        rejected="Provide vague suggestions, TODOs, or partial snippets.",
        tags=["debugging", "code", "preference"],
    )

    with ledger.transaction() as conn:
        ledger.save_event(event, conn=conn)
        ledger.save_candidate(candidate, conn=conn)

    report, block = ConsensusEngine(ledger).evaluate_candidate(candidate)

    print("decision:", report.decision.value)
    print("score:", round(report.consensus_score, 3))
    print("block:", block.block_hash)
    print("status:", ledger.status())


if __name__ == "__main__":
    main()
