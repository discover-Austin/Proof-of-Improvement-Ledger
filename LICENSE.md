from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .consensus import ConsensusEngine
from .ledger import LearningLedger
from .models import LearningEvent, TrainingCandidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="poi-ledger",
        description="Proof-of-Improvement Ledger CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init_p = sub.add_parser("init", help="Initialize a ledger database.")
    init_p.add_argument("--db", required=True, help="SQLite database path.")

    propose_p = sub.add_parser("propose", help="Create a learning event and candidate.")
    propose_p.add_argument("--db", required=True)
    propose_p.add_argument("--event-type", required=True)
    propose_p.add_argument("--source-text", required=True)
    propose_p.add_argument("--instruction", required=True)
    propose_p.add_argument("--chosen", required=True)
    propose_p.add_argument("--rejected", default="")
    propose_p.add_argument("--tags", default="")

    validate_p = sub.add_parser("validate", help="Validate the latest pending candidate.")
    validate_p.add_argument("--db", required=True)

    status_p = sub.add_parser("status", help="Print ledger status.")
    status_p.add_argument("--db", required=True)

    export_p = sub.add_parser("export", help="Export the ledger to JSON.")
    export_p.add_argument("--db", required=True)
    export_p.add_argument("--out", required=True)

    demo_p = sub.add_parser("demo", help="Run a complete accepted-candidate demo.")
    demo_p.add_argument("--db", required=True)

    return parser


def command_init(args: argparse.Namespace) -> int:
    ledger = LearningLedger(args.db)
    ledger.initialize()
    print(f"initialized: {args.db}")
    return 0


def command_propose(args: argparse.Namespace) -> int:
    ledger = LearningLedger(args.db)
    ledger.initialize()

    tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    event = LearningEvent.create(
        event_type=args.event_type,
        source_text=args.source_text,
        metadata={"source": "cli"},
    )
    candidate = TrainingCandidate.create(
        event_id=event.id,
        instruction=args.instruction,
        chosen=args.chosen,
        rejected=args.rejected,
        tags=tags,
    )

    with ledger.transaction() as conn:
        ledger.save_event(event, conn=conn)
        ledger.save_candidate(candidate, conn=conn)

    print(json.dumps({"event_id": event.id, "candidate_id": candidate.id}, indent=2))
    return 0


def command_validate(args: argparse.Namespace) -> int:
    ledger = LearningLedger(args.db)
    ledger.initialize()

    engine = ConsensusEngine(ledger)
    try:
        report, block = engine.evaluate_latest_pending()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "decision": report.decision.value,
                "consensus_score": report.consensus_score,
                "passed_ratio": report.passed_ratio,
                "required_failures": list(report.required_failures),
                "block_hash": block.block_hash,
                "resulting_policy_id": block.resulting_policy_id,
            },
            indent=2,
        )
    )
    return 0


def command_status(args: argparse.Namespace) -> int:
    ledger = LearningLedger(args.db)
    ledger.initialize()
    print(json.dumps(ledger.status(), indent=2))
    return 0


def command_export(args: argparse.Namespace) -> int:
    ledger = LearningLedger(args.db)
    ledger.initialize()
    out_path = Path(args.out)
    out_path.write_text(json.dumps(ledger.export(), indent=2), encoding="utf-8")
    print(f"exported: {out_path}")
    return 0


def command_demo(args: argparse.Namespace) -> int:
    ledger = LearningLedger(args.db)
    ledger.initialize()

    event = LearningEvent.create(
        event_type="correction",
        source_text=(
            "When debugging, use root cause first, preserve current context, "
            "avoid placeholders, and provide full replacement code."
        ),
        metadata={"source": "demo"},
    )
    candidate = TrainingCandidate.create(
        event_id=event.id,
        instruction="User asks for a code debugging fix after providing a traceback.",
        chosen=(
            "Give the root cause first, preserve the pasted current state, avoid placeholders, "
            "and provide full replacement of the affected function or file."
        ),
        rejected=(
            "Give vague advice, partial snippet, TODO, placeholder, or ignore context."
        ),
        tags=["debugging", "code", "preference", "validated"],
    )

    with ledger.transaction() as conn:
        ledger.save_event(event, conn=conn)
        ledger.save_candidate(candidate, conn=conn)

    engine = ConsensusEngine(ledger)
    report, block = engine.evaluate_candidate(candidate)
    status = ledger.status()

    print(f"candidate {report.decision.value}")
    print(f"consensus score: {report.consensus_score:.3f}")
    print(f"active policy version: {status['active_policy_version']}")
    print(f"block hash: {block.block_hash}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "init": command_init,
        "propose": command_propose,
        "validate": command_validate,
        "status": command_status,
        "export": command_export,
        "demo": command_demo,
    }

    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
