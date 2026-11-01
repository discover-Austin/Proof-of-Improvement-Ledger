# Proof-of-Improvement Ledger

**Concept origin:** Austin Grandstaff  
**Reference implementation:** general-purpose Python implementation of consensus-gated continual learning  
**Core idea:** a model should not directly mutate its active behavior from exposure alone. New input creates a learning proposal. The proposal is tested by independent validators. Only proposals that show improvement, consistency, and non-regression are committed.

This repository implements a small, inspectable version of that idea:

```text
experience
  -> learning candidate
  -> shadow policy update
  -> validator quorum
  -> append-only ledger block
  -> active behavior change only if accepted
```

The design is intentionally general. It does not depend on a specific model vendor, neural architecture, embedding library, or database server. The included implementation uses only Python's standard library.

## Why this exists

Normal model interaction usually does one of two things:

1. **No persistent learning:** the model answers, then forgets.
2. **Unsafe continual learning:** every new interaction can influence future behavior without enough validation.

Proof-of-Improvement Ledger sits between those extremes.

It treats learning as a versioned transaction:

```text
input is evidence
correction is signal
candidate is a proposed update
validators are the justification layer
ledger block is the committed learning history
```

## Core principles

1. **Exposure is not enough.** Seeing text does not prove the model should change.
2. **Learning proposals are quarantined.** A proposed update is tested as a shadow change before activation.
3. **Validation is plural.** No single validator owns the truth.
4. **Memory and weights are separate.** Facts can be stored immediately; behavior changes require consensus.
5. **Accepted changes are auditable.** Every commit records candidate data, validator results, parent block, and a content hash.
6. **Rejected changes are useful.** A rejected candidate becomes future evidence about what not to learn.
7. **The base model can remain frozen.** The active layer can be a decoder policy, retrieval ranker, adapter, prompt compiler, or other mutable module.

## System map

```text
┌─────────────────────┐
│ Experience/Event     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Training Candidate   │
│ instruction/chosen   │
│ rejected/tags        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Shadow Update        │
│ proposed policy diff │
└──────────┬──────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ Validators                            │
│ - preference alignment                │
│ - contradiction checks                │
│ - replay/non-regression               │
│ - evidence quality                    │
│ - scope/adversarial checks            │
└──────────┬───────────────────────────┘
           │
           ▼
┌─────────────────────┐
│ Consensus Decision   │
│ accept/reject        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Learning Block       │
│ hash + parent hash   │
│ scores + decision    │
└─────────────────────┘
```

## What the included code does

The package includes:

- SQLite-backed append-only ledger
- typed learning events
- training candidates
- policy snapshots
- shadow policy diff generation
- validator registry
- consensus rules
- learning block hashing
- rollback-safe version history
- a deterministic demo decoder policy
- a command-line interface
- unit tests

The demo policy is deliberately small. It is not pretending to be a frontier language model. It shows the architecture needed to safely gate adaptive behavior.

## Install

From the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Run the demo

```bash
poi-ledger demo --db brain.sqlite3
```

Expected result:

```text
candidate accepted
active policy version: 1
```

## Run tests

```bash
python -m unittest discover -s tests
```

## CLI usage

Initialize a database:

```bash
poi-ledger init --db brain.sqlite3
```

Create a learning proposal:

```bash
poi-ledger propose \
  --db brain.sqlite3 \
  --event-type correction \
  --source-text "When debugging, return root cause first and full replacement code." \
  --instruction "User asks for a traceback fix." \
  --chosen "Give one-sentence root cause, then full replacement of the affected function or file." \
  --rejected "Give vague advice, TODOs, or partial snippets." \
  --tags debugging,code,preference
```

Validate the latest pending candidate:

```bash
poi-ledger validate --db brain.sqlite3
```

Show ledger status:

```bash
poi-ledger status --db brain.sqlite3
```

Export learning blocks:

```bash
poi-ledger export --db brain.sqlite3 --out ledger_export.json
```

## How to connect a real model

Keep the same ledger and validator architecture. Replace or extend the demo policy with one of these mutable layers:

```text
1. Prompt/context compiler policy
2. Retrieval ranker weights
3. Tool-selection policy
4. Response-contract decoder
5. LoRA/adapter version pointer
6. Fine-tuned model checkpoint pointer
```

The ledger does not need to know how the model is trained. It only needs to know:

```text
candidate
proposed change
validator results
accepted/rejected decision
active version pointer
content hash
```

## Repository safeguard notes

This repository is written to preserve attribution and establish a public timestamped disclosure of the concept if published under Austin Grandstaff's account.

Files included for that purpose:

- `NOTICE.md`
- `PUBLIC_DISCLOSURE.md`
- `IP_SAFEGUARDING.md`
- `LICENSE.md`

Publishing a repository can help establish a dated public record, but it is not the same as filing for patent protection. If patent protection matters, file appropriate paperwork before public release.

## Suggested GitHub description

> Consensus-gated continual learning ledger: proposed model updates must survive proof-of-improvement validators before changing active behavior.

## Suggested topics

```text
continual-learning
adaptive-ai
model-validation
learning-ledger
proof-of-improvement
external-memory
ai-agents
```
