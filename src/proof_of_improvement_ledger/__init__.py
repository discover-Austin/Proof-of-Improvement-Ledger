"""
Proof-of-Improvement Ledger.

A small, inspectable architecture for consensus-gated continual learning:
experience -> candidate update -> shadow validation -> accepted/rejected ledger block.
"""

from .models import (
    ConsensusDecision,
    LearningBlock,
    LearningEvent,
    PolicySnapshot,
    ProposedPolicyDiff,
    TrainingCandidate,
    ValidatorResult,
)
from .ledger import LearningLedger
from .policy import DecoderPolicy
from .consensus import ConsensusEngine, ConsensusRule
from .validators import (
    ContradictionValidator,
    EvidenceQualityValidator,
    PreferenceValidator,
    ReplayValidator,
    ScopeValidator,
)

__all__ = [
    "ConsensusDecision",
    "LearningBlock",
    "LearningEvent",
    "PolicySnapshot",
    "ProposedPolicyDiff",
    "TrainingCandidate",
    "ValidatorResult",
    "LearningLedger",
    "DecoderPolicy",
    "ConsensusEngine",
    "ConsensusRule",
    "ContradictionValidator",
    "EvidenceQualityValidator",
    "PreferenceValidator",
    "ReplayValidator",
    "ScopeValidator",
]
