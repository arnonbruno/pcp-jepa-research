"""
Simulation Environments for PCP-JEPA Research

Tier 1: Smooth continuous dynamics (Pendulum, Cartpole, etc.)
Tier 2: Hybrid/discontinuous dynamics (BouncingBall, StickSlip, etc.) - HAS EVENTS
Tier 3: Partial observability + active perception
"""

from .tier2_hybrid import (
    BouncingBall,
    BouncingBallParams,
    StickSlipBlock,
    StickSlipParams,
    EventType,
    Event,
    EventLog,
    EventMetrics,
)

__all__ = [
    'BouncingBall',
    'BouncingBallParams',
    'StickSlipBlock',
    'StickSlipParams',
    'EventType',
    'Event',
    'EventLog',
    'EventMetrics',
]