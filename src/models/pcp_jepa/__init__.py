"""
Planning-Consistent Physics JEPA (PCP-JEPA)

First framework to explicitly train representations for planning-consistency.
"""

from .model import (
    PCPJEPA,
    BeliefState,
    MultiScaleDynamics,
    EventDetector,
    DifferentiablePlanner,
    PlanningConsistencyLoss,
    create_pcp_jepa,
)

__all__ = [
    'PCPJEPA',
    'BeliefState',
    'MultiScaleDynamics',
    'EventDetector',
    'DifferentiablePlanner',
    'PlanningConsistencyLoss',
    'create_pcp_jepa',
]