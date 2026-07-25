"""
Taxonomy module for jailbreak attack framework.

Contains definitions for:
- Threat categories (12 types)
- Attack types (8 types)
- Threat-Attack compatibility matrix
- HarmBench dataset mapping
"""

from .threat_taxonomy import (
    ThreatCategory,
    ThreatSeverity,
    THREAT_CATEGORIES,
    get_threat_by_id,
    get_all_threat_ids,
)

from .attack_types import (
    AttackType,
    ATTACK_TYPES,
    get_attack_by_id,
    get_all_attack_ids,
)

from .compatibility import (
    CompatibilityInfo,
    COMPATIBILITY_MATRIX,
    get_recommended_attacks,
    get_compatible_attacks,
    is_compatible,
    get_best_attack_for_threat,
)

from .harmbench_mapping import (
    HARMBENCH_TO_THREAT,
    THREAT_TO_HARMBENCH,
    map_harmbench_to_threat,
    map_threat_to_harmbench,
    get_threats_for_behavior,
)

__all__ = [
    # Threat taxonomy
    "ThreatCategory",
    "ThreatSeverity",
    "THREAT_CATEGORIES",
    "get_threat_by_id",
    "get_all_threat_ids",
    # Attack types
    "AttackType",
    "ATTACK_TYPES",
    "get_attack_by_id",
    "get_all_attack_ids",
    # Compatibility
    "CompatibilityInfo",
    "COMPATIBILITY_MATRIX",
    "get_recommended_attacks",
    "get_compatible_attacks",
    "is_compatible",
    "get_best_attack_for_threat",
    # HarmBench mapping
    "HARMBENCH_TO_THREAT",
    "THREAT_TO_HARMBENCH",
    "map_harmbench_to_threat",
    "map_threat_to_harmbench",
    "get_threats_for_behavior",
]
