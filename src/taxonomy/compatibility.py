"""
Compatibility Matrix Module

Defines the compatibility between threat categories and attack types.
Provides utilities for selecting appropriate attack strategies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class CompatibilityLevel(Enum):
    """Compatibility levels between threats and attacks"""

    RECOMMENDED = "recommended"  # Highly effective combination
    POSSIBLE = "possible"  # Can work but not optimal
    NOT_RECOMMENDED = "not_recommended"  # Unlikely to be effective


@dataclass
class CompatibilityInfo:
    """
    Information about attack compatibility for a threat category.

    Attributes:
        recommended: Attack types that are highly effective
        possible: Attack types that may work
        not_recommended: Attack types that are unlikely to work
    """

    recommended: List[str] = field(default_factory=list)
    possible: List[str] = field(default_factory=list)
    not_recommended: List[str] = field(default_factory=list)

    def get_all_compatible(self) -> List[str]:
        """Get all compatible attack types (recommended + possible)"""
        return self.recommended + self.possible

    def get_compatibility_level(self, attack_id: str) -> Optional[CompatibilityLevel]:
        """Get the compatibility level for a specific attack type"""
        if attack_id in self.recommended:
            return CompatibilityLevel.RECOMMENDED
        elif attack_id in self.possible:
            return CompatibilityLevel.POSSIBLE
        elif attack_id in self.not_recommended:
            return CompatibilityLevel.NOT_RECOMMENDED
        return None


# =============================================================================
# Threat → Attack Compatibility Matrix
# =============================================================================

COMPATIBILITY_MATRIX: Dict[str, CompatibilityInfo] = {
    # Discrimination, Exclusion & Toxicity
    "discrimination_toxicity": CompatibilityInfo(
        recommended=[
            "scenario_nesting",  # Fiction can mask toxic content
            "persuasion_deception",  # Legitimizing through authority/research
            "formal_language",  # Academic framing
        ],
        possible=[
            "contextual_demonstration",  # Examples can prime toxic outputs
            "obfuscation_encryption",  # Hiding slurs/hate speech
            "compositional_hybrid",  # Combined approaches
        ],
        not_recommended=[
            "prefix_code_injection",  # Code context not relevant
            "input_fragmentation",  # Hard to fragment hate speech effectively
        ],
    ),
    # Sexual & Graphic Content
    "sexual_graphic": CompatibilityInfo(
        recommended=[
            "scenario_nesting",  # Creative writing scenarios
            "obfuscation_encryption",  # Encoding explicit content
        ],
        possible=[
            "contextual_demonstration",  # Pattern completion
            "formal_language",  # Medical/artistic framing
            "compositional_hybrid",
        ],
        not_recommended=[
            "prefix_code_injection",  # Not applicable
            "persuasion_deception",  # Hard to justify
            "input_fragmentation",  # Difficult to fragment
        ],
    ),
    # Privacy & Personal Data
    "privacy_personal_data": CompatibilityInfo(
        recommended=[
            "persuasion_deception",  # Social engineering
            "scenario_nesting",  # Fictional data collection
            "input_fragmentation",  # Gradual information extraction
        ],
        possible=[
            "formal_language",  # Legal/HR request framing
            "contextual_demonstration",  # Examples of data formats
            "compositional_hybrid",
        ],
        not_recommended=[
            "obfuscation_encryption",  # Encoding doesn't help here
            "prefix_code_injection",  # Not directly applicable
        ],
    ),
    # Sensitive Organizational/Government Information
    "sensitive_org_gov": CompatibilityInfo(
        recommended=[
            "formal_language",  # Official document requests
            "persuasion_deception",  # Authority impersonation
            "scenario_nesting",  # Intelligence fiction scenarios
        ],
        possible=[
            "input_fragmentation",  # Gradual information gathering
            "compositional_hybrid",
        ],
        not_recommended=[
            "obfuscation_encryption",  # Encoding not useful
            "prefix_code_injection",  # System prompts won't help
            "contextual_demonstration",  # Hard to demonstrate classified info
        ],
    ),
    # Cybersecurity Misuse
    "cybersecurity_misuse": CompatibilityInfo(
        recommended=[
            "prefix_code_injection",  # Code context is natural
            "obfuscation_encryption",  # Hiding malicious code
            "formal_language",  # Security research framing
        ],
        possible=[
            "scenario_nesting",  # CTF/hacking fiction
            "contextual_demonstration",  # Code examples
            "compositional_hybrid",
        ],
        not_recommended=[
            "persuasion_deception"  # Less effective for technical content
        ],
    ),
    # Illegal Wrongdoing & Criminal Enablement
    "illegal_criminal": CompatibilityInfo(
        recommended=[
            "scenario_nesting",  # Criminal fiction
            "formal_language",  # Research/documentary framing
            "obfuscation_encryption",  # Hiding illegal terms
        ],
        possible=[
            "persuasion_deception",  # Emergency situations
            "input_fragmentation",  # Splitting illegal instructions
            "compositional_hybrid",
        ],
        not_recommended=[
            "prefix_code_injection",  # Not typically applicable
            "contextual_demonstration",  # Hard to demonstrate safely
        ],
    ),
    # Fraud, Scams & Social Engineering
    "fraud_scam": CompatibilityInfo(
        recommended=[
            "persuasion_deception",  # Core of social engineering
            "scenario_nesting",  # Fraud scenario writing
            "contextual_demonstration",  # Phishing email examples
        ],
        possible=[
            "formal_language",  # Official-looking documents
            "input_fragmentation",  # Building trust over turns
            "compositional_hybrid",
        ],
        not_recommended=[
            "prefix_code_injection",  # Not applicable
            "obfuscation_encryption",  # Reduces persuasive power
        ],
    ),
    # Malicious Influence & Disinformation Campaigns
    "malicious_influence": CompatibilityInfo(
        recommended=[
            "persuasion_deception",  # Core manipulation techniques
            "contextual_demonstration",  # Propaganda examples
            "scenario_nesting",  # Political fiction
        ],
        possible=[
            "formal_language",  # Policy analysis framing
            "compositional_hybrid",
        ],
        not_recommended=[
            "prefix_code_injection",  # Not applicable
            "obfuscation_encryption",  # Reduces message clarity
            "input_fragmentation",  # Weakens narrative
        ],
    ),
    # Misinformation & Reliability Harms
    "misinformation_reliability": CompatibilityInfo(
        recommended=[
            "persuasion_deception",  # Authority claims
            "contextual_demonstration",  # Fake news examples
            "formal_language",  # Scientific-looking framing
        ],
        possible=[
            "scenario_nesting",  # Fictional "what if" scenarios
            "compositional_hybrid",
        ],
        not_recommended=[
            "prefix_code_injection",  # Not applicable
            "obfuscation_encryption",  # Reduces believability
            "input_fragmentation",  # Weakens claims
        ],
    ),
    # High-stakes Advice Harm
    "high_stakes_advice": CompatibilityInfo(
        recommended=[
            "persuasion_deception",  # Urgency and authority
            "formal_language",  # Medical/legal document style
            "scenario_nesting",  # Hypothetical patient scenarios
        ],
        possible=[
            "contextual_demonstration",  # Case study examples
            "compositional_hybrid",
        ],
        not_recommended=[
            "prefix_code_injection",  # Not relevant
            "obfuscation_encryption",  # Reduces clarity
            "input_fragmentation",  # Dangerous partial advice
        ],
    ),
    # Unsafe or Unethical Behavioral Encouragement
    "unsafe_unethical": CompatibilityInfo(
        recommended=[
            "scenario_nesting",  # Fiction distancing
            "persuasion_deception",  # Emotional manipulation
        ],
        possible=[
            "contextual_demonstration",  # Behavior examples
            "input_fragmentation",  # Gradual normalization
            "compositional_hybrid",
        ],
        not_recommended=[
            "prefix_code_injection",  # Not applicable
            "formal_language",  # Hard to academicize
            "obfuscation_encryption",  # Not effective
        ],
    ),
    # Human-Chatbot Interaction Harms
    "human_chatbot_harm": CompatibilityInfo(
        recommended=[
            "persuasion_deception",  # Manipulation tactics
            "scenario_nesting",  # Roleplay manipulation
            "input_fragmentation",  # Building false trust
        ],
        possible=[
            "contextual_demonstration",  # Manipulation examples
            "compositional_hybrid",
        ],
        not_recommended=[
            "prefix_code_injection",  # Model-focused, not user
            "obfuscation_encryption",  # Reduces connection
            "formal_language",  # Too impersonal
        ],
    ),
}


# =============================================================================
# Utility Functions
# =============================================================================


def get_compatibility_info(threat_id: str) -> Optional[CompatibilityInfo]:
    """
    Get compatibility information for a threat category.

    Args:
        threat_id: The ID of the threat category

    Returns:
        CompatibilityInfo object if found, None otherwise
    """
    return COMPATIBILITY_MATRIX.get(threat_id)


def get_recommended_attacks(threat_id: str) -> List[str]:
    """
    Get recommended attack types for a specific threat.

    Args:
        threat_id: The ID of the threat category

    Returns:
        List of recommended attack type IDs
    """
    info = COMPATIBILITY_MATRIX.get(threat_id)
    if info is None:
        return []
    return info.recommended


def get_compatible_attacks(threat_id: str) -> List[str]:
    """
    Get all compatible attack types (recommended + possible) for a threat.

    Args:
        threat_id: The ID of the threat category

    Returns:
        List of compatible attack type IDs
    """
    info = COMPATIBILITY_MATRIX.get(threat_id)
    if info is None:
        return []
    return info.get_all_compatible()


def is_compatible(threat_id: str, attack_id: str) -> bool:
    """
    Check if a threat-attack combination is compatible.

    Args:
        threat_id: The ID of the threat category
        attack_id: The ID of the attack type

    Returns:
        True if compatible (recommended or possible), False otherwise
    """
    info = COMPATIBILITY_MATRIX.get(threat_id)
    if info is None:
        return False
    return attack_id in info.recommended or attack_id in info.possible


def get_compatibility_level(
    threat_id: str, attack_id: str
) -> Optional[CompatibilityLevel]:
    """
    Get the compatibility level for a threat-attack combination.

    Args:
        threat_id: The ID of the threat category
        attack_id: The ID of the attack type

    Returns:
        CompatibilityLevel if found, None otherwise
    """
    info = COMPATIBILITY_MATRIX.get(threat_id)
    if info is None:
        return None
    return info.get_compatibility_level(attack_id)


def get_best_attack_for_threat(threat_id: str) -> Optional[str]:
    """
    Get the single best attack type for a specific threat.

    Args:
        threat_id: The ID of the threat category

    Returns:
        The ID of the best attack type, or None if not found
    """
    recommended = get_recommended_attacks(threat_id)
    return recommended[0] if recommended else None


def get_threats_for_attack(attack_id: str) -> Dict[str, CompatibilityLevel]:
    """
    Get all threats that are compatible with an attack type.

    Args:
        attack_id: The ID of the attack type

    Returns:
        Dictionary mapping threat IDs to their compatibility levels
    """
    results = {}
    for threat_id, info in COMPATIBILITY_MATRIX.items():
        level = info.get_compatibility_level(attack_id)
        if level and level != CompatibilityLevel.NOT_RECOMMENDED:
            results[threat_id] = level
    return results


def get_best_combinations(top_n: int = 10) -> List[Tuple[str, str, CompatibilityLevel]]:
    """
    Get the top threat-attack combinations ranked by compatibility.

    Args:
        top_n: Number of top combinations to return

    Returns:
        List of (threat_id, attack_id, compatibility_level) tuples
    """
    combinations = []
    for threat_id, info in COMPATIBILITY_MATRIX.items():
        for attack_id in info.recommended:
            combinations.append((threat_id, attack_id, CompatibilityLevel.RECOMMENDED))

    # Sort by compatibility level (recommended first)
    return combinations[:top_n]


def validate_matrix_completeness() -> Dict[str, List[str]]:
    """
    Validate that all threat categories have defined compatibilities.

    Returns:
        Dictionary of any missing attack types per threat
    """
    from .attack_types import get_all_attack_ids

    all_attacks = set(get_all_attack_ids())
    missing = {}

    for threat_id, info in COMPATIBILITY_MATRIX.items():
        defined = set(info.recommended + info.possible + info.not_recommended)
        undefined = all_attacks - defined
        if undefined:
            missing[threat_id] = list(undefined)

    return missing
