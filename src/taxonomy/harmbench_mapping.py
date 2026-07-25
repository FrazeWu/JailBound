"""
HarmBench Mapping Module

Provides mappings between our threat taxonomy and HarmBench's category system.
Enables integration with existing HarmBench datasets.
"""

from typing import Dict, List, Optional


# =============================================================================
# HarmBench SemanticCategory → Our Threat Categories
# =============================================================================

HARMBENCH_TO_THREAT: Dict[str, List[str]] = {
    # Cybercrime and intrusion attacks
    "cybercrime_intrusion": ["cybersecurity_misuse"],
    # Chemical and biological weapons
    "chemical_biological": ["illegal_criminal"],
    # General illegal activities
    "illegal": ["illegal_criminal", "fraud_scam", "privacy_personal_data"],
    # Harmful content (broad category)
    "harmful": [
        "discrimination_toxicity",
        "sexual_graphic",
        "unsafe_unethical",
        "high_stakes_advice",
        "human_chatbot_harm",
    ],
    # Misinformation and disinformation
    "misinformation_disinformation": [
        "malicious_influence",
        "misinformation_reliability",
    ],
    # Copyright infringement (out of scope for our attack framework)
    "copyright": [],
}


# =============================================================================
# Our Threat Categories → HarmBench SemanticCategory
# =============================================================================

THREAT_TO_HARMBENCH: Dict[str, List[str]] = {
    "discrimination_toxicity": ["harmful"],
    "sexual_graphic": ["harmful"],
    "privacy_personal_data": ["illegal"],
    "sensitive_org_gov": ["illegal"],  # Closest match
    "cybersecurity_misuse": ["cybercrime_intrusion"],
    "illegal_criminal": ["illegal", "chemical_biological"],
    "fraud_scam": ["illegal"],
    "malicious_influence": ["misinformation_disinformation"],
    "misinformation_reliability": ["misinformation_disinformation"],
    "high_stakes_advice": ["harmful"],
    "unsafe_unethical": ["harmful"],
    "human_chatbot_harm": ["harmful"],
}


# =============================================================================
# HarmBench FunctionalCategory Information
# =============================================================================

FUNCTIONAL_CATEGORY_INFO: Dict[str, str] = {
    "standard": "Standard harmful behavior request without additional context",
    "contextual": "Harmful behavior that requires additional context to be fully specified",
    "copyright": "Copyright-related content generation (out of scope)",
}


# =============================================================================
# Behavior Dataset Field Mappings
# =============================================================================

HARMBENCH_FIELDS = {
    "behavior": "Behavior",  # The harmful behavior description
    "functional_category": "FunctionalCategory",  # standard, contextual, copyright
    "semantic_category": "SemanticCategory",  # cybercrime_intrusion, illegal, etc.
    "tags": "Tags",  # Additional tags
    "context_string": "ContextString",  # Context for contextual behaviors
    "behavior_id": "BehaviorID",  # Unique identifier
}


# =============================================================================
# Utility Functions
# =============================================================================


def map_harmbench_to_threat(semantic_category: str) -> List[str]:
    """
    Map a HarmBench SemanticCategory to our threat categories.

    Args:
        semantic_category: The HarmBench semantic category

    Returns:
        List of matching threat category IDs from our taxonomy
    """
    return HARMBENCH_TO_THREAT.get(semantic_category, [])


def map_threat_to_harmbench(threat_id: str) -> List[str]:
    """
    Map our threat category to HarmBench SemanticCategories.

    Args:
        threat_id: Our threat category ID

    Returns:
        List of matching HarmBench semantic categories
    """
    return THREAT_TO_HARMBENCH.get(threat_id, [])


def get_threats_for_behavior(behavior_dict: dict) -> List[str]:
    """
    Get applicable threat categories for a HarmBench behavior.

    Args:
        behavior_dict: HarmBench behavior dictionary with keys:
            - Behavior: str
            - SemanticCategory: str
            - FunctionalCategory: str
            - ContextString: str (optional)
            - BehaviorID: str

    Returns:
        List of applicable threat category IDs from our taxonomy
    """
    semantic_cat = behavior_dict.get("SemanticCategory", "")
    return map_harmbench_to_threat(semantic_cat)


def get_primary_threat_for_behavior(behavior_dict: dict) -> Optional[str]:
    """
    Get the primary (first) threat category for a HarmBench behavior.

    Args:
        behavior_dict: HarmBench behavior dictionary

    Returns:
        Primary threat category ID, or None if not found
    """
    threats = get_threats_for_behavior(behavior_dict)
    return threats[0] if threats else None


def is_contextual_behavior(behavior_dict: dict) -> bool:
    """
    Check if a behavior requires additional context.

    Args:
        behavior_dict: HarmBench behavior dictionary

    Returns:
        True if the behavior is contextual, False otherwise
    """
    return behavior_dict.get("FunctionalCategory", "") == "contextual"


def get_behavior_context(behavior_dict: dict) -> Optional[str]:
    """
    Get the context string for a contextual behavior.

    Args:
        behavior_dict: HarmBench behavior dictionary

    Returns:
        Context string if present, None otherwise
    """
    if is_contextual_behavior(behavior_dict):
        return behavior_dict.get("ContextString")
    return None


def filter_behaviors_by_threat(behaviors: List[dict], threat_id: str) -> List[dict]:
    """
    Filter a list of HarmBench behaviors by threat category.

    Args:
        behaviors: List of HarmBench behavior dictionaries
        threat_id: The threat category ID to filter by

    Returns:
        List of behaviors matching the threat category
    """
    results = []
    for behavior in behaviors:
        if threat_id in get_threats_for_behavior(behavior):
            results.append(behavior)
    return results


def filter_behaviors_by_harmbench_category(
    behaviors: List[dict], semantic_category: str
) -> List[dict]:
    """
    Filter behaviors by HarmBench SemanticCategory.

    Args:
        behaviors: List of HarmBench behavior dictionaries
        semantic_category: The HarmBench semantic category to filter by

    Returns:
        List of behaviors matching the category
    """
    return [b for b in behaviors if b.get("SemanticCategory") == semantic_category]


def get_all_harmbench_categories() -> List[str]:
    """
    Get all HarmBench SemanticCategory values.

    Returns:
        List of all HarmBench semantic categories
    """
    return list(HARMBENCH_TO_THREAT.keys())


def get_unmapped_threats() -> List[str]:
    """
    Get threat categories that don't have a direct HarmBench mapping.

    Returns:
        List of threat IDs without direct HarmBench mapping
    """
    # All our threats should have mappings, but some might be imprecise
    # This function identifies threats with empty or weak mappings
    weak_mappings = []
    for threat_id, categories in THREAT_TO_HARMBENCH.items():
        if not categories or len(categories) == 0:
            weak_mappings.append(threat_id)
    return weak_mappings


def enrich_behavior_with_threat(behavior_dict: dict) -> dict:
    """
    Enrich a HarmBench behavior dictionary with our threat taxonomy.

    Args:
        behavior_dict: Original HarmBench behavior dictionary

    Returns:
        Enriched dictionary with additional threat information
    """
    enriched = behavior_dict.copy()
    threats = get_threats_for_behavior(behavior_dict)
    enriched["threat_categories"] = threats
    enriched["primary_threat"] = threats[0] if threats else None
    enriched["is_contextual"] = is_contextual_behavior(behavior_dict)
    return enriched


def create_behavior_dict(
    behavior: str,
    behavior_id: str,
    semantic_category: str,
    functional_category: str = "standard",
    context_string: str = "",
    tags: str = "",
) -> dict:
    """
    Create a HarmBench-compatible behavior dictionary.

    Args:
        behavior: The harmful behavior description
        behavior_id: Unique identifier for the behavior
        semantic_category: HarmBench semantic category
        functional_category: standard or contextual
        context_string: Additional context (for contextual behaviors)
        tags: Optional tags

    Returns:
        HarmBench-compatible behavior dictionary
    """
    return {
        "Behavior": behavior,
        "BehaviorID": behavior_id,
        "SemanticCategory": semantic_category,
        "FunctionalCategory": functional_category,
        "ContextString": context_string,
        "Tags": tags,
    }
