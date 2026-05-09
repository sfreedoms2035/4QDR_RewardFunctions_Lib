"""
concepts.py — Task-Specific Rewards for 10-Element Concept Tasks

RF-11: concept_completeness_reward — All 10 elements present and populated
RF-17: mathematical_rigor_reward — Axiomatic base and dependency graph quality
"""

import re

from reward_functions.utils.parsers import extract_block_sections, CONCEPT_BLOCKS
from reward_functions.utils.text_analysis import word_count, count_math_expressions, clamp


# The 10 concept element block names (excluding meta/conversation blocks)
CONCEPT_ELEMENT_BLOCKS = [
    "CONCEPT_HEADER",
    "ONTOLOGICAL_SCAFFOLDING",
    "ABSTRACTION_LEVEL",
    "AXIOMATIC_BASE",
    "RELATIONAL_NETWORK",
    "INFERENTIAL_FRAMEWORK",
    "METHODOLOGICAL_APPARATUS",
    "ILLUSTRATIVE_CORPUS",
    "GOAL_ORIENTATION",
    "LIMITATIONS_AND_RISKS",
    "INTER_CONCEPT_RELATIONSHIPS",
]

# Required sub-headings within elements
ELEMENT_SUBHEADINGS = {
    "ONTOLOGICAL_SCAFFOLDING": ["Definitions", "Taxonomies", "Modular Composition"],
    "ABSTRACTION_LEVEL": ["Description", "Indicators", "Assigned Abstraction"],
    "AXIOMATIC_BASE": ["Assumptions", "Formal Axiomatic", "Adversarial"],
    "RELATIONAL_NETWORK": ["Intra-Concept", "Inter-Subconcept", "Formal Typed", "Formal Graph"],
    "INFERENTIAL_FRAMEWORK": ["Deductions", "Reasoning Patterns"],
    "METHODOLOGICAL_APPARATUS": ["Methods", "Operational Constraints"],
    "ILLUSTRATIVE_CORPUS": ["Exemplars", "Non-Exemplars", "Boundary"],
    "GOAL_ORIENTATION": ["Problem Space", "Domain of Applicability", "Targeted Roles"],
    "LIMITATIONS_AND_RISKS": ["Known Limitations", "Potential Risks", "Mitigation"],
    "INTER_CONCEPT_RELATIONSHIPS": ["Description", "Types of Relationships", "Synergistic"],
}


def concept_completeness_reward(completions: list, **kwargs) -> list:
    """RF-11: 10-element concept completeness check.
    
    Scoring:
        +0.05 per element block present (11 total → max 0.55)
        +0.03 per element with >800 words (max 0.33)
        Sub-heading presence bonus (max 0.12)
    
    Returns: list[float] in [0.0, 1.0]
    """
    rewards = []
    for completion in completions:
        if not completion or not completion.strip():
            rewards.append(0.0)
            continue
        
        sections = extract_block_sections(completion)
        score = 0.0
        
        # Check element presence
        for block in CONCEPT_ELEMENT_BLOCKS:
            if block in sections:
                score += 0.05
                
                # Check word count
                wc = word_count(sections[block])
                if wc >= 800:
                    score += 0.03
                
                # Check sub-headings
                if block in ELEMENT_SUBHEADINGS:
                    for subheading in ELEMENT_SUBHEADINGS[block]:
                        if re.search(rf'\*\*{re.escape(subheading)}', sections[block], re.IGNORECASE):
                            score += 0.004
        
        rewards.append(clamp(score))
    return rewards


def mathematical_rigor_reward(completions: list, **kwargs) -> list:
    """RF-17: Mathematical rigor in concept outputs.
    
    Checks:
        +0.30  Axiomatic base contains math equations
        +0.25  Dependency graph uses set-theoretic notation (V, E, P, C)
        +0.20  No banned formal languages (SMT-LIB/Lisp syntax)
        +0.25  Minimum graph complexity (8+ vertices, 12+ edges defined)
    
    Returns: list[float] in [0.0, 1.0]
    """
    rewards = []
    for completion in completions:
        if not completion or not completion.strip():
            rewards.append(0.0)
            continue
        
        sections = extract_block_sections(completion)
        score = 0.0
        
        # Check axiomatic base for math
        axiomatic = sections.get("AXIOMATIC_BASE", "")
        if axiomatic:
            math_count = count_math_expressions(axiomatic)
            if math_count >= 5:
                score += 0.30
            elif math_count >= 2:
                score += 0.15
        
        # Check relational network for graph notation
        relational = sections.get("RELATIONAL_NETWORK", "")
        if relational:
            has_graph = bool(re.search(r'G\s*=\s*\(', relational))
            has_vertices = bool(re.search(r'V\s*=\s*[\{(]', relational))
            has_edges = bool(re.search(r'E\s*=\s*[\{(]', relational))
            
            graph_elements = sum([has_graph, has_vertices, has_edges])
            score += min(0.25, graph_elements * 0.083)
        
        # Check for banned formal languages
        banned_patterns = [
            r'\(declare-',   # SMT-LIB
            r'\(assert\b',   # SMT-LIB
            r'\(define-fun', # SMT-LIB  
            r'SPECIFICATION\s+\w+\s+IS',  # TLA+
        ]
        has_banned = any(re.search(p, completion) for p in banned_patterns)
        score += 0.0 if has_banned else 0.20
        
        # Check graph complexity
        if relational:
            vertex_defs = len(re.findall(r'(?:v\d+|vertex|node)\s*[:=]', relational, re.IGNORECASE))
            edge_defs = len(re.findall(r'(?:e\d+|edge|→|->)\s*[:=]', relational, re.IGNORECASE))
            
            if vertex_defs >= 8 and edge_defs >= 12:
                score += 0.25
            elif vertex_defs >= 4 or edge_defs >= 6:
                score += 0.12
        
        rewards.append(clamp(score))
    return rewards
