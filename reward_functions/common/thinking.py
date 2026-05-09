"""
thinking.py — Tier 3: Thinking Quality Rewards

RF-06: cot_structure_reward — 8-step CoT template verification
RF-07: reasoning_depth_reward — Genuine analytical depth assessment
RF-09: mandatory_dead_end_reward — Dead-end → diagnose → correct pattern
RF-19: answer_structure_consistency_reward — JSON structure validation
"""

import re
import json

from reward_functions.utils.parsers import (
    extract_think_content, count_cot_substeps, count_cot_major_steps,
    has_fmea_table, has_comparison_matrix, extract_step_content,
    extract_block_sections, extract_json_metadata,
)
from reward_functions.utils.text_analysis import (
    count_math_expressions, char_count, clamp,
)


# ═══════════════════════════════════════════════════════════════════════════════
# RF-06: COT STRUCTURE REWARD
# ═══════════════════════════════════════════════════════════════════════════════

def cot_structure_reward(completions: list, **kwargs) -> list:
    """RF-06: 8-step CoT template structure verification.
    
    Checks that the chain-of-thought follows the mandatory 8-step monologue
    template with all 31 sub-step headers present.
    
    Scoring:
        Base:  found_substeps / 31  (linear, max 0.80)
        Bonus: +0.10 if FMEA table in step 5.2
        Bonus: +0.10 if comparison matrix in step 6.1
    
    Returns: list[float] in [0.0, 1.0]
    """
    rewards = []
    for completion in completions:
        if not completion or not completion.strip():
            rewards.append(0.0)
            continue
        
        think = extract_think_content(completion)
        if not think:
            rewards.append(0.0)
            continue
        
        # Count sub-step headers
        substep_count = count_cot_substeps(think)
        base_score = (substep_count / 31.0) * 0.80
        
        # Bonus checks
        fmea_bonus = 0.10 if has_fmea_table(think) else 0.0
        matrix_bonus = 0.10 if has_comparison_matrix(think) else 0.0
        
        score = base_score + fmea_bonus + matrix_bonus
        rewards.append(clamp(score))
    return rewards


# ═══════════════════════════════════════════════════════════════════════════════
# RF-07: REASONING DEPTH REWARD
# ═══════════════════════════════════════════════════════════════════════════════

# Self-correction language indicators
SELF_CORRECTION_KEYWORDS = [
    "flaw", "error", "incorrect", "instead", "corrected", "however",
    "but this fails", "upon reflection", "reconsidering", "mistake",
    "oversight", "missed", "actually", "wait", "no,", "revising",
    "re-examining", "the issue is", "the problem with",
    "this approach fails because", "a better approach",
    "root cause", "autopsy", "diagnosis", "revision",
]

# Confidence/limitation keywords for step 8
CONFIDENCE_KEYWORDS = [
    "confidence", "score", "certainty", "probability",
    "limitation", "caveat", "assumption", "risk",
    "uncertainty", "constraint", "boundary",
]


def reasoning_depth_reward(completions: list, **kwargs) -> list:
    """RF-07: Genuine analytical depth assessment.
    
    Evaluates whether the reasoning demonstrates real analytical thinking
    versus superficial padding.
    
    Scoring:
        +0.20  Three distinct scenarios in Steps 4.1-4.3
        +0.20  Self-correction language in Steps 5.3-5.4
        +0.20  Mathematical/formal expressions present
        +0.15  Confidence + limitations in Step 8
        +0.25  Dead-end pattern detected (draft → diagnose → correct)
    
    Returns: list[float] in [0.0, 1.0]
    """
    mode = kwargs.get("mode", "lightweight")
    rewards = []
    
    for completion in completions:
        if not completion or not completion.strip():
            rewards.append(0.0)
            continue
        
        think = extract_think_content(completion)
        if not think:
            rewards.append(0.0)
            continue
        
        score = 0.0
        
        # ── Check 1: Scenario Diversity (Steps 4.1-4.3) ────────────────
        step_41 = extract_step_content(think, "4.1")
        step_42 = extract_step_content(think, "4.2")
        step_43 = extract_step_content(think, "4.3")
        
        if step_41 and step_42 and step_43:
            # Lightweight: check that each step has unique content (> 50 chars each)
            if (char_count(step_41) > 50 and char_count(step_42) > 50 
                    and char_count(step_43) > 50):
                # Check word overlap between consecutive steps
                words_41 = set(step_41.lower().split()[:50])
                words_42 = set(step_42.lower().split()[:50])
                words_43 = set(step_43.lower().split()[:50])
                
                overlap_12 = _word_overlap(words_41, words_42)
                overlap_23 = _word_overlap(words_42, words_43)
                
                if overlap_12 < 0.6 and overlap_23 < 0.6:
                    score += 0.20  # Distinct scenarios
                elif overlap_12 < 0.8 or overlap_23 < 0.8:
                    score += 0.10  # Partially distinct
        
        # ── Check 2: Self-Correction (Steps 5.3-5.4) ───────────────────
        step_53 = extract_step_content(think, "5.3")
        step_54 = extract_step_content(think, "5.4")
        combined_5 = (step_53 + " " + step_54).lower()
        
        correction_count = sum(1 for kw in SELF_CORRECTION_KEYWORDS 
                              if kw in combined_5)
        if correction_count >= 3:
            score += 0.20
        elif correction_count >= 1:
            score += 0.10
        
        # ── Check 3: Mathematical/Formal Content ───────────────────────
        math_count = count_math_expressions(think)
        if math_count >= 10:
            score += 0.20
        elif math_count >= 3:
            score += 0.10
        
        # ── Check 4: Confidence + Limitations (Step 8) ─────────────────
        step_81 = extract_step_content(think, "8.1")
        step_83 = extract_step_content(think, "8.3")
        combined_8 = (step_81 + " " + step_83).lower()
        
        confidence_count = sum(1 for kw in CONFIDENCE_KEYWORDS 
                              if kw in combined_8)
        has_numeric_score = bool(re.search(r'\d+(?:\.\d+)?(?:\s*[/%]|\s*out\s*of)', combined_8))
        
        if confidence_count >= 2 and has_numeric_score:
            score += 0.15
        elif confidence_count >= 1:
            score += 0.07
        
        # ── Check 5: Dead-End Pattern ──────────────────────────────────
        score += _detect_dead_end_pattern(think) * 0.25
        
        rewards.append(clamp(score))
    return rewards


def _word_overlap(set_a: set, set_b: set) -> float:
    """Compute Jaccard-style overlap between two word sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _detect_dead_end_pattern(think_content: str) -> float:
    """Detect the mandatory dead-end reasoning pattern.
    
    Looks for a three-phase pattern:
    1. First-pass attempt or "Scenario A" 
    2. Explicit diagnosis of why it fails
    3. Corrected "Scenario B" or revised approach
    
    Returns: 1.0 if full pattern found, 0.5 if partial, 0.0 if absent.
    """
    text_lower = think_content.lower()
    
    # Phase 1 indicators: initial attempt
    phase1_markers = [
        "scenario a", "first attempt", "initial approach", "naive approach",
        "straightforward approach", "let me try", "first pass",
        "preliminary", "draft approach",
    ]
    
    # Phase 2 indicators: diagnosis
    phase2_markers = [
        "root cause", "autopsy", "this fails because", "the flaw",
        "the problem with this", "why this doesn't work",
        "upon closer inspection", "critical error", "however, this",
        "breaks down when", "does not account for", "overlooks",
    ]
    
    # Phase 3 indicators: correction
    phase3_markers = [
        "scenario b", "corrected approach", "revised approach",
        "improved version", "the fix", "better approach",
        "instead, we", "the correction is", "resolving this",
        "a more robust", "corrected version",
    ]
    
    has_phase1 = any(m in text_lower for m in phase1_markers)
    has_phase2 = any(m in text_lower for m in phase2_markers)
    has_phase3 = any(m in text_lower for m in phase3_markers)
    
    if has_phase1 and has_phase2 and has_phase3:
        return 1.0
    elif has_phase2 and has_phase3:
        return 0.7  # Implicit first pass, explicit correction
    elif has_phase2 or has_phase3:
        return 0.3  # Partial pattern
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# RF-19: ANSWER STRUCTURE CONSISTENCY REWARD
# ═══════════════════════════════════════════════════════════════════════════════

def answer_structure_consistency_reward(completions: list, **kwargs) -> list:
    """RF-19: Answer structural consistency.
    
    Verifies that the answer portion follows expected structural conventions:
    - Metadata JSON is parseable (if present)
    - Required block sections are present per task type
    - Content is not stringified JSON where nested objects are expected
    
    Scoring:
        +0.30  JSON metadata parseable
        +0.40  Required sections present (task-type-dependent)
        +0.30  No structural anomalies (broken JSON, stringified objects)
    
    Returns: list[float] in [0.0, 1.0]
    """
    rewards = []
    for completion in completions:
        if not completion or not completion.strip():
            rewards.append(0.0)
            continue
        
        score = 0.0
        sections = extract_block_sections(completion)
        
        # Check 1: Metadata JSON
        metadata = extract_json_metadata(completion)
        if metadata and isinstance(metadata, dict):
            score += 0.30
        elif sections.get("METADATA"):
            score += 0.10  # Present but not parseable
        
        # Check 2: Required sections present
        if sections:
            # At minimum, we need REASONING and at least one content section
            has_reasoning = "REASONING" in sections
            has_content = any(k in sections for k in [
                "CONCEPT_HEADER", "TURN-1-USER", "TURN-4-ASSISTANT",
                "ONTOLOGICAL_SCAFFOLDING", "AXIOMATIC_BASE",
            ])
            
            if has_reasoning and has_content:
                score += 0.40
            elif has_reasoning or has_content:
                score += 0.20
        
        # Check 3: No structural anomalies
        anomaly_count = 0
        # Check for broken JSON fragments
        for section_content in sections.values():
            if section_content.count('{') != section_content.count('}'):
                anomaly_count += 1
            if section_content.count('[') != section_content.count(']'):
                anomaly_count += 1
        
        anomaly_penalty = min(0.30, anomaly_count * 0.05)
        score += 0.30 - anomaly_penalty
        
        rewards.append(clamp(score))
    return rewards
