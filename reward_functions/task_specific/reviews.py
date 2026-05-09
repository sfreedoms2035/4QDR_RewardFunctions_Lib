"""
reviews.py — Task-Specific Rewards for Expert Review Tasks
qa.py — Task-Specific Rewards for Q&A Dialectic Tasks

RF-12: review_findings_reward — 15-finding audit report structure
RF-13: qa_dialectic_reward — White-paper style dialectic resolution
"""

import re

from reward_functions.utils.text_analysis import word_count, char_count, clamp
from reward_functions.utils.parsers import extract_block_sections, extract_findings


def review_findings_reward(completions: list, **kwargs) -> list:
    """RF-12: Expert review findings completeness.
    
    Checks for the mandatory 15-finding audit structure with proper
    sub-fields per finding.
    
    Scoring:
        min(count/15, 1.0) * 0.40  — Finding count
        +0.03 per complete finding (ID + severity + description + recommendation) → max 0.45
        +0.15 if rewritten/corrected artifact section present (>1000 words)
    
    Returns: list[float] in [0.0, 1.0]
    """
    rewards = []
    for completion in completions:
        if not completion or not completion.strip():
            rewards.append(0.0)
            continue
        
        score = 0.0
        answer = completion  # Use full text for review tasks
        
        # Count findings
        findings = extract_findings(answer)
        finding_count = len(findings)
        score += min(finding_count / 15.0, 1.0) * 0.40
        
        # Check finding completeness
        finding_fields = ["severity", "classification", "description", 
                         "recommendation", "root cause", "impact"]
        complete_findings = 0
        for finding_id in findings:
            # Look for the finding block and check for sub-fields
            pattern = rf'{re.escape(finding_id)}.*?(?=F-\d{{2}}|\Z)'
            match = re.search(pattern, answer, re.DOTALL)
            if match:
                finding_text = match.group().lower()
                fields_found = sum(1 for f in finding_fields if f in finding_text)
                if fields_found >= 3:  # At least 3 of 6 fields
                    complete_findings += 1
        
        score += min(0.45, complete_findings * 0.03)
        
        # Check for corrected artifact
        corrected_patterns = [
            r'rewritten.*artifact', r'corrected.*artifact',
            r'revised.*version', r'corrected.*version',
        ]
        has_corrected = any(re.search(p, answer, re.IGNORECASE) for p in corrected_patterns)
        if has_corrected:
            # Check if the corrected section has substance
            corrected_section = answer[answer.lower().rfind("corrected"):] if "corrected" in answer.lower() else ""
            if word_count(corrected_section) >= 1000:
                score += 0.15
            elif word_count(corrected_section) >= 500:
                score += 0.08
        
        rewards.append(clamp(score))
    return rewards


def qa_dialectic_reward(completions: list, **kwargs) -> list:
    """RF-13: Q&A dialectic quality assessment.
    
    Checks that the Q&A output follows a white-paper style dialectic
    resolution with proper structure.
    
    Scoring:
        +0.15  Executive summary present
        +0.20  Trade-off matrix (markdown table)
        +0.20  Strategic resolution section
        +0.20  Multiple stakeholder perspectives addressed
        +0.25  Actionable recommendations with specifics
    
    Returns: list[float] in [0.0, 1.0]
    """
    # Structural indicators for quality dialectic
    EXEC_SUMMARY_MARKERS = [
        "executive summary", "overview", "abstract", "tldr", "summary",
    ]
    TRADEOFF_MARKERS = [
        "trade-off", "tradeoff", "comparison matrix", "versus",
        "pros and cons", "advantages and disadvantages",
    ]
    RESOLUTION_MARKERS = [
        "strategic resolution", "recommendation", "proposed solution",
        "our approach", "the solution", "resolution strategy",
        "synthesis", "final recommendation",
    ]
    STAKEHOLDER_MARKERS = [
        "stakeholder", "engineer", "manager", "architect", "safety",
        "developer", "operator", "regulator", "auditor",
        "from the perspective of", "considering the viewpoint",
    ]
    ACTION_MARKERS = [
        "action item", "next step", "implementation plan",
        "deliverable", "milestone", "timeline", "deadline",
        "responsible", "owner", "priority",
    ]
    
    rewards = []
    for completion in completions:
        if not completion or not completion.strip():
            rewards.append(0.0)
            continue
        
        text_lower = completion.lower()
        score = 0.0
        
        # Executive summary
        if any(m in text_lower for m in EXEC_SUMMARY_MARKERS):
            score += 0.15
        
        # Trade-off matrix (bonus if markdown table found)
        has_tradeoff = any(m in text_lower for m in TRADEOFF_MARKERS)
        has_table = bool(re.search(r'\|.*\|.*\|', completion))
        if has_tradeoff and has_table:
            score += 0.20
        elif has_tradeoff or has_table:
            score += 0.10
        
        # Strategic resolution
        resolution_count = sum(1 for m in RESOLUTION_MARKERS if m in text_lower)
        if resolution_count >= 2:
            score += 0.20
        elif resolution_count >= 1:
            score += 0.10
        
        # Multiple stakeholder perspectives
        stakeholder_count = sum(1 for m in STAKEHOLDER_MARKERS if m in text_lower)
        if stakeholder_count >= 3:
            score += 0.20
        elif stakeholder_count >= 1:
            score += 0.10
        
        # Actionable recommendations
        action_count = sum(1 for m in ACTION_MARKERS if m in text_lower)
        if action_count >= 3:
            score += 0.25
        elif action_count >= 1:
            score += 0.12
        
        rewards.append(clamp(score))
    return rewards
