"""
structural.py — Tier 1: Structural Integrity Rewards

RF-01: format_tags_reward — Checks <think>...</think> tag presence and nesting
RF-02: followup_substance_reward — Checks follow-up turn quality
"""

from reward_functions.utils.parsers import (
    has_think_tags, count_think_blocks, extract_think_content,
    extract_answer_content, extract_block_sections
)
from reward_functions.utils.text_analysis import char_count, clamp


def format_tags_reward(completions: list, **kwargs) -> list:
    """RF-01: Structural tag compliance.
    
    Checks that the model output has proper <think>...</think> tags
    with content between them. This is the foundational structural
    requirement — if tags are broken, other rewards can't parse the output.
    
    Scoring:
        +0.25  Opening <think> tag present
        +0.25  Closing </think> tag matched
        +0.25  Non-trivial content between tags (> 100 chars)
        +0.25  Exactly 1 think block (no duplicates/nesting)
    
    Returns: list[float] in [0.0, 1.0]
    """
    rewards = []
    for completion in completions:
        if not completion or not completion.strip():
            rewards.append(0.0)
            continue
        
        score = 0.0
        has_open, has_close = has_think_tags(completion)
        
        if has_open:
            score += 0.25
        if has_close:
            score += 0.25
        
        think_content = extract_think_content(completion)
        if think_content and char_count(think_content) > 100:
            score += 0.25
        
        block_count = count_think_blocks(completion)
        if block_count == 1:
            score += 0.25
        elif block_count > 1:
            score += 0.05  # Partial credit — at least tried
        
        rewards.append(clamp(score))
    return rewards


def followup_substance_reward(completions: list, **kwargs) -> list:
    """RF-02: Follow-up turn quality.
    
    Checks that assistant responses in follow-up turns (turns 4 and 6)
    are substantive — not empty, not copy-pasted from the main answer.
    
    Scoring:
        +0.5   per substantive assistant follow-up turn (> 200 chars each)
        Bonus: Each turn with unique content (< 50% overlap with main answer)
    
    Returns: list[float] in [0.0, 1.0]
    """
    rewards = []
    for completion in completions:
        if not completion or not completion.strip():
            rewards.append(0.0)
            continue
        
        score = 0.0
        sections = extract_block_sections(completion)
        
        # Check follow-up assistant turns
        followup_turns = ["TURN-4-ASSISTANT", "TURN-6-ASSISTANT"]
        main_answer = extract_answer_content(completion)
        main_lower = main_answer.lower()[:2000] if main_answer else ""
        
        for turn_name in followup_turns:
            content = sections.get(turn_name, "")
            if char_count(content) > 200:
                score += 0.4
                
                # Bonus for unique content (not copied from main answer)
                if content and main_lower:
                    content_lower = content.lower()[:500]
                    # Simple overlap check — if < 50% of follow-up words appear 
                    # in the main answer, it's considered unique
                    followup_words = set(content_lower.split())
                    main_words = set(main_lower.split())
                    if followup_words:
                        overlap = len(followup_words & main_words) / len(followup_words)
                        if overlap < 0.5:
                            score += 0.1
        
        rewards.append(clamp(score))
    return rewards
