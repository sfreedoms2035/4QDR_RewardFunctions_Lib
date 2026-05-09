"""
parsers.py — Parsing utilities for extracting structured content from model output.

Handles:
- <think>...</think> tag extraction
- !!!!!BLOCK!!!!! section extraction
- Conversation turn parsing
- JSON metadata extraction
"""

import re
import json
from typing import Optional, Dict, List, Tuple


def extract_think_content(text: str) -> Optional[str]:
    """Extract content between <think> and </think> tags.
    
    Returns None if tags are not found.
    Handles multiline content and whitespace.
    """
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def has_think_tags(text: str) -> Tuple[bool, bool]:
    """Check for presence of opening and closing think tags.
    
    Returns (has_opening, has_closing) tuple.
    """
    has_opening = bool(re.search(r'<think>', text, re.IGNORECASE))
    has_closing = bool(re.search(r'</think>', text, re.IGNORECASE))
    return has_opening, has_closing


def count_think_blocks(text: str) -> int:
    """Count the number of <think>...</think> blocks in text.
    
    More than 1 indicates duplicate/nested blocks (a structural error).
    """
    return len(re.findall(r'<think>', text, re.IGNORECASE))


def extract_answer_content(text: str) -> str:
    """Extract the answer portion (everything after </think>).
    
    If no think tags are found, returns the full text.
    """
    match = re.search(r'</think>\s*(.*)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_block_sections(text: str) -> Dict[str, str]:
    """Extract sections delimited by !!!!!BLOCK_NAME!!!!! markers.
    
    Returns a dict mapping block names to their content.
    """
    pattern = r'!!!!!([A-Z_\-0-9]+)!!!!!'
    parts = re.split(pattern, text)
    
    sections = {}
    for i in range(1, len(parts) - 1, 2):
        name = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections[name] = content
    return sections


# The list of expected block names for the 10-element concept tasks
CONCEPT_BLOCKS = [
    "METADATA",
    "REASONING",
    "TURN-1-USER",
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
    "TURN-3-USER",
    "TURN-4-ASSISTANT",
    "TURN-5-USER",
    "TURN-6-ASSISTANT",
]

# The 31 sub-step headers for the 8-step CoT template
COT_SUBSTEP_HEADERS = [
    "1.1", "1.2",
    "2.1", "2.2", "2.3", "2.4", "2.5",
    "3.1", "3.2", "3.3", "3.4", "3.5", "3.6",
    "4.1", "4.2", "4.3",
    "5.1", "5.2", "5.3", "5.4", "5.5",
    "6.1", "6.2", "6.3",
    "7.1", "7.2", "7.3",
    "8.1", "8.2", "8.3", "8.4",
]

# Major step headers (8 steps)
COT_MAJOR_STEPS = [
    "Step 1", "Step 2", "Step 3", "Step 4",
    "Step 5", "Step 6", "Step 7", "Step 8",
]


def count_cot_substeps(think_content: str) -> int:
    """Count how many of the 31 CoT sub-step headers are present."""
    if not think_content:
        return 0
    count = 0
    for header in COT_SUBSTEP_HEADERS:
        # Match patterns like "1.1", "1.1.", "1.1:", "1.1 -", "**1.1**"
        pattern = rf'(?:^|\n)\s*(?:\*\*)?{re.escape(header)}(?:\.\s|\s*[-:)]|\*\*)'
        if re.search(pattern, think_content):
            count += 1
    return count


def count_cot_major_steps(think_content: str) -> int:
    """Count how many of the 8 major step headers are present."""
    if not think_content:
        return 0
    count = 0
    for step in COT_MAJOR_STEPS:
        if re.search(rf'\b{re.escape(step)}\b', think_content, re.IGNORECASE):
            count += 1
    return count


def has_fmea_table(think_content: str) -> bool:
    """Check if the CoT contains a FMEA-style markdown table (Step 5.2)."""
    if not think_content:
        return False
    # Look for markdown table indicators near step 5.2
    section_5 = _extract_step_range(think_content, "5.2", "5.3")
    if not section_5:
        section_5 = think_content
    return bool(re.search(r'\|.*\|.*\|', section_5))


def has_comparison_matrix(think_content: str) -> bool:
    """Check if the CoT contains a comparison matrix table (Step 6.1)."""
    if not think_content:
        return False
    section_6 = _extract_step_range(think_content, "6.1", "6.2")
    if not section_6:
        section_6 = think_content
    return bool(re.search(r'\|.*\|.*\|', section_6))


def extract_step_content(think_content: str, step: str) -> str:
    """Extract the content of a specific sub-step from CoT."""
    if not think_content:
        return ""
    next_steps = {
        "4.1": "4.2", "4.2": "4.3", "4.3": "5.1",
        "5.3": "5.4", "5.4": "5.5",
        "8.1": "8.2", "8.2": "8.3", "8.3": "8.4",
    }
    next_step = next_steps.get(step)
    return _extract_step_range(think_content, step, next_step)


def _extract_step_range(text: str, start_step: str, end_step: Optional[str]) -> str:
    """Extract text between two step markers."""
    start_pattern = rf'(?:^|\n)\s*(?:\*\*)?{re.escape(start_step)}(?:\.\s|\s*[-:)]|\*\*)'
    start_match = re.search(start_pattern, text)
    if not start_match:
        return ""
    
    start_pos = start_match.end()
    
    if end_step:
        end_pattern = rf'(?:^|\n)\s*(?:\*\*)?{re.escape(end_step)}(?:\.\s|\s*[-:)]|\*\*)'
        end_match = re.search(end_pattern, text[start_pos:])
        if end_match:
            return text[start_pos:start_pos + end_match.start()].strip()
    
    # Return everything from start to end of text (or next 2000 chars)
    return text[start_pos:start_pos + 2000].strip()


def extract_conversation_turns(text: str) -> List[Dict[str, str]]:
    """Extract conversation turns from block-structured output.
    
    Returns a list of dicts with 'role' and 'content' keys.
    """
    blocks = extract_block_sections(text)
    turns = []
    
    turn_mapping = [
        ("TURN-1-USER", "user"),
        ("TURN-2-ASSISTANT", "assistant"),  # Main answer (concepts elements)
        ("TURN-3-USER", "user"),
        ("TURN-4-ASSISTANT", "assistant"),
        ("TURN-5-USER", "user"),
        ("TURN-6-ASSISTANT", "assistant"),
    ]
    
    for block_name, role in turn_mapping:
        content = blocks.get(block_name, "")
        if content:
            turns.append({"role": role, "content": content})
    
    return turns


def extract_json_metadata(text: str) -> Optional[dict]:
    """Try to extract JSON metadata from the METADATA block."""
    blocks = extract_block_sections(text)
    metadata_text = blocks.get("METADATA", "")
    if not metadata_text:
        return None
    
    # Try to find JSON object in the metadata text
    try:
        # Look for {...} pattern
        match = re.search(r'\{[^{}]*\}', metadata_text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        pass
    return None


def extract_findings(text: str) -> List[str]:
    """Extract numbered findings (F-01, F-02, etc.) from review output."""
    pattern = r'F-(\d{2})[:\s]'
    matches = re.findall(pattern, text)
    return [f"F-{m}" for m in matches]


def count_list_items(text: str) -> int:
    """Count numbered or bulleted list items in text."""
    numbered = len(re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', text))
    bulleted = len(re.findall(r'(?:^|\n)\s*[-*•]\s', text))
    return numbered + bulleted
