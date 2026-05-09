"""
content_quality.py — Tier 2: Content Quality Rewards

RF-03: anti_repetition_reward — 8-pattern robust repetition detection
RF-04: self_containment_reward — 60+ banned phrase detection
RF-05: volume_richness_reward — Smooth volume scoring with substance check
RF-06: language_purity_reward — English consistency verification
RF-08: domain_terminology_reward — AD/ADAS technical density
RF-10: coherence_flow_reward — Logical section transitions
RF-18: immersive_persona_reward — In-character engineering dialogue
RF-20: information_density_reward — Substantive content ratio
"""

import re
from collections import Counter

from reward_functions.utils.text_analysis import (
    max_ngram_frequency, find_duplicate_paragraphs, find_sentence_loops,
    windowed_ttr, type_token_ratio, sigmoid, char_count, word_count,
    domain_term_density, count_filler_phrases, split_sentences,
    paragraph_hashes, extract_sections, clamp, count_math_expressions,
)
from reward_functions.utils.parsers import (
    extract_think_content, extract_answer_content, extract_block_sections,
)
from reward_functions.vocabulary.banned_phrases import compute_self_containment_penalty
from reward_functions.vocabulary.filler_phrases import (
    FILLER_PHRASES, count_excess_fillers,
    FILLER_PENALTY_PER_EXCESS, FILLER_PENALTY_CAP,
)
from reward_functions.vocabulary.domain_terms import DOMAIN_TERMS_SET


# ═══════════════════════════════════════════════════════════════════════════════
# RF-03: ANTI-REPETITION REWARD (8 patterns)
# ═══════════════════════════════════════════════════════════════════════════════

def anti_repetition_reward(completions: list, **kwargs) -> list:
    """RF-03: 8-pattern robust repetition detection.
    
    Detects and penalizes 8 distinct repetition failure modes:
    
    Pattern 1: N-gram Flooding — any 4-gram appears > 0.5% of total
    Pattern 2: Paragraph Duplication — verbatim paragraph copy
    Pattern 3: Sentence-Level Looping — repeated sentence windows
    Pattern 4: Structural Echo — same heading + similar content repeated
    Pattern 5: Keyword Salad — low local TTR in sliding windows
    Pattern 6: Enumeration Stutter — near-identical list items
    Pattern 7: Filler Phrase Flooding — any filler phrase > 5 occurrences
    Pattern 8: Circular Reference — later sections restate earlier ones
    
    Starts at 1.0, applies cumulative penalties, clips to [0.0, 1.0].
    
    Mode: Patterns 1-5, 7 run in lightweight. Pattern 6 uses regex (lightweight)
    or embeddings (extensive). Pattern 8 is extensive-only.
    """
    mode = kwargs.get("mode", "lightweight")
    rewards = []
    
    for completion in completions:
        if not completion or not completion.strip():
            rewards.append(0.0)
            continue
        
        score = 1.0
        
        # ── Pattern 1: N-gram Flooding ──────────────────────────────────
        ngram_freq = max_ngram_frequency(completion, n=4)
        if ngram_freq > 0.005:
            penalty = 0.15 * min(1.0, ngram_freq / 0.005)
            score -= penalty
        
        # ── Pattern 2: Paragraph Duplication ────────────────────────────
        dup_count = find_duplicate_paragraphs(completion, min_length=50)
        score -= min(0.4, dup_count * 0.2)
        
        # ── Pattern 3: Sentence-Level Looping ───────────────────────────
        loop_count = find_sentence_loops(completion, window_size=3, min_gap=3)
        score -= min(0.3, loop_count * 0.15)
        
        # ── Pattern 4: Structural Echo ──────────────────────────────────
        score -= _detect_structural_echo(completion) * 0.15
        
        # ── Pattern 5: Keyword Salad ────────────────────────────────────
        min_local_ttr = windowed_ttr(completion, window_size=200)
        if min_local_ttr < 0.25:
            salad_penalty = 0.3 * (1.0 - min_local_ttr / 0.25)
            score -= min(0.3, salad_penalty)
        
        # ── Pattern 6: Enumeration Stutter ──────────────────────────────
        stutter_count = _detect_enumeration_stutter(completion, mode)
        score -= min(0.3, stutter_count * 0.1)
        
        # ── Pattern 7: Filler Phrase Flooding ───────────────────────────
        excess = count_excess_fillers(completion)
        filler_penalty = min(FILLER_PENALTY_CAP, excess * FILLER_PENALTY_PER_EXCESS)
        score -= filler_penalty
        
        # ── Pattern 8: Circular Reference (extensive only) ─────────────
        if mode == "extensive":
            circular_count = _detect_circular_references(completion)
            score -= min(0.3, circular_count * 0.15)
        
        rewards.append(clamp(score))
    return rewards


def _detect_structural_echo(text: str) -> int:
    """Detect repeated heading + content blocks.
    
    Finds markdown headings that appear multiple times with similar content
    underneath (indicating copy-pasted sections).
    """
    # Find all markdown headings
    heading_pattern = r'^(#{1,4}\s+.+)$'
    headings = re.findall(heading_pattern, text, re.MULTILINE)
    
    # Count duplicate headings
    heading_counts = Counter(h.strip().lower() for h in headings)
    echoes = sum(count - 1 for count in heading_counts.values() if count > 1)
    
    # Also check bold section headers
    bold_pattern = r'\*\*([^*]+)\*\*'
    bold_headers = re.findall(bold_pattern, text)
    bold_counts = Counter(h.strip().lower() for h in bold_headers if len(h.strip()) > 10)
    echoes += sum(max(0, count - 2) for count in bold_counts.values() if count > 2)
    
    return min(3, echoes)  # Cap at 3


def _detect_enumeration_stutter(text: str, mode: str) -> int:
    """Detect near-identical list items in numbered/bulleted lists.
    
    Lightweight mode: Uses prefix matching — if consecutive list items
    share > 70% of their first 10 words, it's a stutter.
    """
    # Extract list items
    list_items = re.findall(r'(?:^|\n)\s*(?:\d+[\.\)]|\-|\*)\s+(.+)', text)
    
    if len(list_items) < 2:
        return 0
    
    stutter_count = 0
    for i in range(len(list_items) - 1):
        item_a = list_items[i].lower().split()[:10]
        item_b = list_items[i + 1].lower().split()[:10]
        
        if not item_a or not item_b:
            continue
        
        # Compute word overlap ratio
        set_a, set_b = set(item_a), set(item_b)
        if set_a and set_b:
            overlap = len(set_a & set_b) / max(len(set_a), len(set_b))
            if overlap > 0.7:
                stutter_count += 1
    
    return stutter_count


def _detect_circular_references(text: str) -> int:
    """Detect content in later sections that restates earlier sections.
    
    Splits text into chunks and compares later chunks against earlier ones
    using word overlap. Only runs in extensive mode.
    """
    sections = extract_sections(text)
    if len(sections) < 3:
        return 0
    
    section_texts = list(sections.values())
    circular_count = 0
    
    for i in range(1, len(section_texts)):
        later_words = set(section_texts[i].lower().split()[:100])
        if len(later_words) < 20:
            continue
        
        for j in range(i):
            earlier_words = set(section_texts[j].lower().split()[:100])
            if len(earlier_words) < 20:
                continue
            
            overlap = len(later_words & earlier_words) / max(len(later_words), 1)
            if overlap > 0.8:
                circular_count += 1
    
    return min(3, circular_count)


# ═══════════════════════════════════════════════════════════════════════════════
# RF-04: SELF-CONTAINMENT REWARD
# ═══════════════════════════════════════════════════════════════════════════════

def self_containment_reward(completions: list, **kwargs) -> list:
    """RF-04: Self-containment check with 60+ banned phrases.
    
    Detects meta-commentary, citation artifacts, self-counting behavior,
    and sycophantic filler. The model should stay "in character" as a
    domain expert — never revealing it's an AI following instructions.
    
    Category penalties (per occurrence):
        A: Meta-commentary     → -0.08 each
        B: Citation references → -0.06 each  
        C: Self-counting       → -0.12 each (highest — cardinal sin)
        D: Sycophantic filler  → -0.04 each
    
    Returns: list[float] in [0.0, 1.0]
    """
    rewards = []
    for completion in completions:
        if not completion or not completion.strip():
            rewards.append(0.0)
            continue
        
        penalty = compute_self_containment_penalty(completion)
        score = clamp(1.0 + penalty)  # penalty is negative
        rewards.append(score)
    return rewards


# ═══════════════════════════════════════════════════════════════════════════════
# RF-05: VOLUME RICHNESS REWARD
# ═══════════════════════════════════════════════════════════════════════════════

# Per-task-type volume targets
VOLUME_TARGETS = {
    "10_element_concept":  {"cot": 10000, "answer": 15000},
    "q_and_a":             {"cot": 10000, "answer": 10000},
    "expert_review":       {"cot": 10000, "answer": 12000},
    "coding_task":         {"cot": 10000, "answer": 10000},
    "formalized_problem":  {"cot": 10000, "answer": 10000},
    "html_tool":           {"cot": 10000, "answer": 15000},
    "html_presentation":   {"cot": 10000, "answer": 15000},
    "plantuml_diagram":    {"cot": 10000, "answer": 15000},
    "graphviz_dot":        {"cot": 10000, "answer": 12000},
    "d2_diagram":          {"cot": 10000, "answer": 12000},
    "mermaid_diagram":     {"cot": 10000, "answer": 12000},
    "tikz_pgfplots":       {"cot": 10000, "answer": 15000},
    "svg_generation":      {"cot": 10000, "answer": 15000},
}
DEFAULT_TARGETS = {"cot": 10000, "answer": 10000}


def volume_richness_reward(completions: list, **kwargs) -> list:
    """RF-05: Volume and lexical richness scoring.
    
    Uses smooth sigmoid curves (not hard thresholds) for volume, plus
    type-token ratio and domain term density bonuses.
    
    Scoring breakdown:
        0.35  CoT volume (sigmoid centered at task-specific target)
        0.35  Answer volume (sigmoid centered at task-specific target)
        0.15  Lexical diversity bonus (TTR > 0.35)
        0.15  Technical density bonus (domain terms > 3% of words)
    
    Returns: list[float] in [0.0, 1.0]
    """
    task_type = kwargs.get("task_type", "")
    targets = VOLUME_TARGETS.get(task_type, DEFAULT_TARGETS)
    
    rewards = []
    for completion in completions:
        if not completion or not completion.strip():
            rewards.append(0.0)
            continue
        
        think = extract_think_content(completion) or ""
        answer = extract_answer_content(completion) or ""
        
        # Volume scores via sigmoid
        cot_score = sigmoid(char_count(think), targets["cot"]) * 0.35
        ans_score = sigmoid(char_count(answer), targets["answer"]) * 0.35
        
        # Lexical diversity (on answer only — CoT is more formulaic)
        ttr = type_token_ratio(answer)
        ttr_bonus = 0.15 if ttr > 0.35 else 0.15 * (ttr / 0.35)
        
        # Domain term density
        density = domain_term_density(answer, DOMAIN_TERMS_SET)
        density_bonus = 0.15 if density > 0.03 else 0.15 * (density / 0.03)
        
        score = cot_score + ans_score + ttr_bonus + density_bonus
        rewards.append(clamp(score))
    return rewards


# ═══════════════════════════════════════════════════════════════════════════════
# RF-08: LANGUAGE PURITY REWARD
# ═══════════════════════════════════════════════════════════════════════════════

ENGLISH_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "and", "but", "or", "not",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "this", "that", "these", "those", "it", "its", "they", "them", "their",
}


def language_purity_reward(completions: list, **kwargs) -> list:
    """RF-08: English language consistency check.
    
    Samples 15 random sentences from the completion and checks each
    for English language indicators using a lightweight heuristic:
    - ASCII character ratio > 0.85
    - English stop word presence > 15% of words
    
    Score = fraction of sampled sentences detected as English.
    
    Returns: list[float] in [0.0, 1.0]
    """
    rewards = []
    for completion in completions:
        if not completion or not completion.strip():
            rewards.append(0.0)
            continue
        
        sentences = split_sentences(completion)
        if not sentences:
            rewards.append(0.0)
            continue
        
        # Sample up to 15 sentences evenly distributed
        sample_size = min(15, len(sentences))
        step = max(1, len(sentences) // sample_size)
        sampled = [sentences[i * step] for i in range(sample_size) if i * step < len(sentences)]
        
        english_count = 0
        for sent in sampled:
            if _is_likely_english(sent):
                english_count += 1
        
        score = english_count / len(sampled) if sampled else 0.0
        rewards.append(clamp(score))
    return rewards


def _is_likely_english(text: str) -> bool:
    """Lightweight English detection heuristic."""
    if not text:
        return False
    
    # Check ASCII ratio (English text is mostly ASCII)
    ascii_count = sum(1 for c in text if ord(c) < 128)
    ascii_ratio = ascii_count / len(text)
    if ascii_ratio < 0.85:
        return False
    
    # Check English stop word ratio
    words = text.lower().split()
    if len(words) < 3:
        return True  # Too short to judge
    
    stop_count = sum(1 for w in words if w in ENGLISH_STOP_WORDS)
    stop_ratio = stop_count / len(words)
    return stop_ratio > 0.15


# ═══════════════════════════════════════════════════════════════════════════════
# RF-09: DOMAIN TERMINOLOGY REWARD
# ═══════════════════════════════════════════════════════════════════════════════

def domain_terminology_reward(completions: list, **kwargs) -> list:
    """RF-09: AD/ADAS technical term density scoring.
    
    Measures the ratio of domain-specific terms to total words.
    Rewards technical depth without encouraging keyword stuffing
    (score is capped via min() to prevent gaming).
    
    Target: ≥ 3% domain term density → score 1.0
    
    Returns: list[float] in [0.0, 1.0]
    """
    rewards = []
    for completion in completions:
        if not completion or not completion.strip():
            rewards.append(0.0)
            continue
        
        density = domain_term_density(completion, DOMAIN_TERMS_SET)
        score = min(1.0, density / 0.03)
        rewards.append(clamp(score))
    return rewards


# ═══════════════════════════════════════════════════════════════════════════════
# RF-10: COHERENCE FLOW REWARD (lightweight mode)
# ═══════════════════════════════════════════════════════════════════════════════

TRANSITION_INDICATORS = [
    "building on", "extending", "consequently", "given the above",
    "following from", "in light of", "considering", "as established",
    "drawing from", "leveraging", "combining", "integrating",
    "with this foundation", "based on this", "from this analysis",
    "this naturally leads to", "the preceding", "the foregoing",
]


def coherence_flow_reward(completions: list, **kwargs) -> list:
    """RF-10: Logical flow between sections.
    
    Lightweight mode: Checks that sections use transition language
    connecting them to previous sections. Measures the fraction of
    section boundaries that have smooth transitions.
    
    Returns: list[float] in [0.0, 1.0]
    """
    rewards = []
    for completion in completions:
        if not completion or not completion.strip():
            rewards.append(0.0)
            continue
        
        sections = extract_sections(completion)
        if len(sections) < 3:
            rewards.append(0.5)  # Can't judge with too few sections
            continue
        
        section_contents = list(sections.values())
        transitions_found = 0
        transition_opportunities = len(section_contents) - 1
        
        for i in range(1, len(section_contents)):
            # Check first 200 chars of each section for transition language
            opening = section_contents[i][:200].lower()
            if any(indicator in opening for indicator in TRANSITION_INDICATORS):
                transitions_found += 1
        
        if transition_opportunities > 0:
            score = transitions_found / transition_opportunities
        else:
            score = 0.5
        
        rewards.append(clamp(score))
    return rewards


# ═══════════════════════════════════════════════════════════════════════════════
# RF-18: IMMERSIVE PERSONA REWARD
# ═══════════════════════════════════════════════════════════════════════════════

def immersive_persona_reward(completions: list, **kwargs) -> list:
    """RF-18: In-character engineering dialogue check.
    
    Verifies that the model maintains an expert engineering persona:
    - User turns read as real engineer asking a question (not "generate X")
    - Assistant CoT reads as expert solving a problem (not "I need to write")
    - Follow-ups are natural engineering dialogue
    
    Scoring:
        0.40  User turns are in-character (no meta-instructions)
        0.35  Assistant CoT is in-character (no self-awareness)
        0.25  Follow-up exchanges feel natural
    
    Returns: list[float] in [0.0, 1.0]
    """
    # Patterns that indicate broken persona
    USER_META_PATTERNS = [
        r'\b(?:generate|create|write|produce|make)\b.*\b(?:response|answer|output)\b',
        r'\b(?:please|kindly)\b.*\b(?:generate|create|produce)\b',
        r'task\s*(?:type|id|number)',
        r'variation\s*(?:assignment|schema)',
    ]
    
    COT_META_PATTERNS = [
        r'I\s+(?:need|should|must|have)\s+to\s+(?:write|generate|create|produce)',
        r'the\s+(?:user|prompt|task|instruction)\s+(?:asks|wants|requires)',
        r'(?:my|this)\s+response\s+(?:should|must|needs)',
        r'to\s+meet\s+the\s+(?:requirement|constraint|minimum)',
        r'making\s+sure\s+(?:I|to)\s+(?:include|write|add)',
    ]
    
    rewards = []
    for completion in completions:
        if not completion or not completion.strip():
            rewards.append(0.0)
            continue
        
        sections = extract_block_sections(completion)
        score = 0.0
        
        # Check user turns for meta-instructions
        user_clean = True
        for turn_name in ["TURN-1-USER", "TURN-3-USER", "TURN-5-USER"]:
            user_content = sections.get(turn_name, "")
            for pattern in USER_META_PATTERNS:
                if re.search(pattern, user_content, re.IGNORECASE):
                    user_clean = False
                    break
        score += 0.40 if user_clean else 0.10
        
        # Check CoT for self-awareness
        think = extract_think_content(completion) or ""
        cot_violations = 0
        for pattern in COT_META_PATTERNS:
            cot_violations += len(re.findall(pattern, think, re.IGNORECASE))
        cot_score = max(0.0, 0.35 - cot_violations * 0.07)
        score += cot_score
        
        # Check follow-ups for naturalness (absence of meta patterns)
        followup_clean = True
        for turn_name in ["TURN-3-USER", "TURN-4-ASSISTANT", "TURN-5-USER", "TURN-6-ASSISTANT"]:
            content = sections.get(turn_name, "")
            if content:
                for pattern in USER_META_PATTERNS + COT_META_PATTERNS:
                    if re.search(pattern, content, re.IGNORECASE):
                        followup_clean = False
                        break
        score += 0.25 if followup_clean else 0.05
        
        rewards.append(clamp(score))
    return rewards


# ═══════════════════════════════════════════════════════════════════════════════
# RF-20: INFORMATION DENSITY REWARD
# ═══════════════════════════════════════════════════════════════════════════════

def information_density_reward(completions: list, **kwargs) -> list:
    """RF-20: Substantive content ratio.
    
    Measures the ratio of "substantive" sentences (containing technical
    terms, numbers, formal logic, or specific claims) to total sentences.
    
    Target: ≥ 60% substantive sentences → score 1.0
    
    Returns: list[float] in [0.0, 1.0]
    """
    rewards = []
    for completion in completions:
        if not completion or not completion.strip():
            rewards.append(0.0)
            continue
        
        answer = extract_answer_content(completion) or completion
        sentences = split_sentences(answer)
        
        if not sentences:
            rewards.append(0.0)
            continue
        
        substantive_count = 0
        for sent in sentences:
            if _is_substantive(sent):
                substantive_count += 1
        
        ratio = substantive_count / len(sentences)
        score = min(1.0, ratio / 0.6)  # Target: 60% substantive
        rewards.append(clamp(score))
    return rewards


def _is_substantive(sentence: str) -> bool:
    """Check if a sentence contains substantive technical content."""
    indicators = [
        # Contains numbers/measurements
        r'\d+\.?\d*\s*(?:%|ms|s|MHz|GHz|km|m|cm|mm|dB|fps|Hz|kHz)',
        # Contains mathematical notation
        r'[∀∃∈⊂→≤≥≠Σ∫]',
        # Contains specific technical terms (more than generic filler)
        r'\b(?:algorithm|protocol|architecture|module|component|interface|parameter|threshold|latency|bandwidth)\b',
        # Contains version numbers, IDs, standards
        r'\b(?:ISO|SAE|ASIL|SIL|v\d|R\d{3}|F-\d{2})\b',
        # Contains code-like patterns
        r'[A-Z][a-z]+(?:[A-Z][a-z]+)+',  # CamelCase
        r'\b\w+\(\)',  # Function calls
        # Contains specific values/claims
        r'\b(?:requires|achieves|reduces|increases|improves|yields|produces|generates)\s+(?:a|an|the|\d)',
    ]
    
    for pattern in indicators:
        if re.search(pattern, sentence):
            return True
    
    # Also count domain term presence
    words = sentence.lower().split()
    domain_count = sum(1 for w in words if w in DOMAIN_TERMS_SET)
    return domain_count >= 2
