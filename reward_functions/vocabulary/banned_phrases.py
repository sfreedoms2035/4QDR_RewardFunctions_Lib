"""
banned_phrases.py — Comprehensive banned vocabulary for self-containment detection.

Organized into 4 categories with different penalty weights.
Total: 60+ phrases that indicate the model is breaking character,
referencing its own instructions, or engaging in sycophantic behavior.
"""

# Category A — Meta-Commentary (model self-awareness, instruction awareness)
# Penalty: -0.08 each
META_COMMENTARY = [
    "prompt",
    "generate",
    "the user requests",
    "this task",
    "meta-strategy",
    "the document",
    "the text",
    "source material",
    "I was asked to",
    "my instructions",
    "as an AI",
    "as a language model",
    "I'm designed to",
    "I cannot",
    "I apologize",
    "I'm sorry",
    "happy to help",
    "here is",
    "sure!",
    "of course!",
    "certainly!",
    "let me",
    "I'll now",
    "I will proceed to",
    "as requested",
    "you asked me to",
    "your question",
    "in this response",
    "I'll provide",
    "I'll explain",
    "I'll describe",
    "I'll outline",
    "I will generate",
    "I'll create",
    "I'll demonstrate",
]

# Category B — Document/Citation References (breaks immersion)
# Penalty: -0.06 each
CITATION_REFERENCES = [
    "according to the text",
    "as shown in figure",
    "based on the document",
    "the paper states",
    "the author mentions",
    "as described in section",
    "see table",
    "see figure",
    "ref.",
    "cf.",
    "cited in",
    "bibliography",
    "the source document",
    "the original text",
    "as written in",
    "the passage states",
    "page number",
    "appendix",
    "the abstract mentions",
    "as noted in the paper",
]

# Category C — Self-Counting / Constraint Awareness (cardinal sin)
# Penalty: -0.12 each (highest — this is blatant constraint gaming)
SELF_COUNTING = [
    "I need to write",
    "I must ensure",
    "I need to generate",
    "characters long",
    "word count",
    "minimum length",
    "15 findings",
    "10,000 characters",
    "10000 characters",
    "800 words",
    "to meet the requirement",
    "as required by the prompt",
    "per the instructions",
    "the constraint says",
    "I should include at least",
    "to satisfy the minimum",
    "making sure I reach",
    "I need to expand this to",
    "let me add more to reach",
    "the target is",
    "I must write at least",
    "need more content to fill",
    "padding to reach",
    "to meet the word count",
    "minimum character count",
    "extending this section to",
]

# Category D — Conversational Filler (anti-sycophancy)
# Penalty: -0.04 each
SYCOPHANTIC_FILLER = [
    "great question",
    "that's a great point",
    "excellent observation",
    "I'd be happy to",
    "feel free to ask",
    "don't hesitate",
    "hope this helps",
    "let me know if",
    "is there anything else",
    "good question",
    "wonderful question",
    "that's an interesting",
    "I appreciate your",
    "thank you for asking",
    "I'm glad you asked",
    "absolutely!",
    "great observation",
]

# All categories combined for quick iteration
ALL_BANNED = {
    "meta_commentary": (META_COMMENTARY, 0.08),
    "citation_references": (CITATION_REFERENCES, 0.06),
    "self_counting": (SELF_COUNTING, 0.12),
    "sycophantic_filler": (SYCOPHANTIC_FILLER, 0.04),
}


def compute_self_containment_penalty(text: str) -> float:
    """Compute total penalty from banned phrase detection.
    
    Returns a negative value (or 0.0 if clean).
    The caller should add this to a starting score of 1.0.
    """
    text_lower = text.lower()
    total_penalty = 0.0
    
    for category_name, (phrases, weight) in ALL_BANNED.items():
        for phrase in phrases:
            count = text_lower.count(phrase.lower())
            if count > 0:
                total_penalty += count * weight
    
    return -total_penalty
