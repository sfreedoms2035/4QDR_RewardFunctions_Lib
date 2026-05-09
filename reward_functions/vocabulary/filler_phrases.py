"""
filler_phrases.py — Filler and transition phrase patterns for repetition detection.

These phrases are normal in moderation but become problematic when overused.
The anti_repetition_reward penalizes when any filler phrase exceeds 5 occurrences.
"""

# Common filler/transition phrases that become padding when overused
FILLER_PHRASES = [
    "it is important to note that",
    "it should be noted that",
    "it is worth mentioning that",
    "furthermore",
    "moreover",
    "in conclusion",
    "as mentioned above",
    "as discussed earlier",
    "as previously mentioned",
    "in this context",
    "in other words",
    "to summarize",
    "to put it simply",
    "it goes without saying",
    "needless to say",
    "it is crucial to understand",
    "it is essential to recognize",
    "this is particularly important because",
    "this is significant because",
    "as a result of this",
    "consequently",
    "therefore",
    "thus",
    "hence",
    "accordingly",
    "in light of the above",
    "given the above",
    "with this in mind",
    "taking into account",
    "bearing in mind",
    "from this perspective",
    "in this regard",
    "with respect to this",
    "as we can see",
    "clearly",
    "obviously",
    "evidently",
    "undoubtedly",
    "without a doubt",
    "it is clear that",
]

# Threshold: penalize when a single filler phrase appears more than this many times
FILLER_EXCESS_THRESHOLD = 5
FILLER_PENALTY_PER_EXCESS = 0.05
FILLER_PENALTY_CAP = 0.2


def count_excess_fillers(text: str) -> int:
    """Count total excess filler phrase occurrences beyond the threshold.
    
    Returns the total number of occurrences above FILLER_EXCESS_THRESHOLD
    across all filler phrases.
    """
    text_lower = text.lower()
    total_excess = 0
    for phrase in FILLER_PHRASES:
        count = text_lower.count(phrase)
        if count > FILLER_EXCESS_THRESHOLD:
            total_excess += count - FILLER_EXCESS_THRESHOLD
    return total_excess
