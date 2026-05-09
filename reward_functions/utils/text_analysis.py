"""
text_analysis.py — Core text analysis utilities for reward functions.

Provides n-gram computation, type-token ratio, paragraph hashing,
sigmoid scoring, and other text metrics used across all reward functions.
"""

import re
import math
import hashlib
from collections import Counter
from typing import List, Tuple, Optional


def sigmoid(value: float, center: float, steepness_factor: float = 0.2) -> float:
    """Smooth sigmoid mapping for volume-based scores.
    
    Returns a value between 0 and 1, centered at `center`.
    When value == center, returns 0.5.
    `steepness_factor` controls how fast the sigmoid transitions (smaller = sharper).
    
    Example: sigmoid(12000, 10000, 0.2) ≈ 0.73
    """
    if center <= 0:
        return 1.0 if value > 0 else 0.0
    x = (value - center) / (center * steepness_factor)
    x = max(-10, min(10, x))  # Prevent overflow
    return 1.0 / (1.0 + math.exp(-x))


def compute_ngrams(text: str, n: int = 4) -> Counter:
    """Compute n-gram frequency distribution from text.
    
    Tokenizes on whitespace after lowering and stripping punctuation.
    Returns a Counter of n-gram tuples.
    """
    words = _tokenize(text)
    if len(words) < n:
        return Counter()
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return Counter(ngrams)


def max_ngram_frequency(text: str, n: int = 4) -> float:
    """Return the frequency ratio of the most common n-gram.
    
    A ratio above 0.005 (0.5%) indicates problematic repetition.
    Returns 0.0 for texts shorter than n words.
    """
    counter = compute_ngrams(text, n)
    if not counter:
        return 0.0
    total = sum(counter.values())
    if total == 0:
        return 0.0
    _, max_count = counter.most_common(1)[0]
    return max_count / total


def type_token_ratio(text: str) -> float:
    """Compute Type-Token Ratio (unique words / total words).
    
    Higher TTR indicates more lexical diversity.
    Typical good text: 0.4-0.7. Padding/filler: < 0.25.
    """
    words = _tokenize(text)
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def windowed_ttr(text: str, window_size: int = 200) -> float:
    """Compute minimum TTR across sliding windows of `window_size` words.
    
    Detects local pockets of low diversity (keyword salad) even if
    the overall TTR is acceptable.
    """
    words = _tokenize(text)
    if len(words) < window_size:
        return type_token_ratio(text)
    
    min_ttr = 1.0
    for i in range(0, len(words) - window_size + 1, window_size // 2):
        window = words[i:i + window_size]
        ttr = len(set(window)) / len(window)
        min_ttr = min(min_ttr, ttr)
    return min_ttr


def paragraph_hashes(text: str, min_length: int = 50) -> List[Tuple[str, str]]:
    """Split text into paragraphs and return (hash, content) tuples.
    
    Only includes paragraphs with at least `min_length` characters.
    Hash is computed on normalized (lowered, whitespace-collapsed) text.
    """
    paragraphs = re.split(r'\n\s*\n', text)
    results = []
    for p in paragraphs:
        p_stripped = p.strip()
        if len(p_stripped) < min_length:
            continue
        normalized = re.sub(r'\s+', ' ', p_stripped.lower())
        h = hashlib.md5(normalized.encode('utf-8')).hexdigest()
        results.append((h, p_stripped))
    return results


def find_duplicate_paragraphs(text: str, min_length: int = 50) -> int:
    """Count the number of duplicate paragraphs in the text."""
    hashes = paragraph_hashes(text, min_length)
    hash_counts = Counter(h for h, _ in hashes)
    return sum(count - 1 for count in hash_counts.values() if count > 1)


def sliding_sentence_windows(text: str, window_size: int = 3) -> List[str]:
    """Generate hashes of sliding windows of consecutive sentences."""
    sentences = split_sentences(text)
    if len(sentences) < window_size:
        return []
    windows = []
    for i in range(len(sentences) - window_size + 1):
        window_text = ' '.join(sentences[i:i + window_size])
        normalized = re.sub(r'\s+', ' ', window_text.lower().strip())
        windows.append(hashlib.md5(normalized.encode('utf-8')).hexdigest())
    return windows


def find_sentence_loops(text: str, window_size: int = 3, min_gap: int = 3) -> int:
    """Count sentence-level loops (repeated consecutive sentence windows).
    
    `min_gap`: minimum number of windows apart to count as a loop (not adjacent overlap).
    """
    window_hashes = sliding_sentence_windows(text, window_size)
    loops = 0
    seen = {}
    for i, h in enumerate(window_hashes):
        if h in seen and (i - seen[h]) >= min_gap:
            loops += 1
        seen[h] = i
    return loops


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using a simple regex heuristic."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def count_filler_phrases(text: str, filler_list: List[str]) -> int:
    """Count occurrences of filler phrases in text (case-insensitive)."""
    text_lower = text.lower()
    return sum(text_lower.count(phrase.lower()) for phrase in filler_list)


def count_pattern_matches(text: str, patterns: List[str]) -> int:
    """Count occurrences of regex patterns in text."""
    total = 0
    for pattern in patterns:
        total += len(re.findall(pattern, text, re.IGNORECASE))
    return total


def extract_sections(text: str, delimiter_pattern: str = r'!!!!!([A-Z_\-0-9]+)!!!!!') -> dict:
    """Split text by section delimiters and return a dict of {name: content}."""
    parts = re.split(delimiter_pattern, text)
    sections = {}
    for i in range(1, len(parts) - 1, 2):
        name = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections[name] = content
    return sections


def word_count(text: str) -> int:
    """Count words in text."""
    return len(_tokenize(text))


def char_count(text: str) -> int:
    """Count characters in text (excluding leading/trailing whitespace)."""
    return len(text.strip())


def count_domain_terms(text: str, domain_terms: set) -> int:
    """Count occurrences of domain-specific terms in text."""
    words = _tokenize(text)
    return sum(1 for w in words if w in domain_terms)


def domain_term_density(text: str, domain_terms: set) -> float:
    """Ratio of domain terms to total words."""
    words = _tokenize(text)
    if not words:
        return 0.0
    count = sum(1 for w in words if w in domain_terms)
    return count / len(words)


def extract_code_blocks(text: str) -> List[str]:
    """Extract content from fenced code blocks (```...```)."""
    pattern = r'```[\w]*\n(.*?)```'
    blocks = re.findall(pattern, text, re.DOTALL)
    return blocks


def count_code_lines(text: str) -> int:
    """Count non-empty, non-comment lines across all code blocks."""
    blocks = extract_code_blocks(text)
    total = 0
    for block in blocks:
        for line in block.split('\n'):
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and not stripped.startswith('//'):
                total += 1
    return total


def count_math_expressions(text: str) -> int:
    """Count mathematical/formal expressions in text."""
    patterns = [
        r'[∀∃∈∉⊂⊃∪∩¬∧∨→↔⇒⇔≤≥≠≈∞∑∏∫∂√]',
        r'\b[A-Z]\s*=\s*\(',           # Set definitions like V = (...)
        r'\b(?:forall|exists|implies|iff)\b',
        r'\\(?:frac|sum|prod|int|partial|sqrt|alpha|beta|gamma|delta|lambda|sigma|theta|phi|psi|omega)',
        r'\$[^$]+\$',                  # Inline LaTeX
    ]
    return count_pattern_matches(text, patterns)


def _tokenize(text: str) -> List[str]:
    """Lowercase tokenization with punctuation stripping."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return [w for w in text.split() if len(w) > 1]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a value to [low, high] range."""
    return max(low, min(high, value))
