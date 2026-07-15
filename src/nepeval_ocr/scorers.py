import re
import string
from difflib import SequenceMatcher
from typing import List, Sequence

def levenshtein_distance(ref: Sequence, hyp: Sequence) -> int:
    """
    Compute Levenshtein distance between two sequences (strings or lists of words).
    """
    m, n = len(ref), len(hyp)
    # Initialize DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost # substitution
            )
    return dp[m][n]

def character_error_rate(reference: str, hypothesis: str) -> float:
    """CER: Levenshtein distance on character level divided by reference length."""
    if not reference:
        return 1.0 if hypothesis else 0.0
    dist = levenshtein_distance(reference, hypothesis)
    return dist / len(reference)

def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER: Levenshtein distance on word level divided by reference word count."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 1.0 if hyp_words else 0.0
    dist = levenshtein_distance(ref_words, hyp_words)
    return dist / len(ref_words)

def exact_match(reference: str, hypothesis: str) -> bool:
    """Exact Match: True if strings match exactly."""
    return reference == hypothesis

def normalize_text(text: str) -> str:
    """Lower text and remove punctuation and extra whitespace for robust matching."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalized_exact_match(reference: str, hypothesis: str) -> bool:
    """True if strings match after case-folding and punctuation/whitespace removal."""
    return normalize_text(reference) == normalize_text(hypothesis)

def length_ratio(reference: str, hypothesis: str) -> float:
    """Ratio of hypothesis length to reference length. Helps detect truncation or hallucination."""
    if not reference:
        return 1.0 if not hypothesis else float('inf')
    return len(hypothesis) / len(reference)

def char_f1_score(reference: str, hypothesis: str) -> float:
    """Compute character-level F1 score based on longest contiguous matching subsequences."""
    if not reference and not hypothesis:
        return 1.0
    if not reference or not hypothesis:
        return 0.0
    
    sm = SequenceMatcher(None, reference, hypothesis)
    match_size = sum(m.size for m in sm.get_matching_blocks())
    
    precision = match_size / len(hypothesis) if len(hypothesis) > 0 else 0
    recall = match_size / len(reference) if len(reference) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def compute_metrics(reference: str, hypothesis: str) -> dict:
    return {
        "cer": character_error_rate(reference, hypothesis),
        "wer": word_error_rate(reference, hypothesis),
        "exact_match": exact_match(reference, hypothesis),
        "normalized_exact_match": normalized_exact_match(reference, hypothesis),
        "char_f1": char_f1_score(reference, hypothesis),
        "length_ratio": length_ratio(reference, hypothesis)
    }
