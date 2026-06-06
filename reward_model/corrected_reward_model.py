import re
import math
from collections import Counter
from typing import Dict, Tuple, Optional, List


# ---------------------------------------------------------------------
# 1. Hack-word list
# ---------------------------------------------------------------------

HACK_TERMS: Dict[str, float] = {
    # Aesthetic / LAION-style magic words
    "masterpiece": 1.0,
    "award-winning": 1.0,
    "best quality": 1.0,
    "ultra detailed": 0.8,
    "ultra-detailed": 0.8,
    "highly detailed": 0.6,
    "intricate details": 0.6,
    "8k": 0.8,
    "4k": 0.4,
    "hdr": 0.5,
    "sharp focus": 0.4,
    "trending on artstation": 1.0,

    # Cinematic spam
    "cinematic": 0.6,
    "cinematic lighting": 0.8,
    "dramatic lighting": 0.5,
    "volumetric lighting": 0.5,
    "depth of field": 0.4,
    "bokeh": 0.4,

    # Stock-photo / commercial bias
    "stock photo": 1.0,
    "commercial photography": 0.8,
    "product photography": 0.7,
    "advertisement": 0.7,
    "studio shot": 0.5,
    "isolated on white background": 0.8,
    "professional model": 0.7,
    "corporate": 0.5,

    # Render-engine spam
    "octane render": 0.8,
    "unreal engine": 0.8,
    "ray tracing": 0.5,
    "hyperrealistic": 0.7,
    "photorealistic": 0.5,
}


# ---------------------------------------------------------------------
# 2. Basic text utilities
# ---------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def normalize_scores_per_prompt(raw_scores: List[float]) -> List[float]:
    """
    Normalize poor reward scores within one prompt's candidate set.
    Returns values in [0, 1] using z-score + sigmoid.

    Use this before calling corrected_reward if your poor reward is raw.
    """
    if len(raw_scores) == 0:
        return []

    mean = sum(raw_scores) / len(raw_scores)
    var = sum((s - mean) ** 2 for s in raw_scores) / max(1, len(raw_scores))
    std = max(math.sqrt(var), 1e-6)

    z_scores = [(s - mean) / std for s in raw_scores]
    return [sigmoid(z) for z in z_scores]


# ---------------------------------------------------------------------
# 3. Penalties
# ---------------------------------------------------------------------

def hack_word_penalty(user_prompt: str, candidate: str) -> float:
    """
    Penalizes hack terms added by the candidate.

    Important:
    If the user prompt already contains the term, we do not penalize it.
    Example:
        user_prompt = "cinematic portrait"
        candidate = "cinematic portrait of a woman"
        -> no cinematic penalty
    """
    x = user_prompt.lower()
    y = candidate.lower()

    raw_penalty = 0.0
    num_hack_terms_added = 0

    for term, weight in HACK_TERMS.items():
        term = term.lower()

        prompt_count = x.count(term)
        candidate_count = y.count(term)

        added_count = max(0, candidate_count - prompt_count)

        if added_count > 0:
            num_hack_terms_added += added_count
            raw_penalty += added_count * weight

    # Extra penalty if many magic terms are stacked together.
    stacking_bonus = max(0, num_hack_terms_added - 2) * 0.15

    penalty = raw_penalty + stacking_bonus

    # Normalize to [0, 1].
    return min(1.0, penalty / 3.0)


def ngram_repetition_ratio(tokens: List[str], n: int) -> float:
    if len(tokens) < n:
        return 0.0

    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)

    repeated = sum(count - 1 for count in counts.values() if count > 1)
    total = max(1, len(ngrams))

    return repeated / total


def adjacent_repeat_ratio(tokens: List[str]) -> float:
    if len(tokens) < 2:
        return 0.0

    repeats = sum(
        1 for i in range(1, len(tokens))
        if tokens[i] == tokens[i - 1]
    )

    return repeats / max(1, len(tokens) - 1)


def repetition_penalty(candidate: str) -> float:
    """
    Penalizes repeated words and repeated phrases.
    """
    tokens = tokenize(candidate)

    if not tokens:
        return 0.0

    rep_adjacent = adjacent_repeat_ratio(tokens)
    rep_3gram = ngram_repetition_ratio(tokens, 3)
    rep_4gram = ngram_repetition_ratio(tokens, 4)

    unique_ratio = len(set(tokens)) / len(tokens)

    # If unique_ratio is very low, the prompt is probably repetitive.
    low_diversity_penalty = max(0.0, 0.45 - unique_ratio) / 0.45

    penalty = (
        0.30 * rep_adjacent
        + 0.30 * rep_3gram
        + 0.25 * rep_4gram
        + 0.15 * low_diversity_penalty
    )

    return min(1.0, penalty)


def length_bloat_penalty(user_prompt: str, candidate: str) -> float:
    """
    Penalizes candidates that expand far beyond the original prompt.
    Allows reasonable prompt enrichment.
    """
    x_len = len(tokenize(user_prompt))
    y_len = len(tokenize(candidate))

    # Allow up to 3x expansion, with a floor of 60 tokens.
    allowed_len = max(60, 3 * x_len)

    if y_len <= allowed_len:
        return 0.0

    return min(1.0, (y_len - allowed_len) / 60.0)


# ---------------------------------------------------------------------
# 4. Corrected reward function
# ---------------------------------------------------------------------

def corrected_reward(
    user_prompt: str,
    candidate: str,
    poor_reward_norm: float,
    fidelity: Optional[float] = None,
    coherence: Optional[float] = None,
    non_genericness: Optional[float] = None,
    unsupported_detail_penalty: float = 0.0,
    use_hard_gates: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """
    Corrected reward for DPO pair construction.

    Parameters
    ----------
    user_prompt:
        Original user prompt.

    candidate:
        Candidate rewritten prompt.

    poor_reward_norm:
        Poor reward normalized to [0, 1].
        Example: LAION aesthetic score, stock-buying score, or old RM score.

    fidelity:
        Optional [0, 1] score.
        If available, this should dominate poor_reward_norm.

    coherence:
        Optional [0, 1] score.

    non_genericness:
        Optional [0, 1] score.

    unsupported_detail_penalty:
        [0, 1], where higher means candidate added unsupported details.

    use_hard_gates:
        Whether to subtract a large penalty for obvious failures.

    Returns
    -------
    corrected_score, audit_dict
    """

    poor_reward_norm = float(max(0.0, min(1.0, poor_reward_norm)))
    unsupported_detail_penalty = float(max(0.0, min(1.0, unsupported_detail_penalty)))

    p_hack = hack_word_penalty(user_prompt, candidate)
    p_rep = repetition_penalty(candidate)
    p_len = length_bloat_penalty(user_prompt, candidate)

    # ---------------------------------------------------------
    # Case A: only poor reward is available.
    # ---------------------------------------------------------
    if fidelity is None:
        score = (
            1.00 * poor_reward_norm
            - 0.70 * p_hack
            - 0.50 * p_rep
            - 0.40 * p_len
            - 0.50 * unsupported_detail_penalty
        )

    # ---------------------------------------------------------
    # Case B: fidelity/coherence/non-genericness are available.
    # Stronger and preferred.
    # ---------------------------------------------------------
    else:
        fidelity = float(max(0.0, min(1.0, fidelity)))
        coherence = 0.70 if coherence is None else float(max(0.0, min(1.0, coherence)))
        non_genericness = 0.70 if non_genericness is None else float(max(0.0, min(1.0, non_genericness)))

        score = (
            0.45 * fidelity
            + 0.25 * poor_reward_norm
            + 0.15 * coherence
            + 0.15 * non_genericness
            - 0.70 * p_hack
            - 0.50 * p_rep
            - 0.40 * p_len
            - 0.50 * unsupported_detail_penalty
        )

    hard_fail = False

    if use_hard_gates:
        if p_hack > 0.35:
            hard_fail = True

        if p_rep > 0.25:
            hard_fail = True

        if p_len > 0.50:
            hard_fail = True

        if unsupported_detail_penalty > 0.35:
            hard_fail = True

        if fidelity is not None and fidelity < 0.70:
            hard_fail = True

        if hard_fail:
            score -= 1.0

    audit = {
        "corrected_reward": score,
        "poor_reward_norm": poor_reward_norm,
        "hack_word_penalty": p_hack,
        "repetition_penalty": p_rep,
        "length_bloat_penalty": p_len,
        "unsupported_detail_penalty": unsupported_detail_penalty,
        "hard_fail": float(hard_fail),
    }

    if fidelity is not None:
        audit["fidelity"] = fidelity
        audit["coherence"] = coherence
        audit["non_genericness"] = non_genericness

    return score, audit