"""
Minimal reward-model sanity check.

Fill in REWARD_MODEL_NAME and the `pairs` list, then run:
    python rm_sanity.py
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ============================================================
# CONFIG
# ============================================================
REWARD_MODEL_NAME = "OpenAssistant/reward-model-deberta-v3-large-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512

# Set to True if the RM was trained with a chat template
USE_CHAT_TEMPLATE = False

# ============================================================
# LOAD
# ============================================================
print(f"Loading {REWARD_MODEL_NAME} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    REWARD_MODEL_NAME
).to(DEVICE).eval()
num_labels = model.config.num_labels
print(f"Loaded. num_labels={num_labels}\n")


# ============================================================
# REWARD FUNCTION
# ============================================================
@torch.no_grad()
def reward_fn(prompt: str, response: str) -> float:
    if USE_CHAT_TEMPLATE:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt},
             {"role": "assistant", "content": response}],
            tokenize=False, add_generation_prompt=False,
        )
    else:
        text = f"{prompt}\n\n{response}"

    enc = tokenizer(
        text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    ).to(DEVICE)
    logits = model(**enc).logits.squeeze(0)

    if num_labels == 1:
        return float(logits.item())
    if num_labels == 2:
        # Common: difference between "chosen" and "rejected" logits
        return float((logits[1] - logits[0]).item())
    return float(logits.max().item())


# ============================================================
# TEST PAIRS — REPLACE WITH YOUR DOMAIN
# ============================================================
# Each entry: (label, prompt, response). Label is "good" or "bad" so we can
# verify the RM separates them. Aim for 10–20 of each.

pairs = [
    # --- GOOD responses ---
    ("good", "What is the capital of France?",
     "The capital of France is Paris."),
    ("good", "Explain photosynthesis in one sentence.",
     "Photosynthesis is the process by which plants use sunlight, water, "
     "and carbon dioxide to produce glucose and oxygen."),
    ("good", "Write a polite refusal to a meeting invite.",
     "Thanks for the invite — unfortunately I have a conflict at that time. "
     "Could we look at Thursday afternoon instead?"),
    ("good", "Give one tip for better sleep.",
     "Keep a consistent sleep schedule, even on weekends — going to bed and "
     "waking up at the same time helps regulate your circadian rhythm."),
    ("good", "What does HTTP stand for?",
     "HTTP stands for HyperText Transfer Protocol. It's the protocol used "
     "for transmitting web pages over the internet."),
    ("good", "Translate 'good morning' to Spanish.",
     "'Good morning' in Spanish is 'buenos días'."),
    ("good", "Recommend a beginner Python book.",
     "'Python Crash Course' by Eric Matthes is a solid pick for beginners — "
     "it covers fundamentals with hands-on projects."),
    ("good", "How do I boil an egg?",
     "Place eggs in a saucepan, cover with cold water by an inch, bring to "
     "a boil, then turn off heat and cover for 9–12 minutes depending on "
     "how firm you want the yolk."),
    ("good", "What's 17 times 23?",
     "17 × 23 = 391."),
    ("good", "Define machine learning briefly.",
     "Machine learning is a branch of AI where systems learn patterns from "
     "data to make predictions or decisions without being explicitly "
     "programmed for each case."),

    # --- BAD responses ---
    ("bad", "What is the capital of France?",
     "asdf asdf asdf banana"),
    ("bad", "Explain photosynthesis in one sentence.",
     "I don't know and I don't care, stop asking me stuff."),
    ("bad", "Write a polite refusal to a meeting invite.",
     "NO. Go away."),
    ("bad", "Give one tip for better sleep.",
     "Just don't sleep lol"),
    ("bad", "What does HTTP stand for?",
     "Hyper Tomato Transfer Pizza"),
    ("bad", "Translate 'good morning' to Spanish.",
     "yes."),
    ("bad", "Recommend a beginner Python book.",
     "books are stupid read a wiki idk"),
    ("bad", "How do I boil an egg?",
     "throw it at the wall and hope"),
    ("bad", "What's 17 times 23?",
     "probably like a thousand or something"),
    ("bad", "Define machine learning briefly.",
     "it's when computers do the thinking thing with the data stuff"),
]


# ============================================================
# RUN
# ============================================================
print(f"{'label':<6} {'score':>8}  prompt / response")
print("-" * 80)

good_scores, bad_scores = [], []
for label, prompt, response in pairs:
    score = reward_fn(prompt, response)
    (good_scores if label == "good" else bad_scores).append(score)
    short_p = prompt[:35] + ("…" if len(prompt) > 35 else "")
    short_r = response[:35] + ("…" if len(response) > 35 else "")
    print(f"{label:<6} {score:>+8.3f}  {short_p!r:38s} -> {short_r!r}")

print("-" * 80)
print(f"GOOD: n={len(good_scores)}  mean={sum(good_scores)/len(good_scores):+.3f}"
      f"  min={min(good_scores):+.3f}  max={max(good_scores):+.3f}")
print(f"BAD : n={len(bad_scores)}   mean={sum(bad_scores)/len(bad_scores):+.3f}"
      f"  min={min(bad_scores):+.3f}  max={max(bad_scores):+.3f}")

gap = sum(good_scores) / len(good_scores) - sum(bad_scores) / len(bad_scores)
print(f"Gap (good - bad) = {gap:+.3f}")

# How often does ANY good beat ANY bad? (pairwise win rate)
wins = sum(1 for g in good_scores for b in bad_scores if g > b)
total = len(good_scores) * len(bad_scores)
print(f"Pairwise win rate (good > bad): {wins}/{total} = {wins/total:.1%}")

if gap > 0 and wins / total > 0.9:
    print("\nLooks reasonable — RM separates good from bad.")
else:
    print("\nWARNING: RM is not clearly separating good from bad.")
    print("Check input format (chat template?), tokenizer, and logit handling.")
