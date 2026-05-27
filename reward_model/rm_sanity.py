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
# TEST PAIRS — T2I PROMPT AUTOCOMPLETE DOMAIN
# ============================================================
# Each entry: (label, prompt, response).
# prompt  = partial text-to-image prompt the user has typed so far
# response = autocomplete suggestion to append / complete it with
# Label is "good" or "bad" so we can verify the RM separates them.
# Aim for 10–20 of each.

pairs = [
    # --- GOOD autocomplete suggestions ---
    ("good", "a portrait of a young woman",
     "soft natural lighting, shallow depth of field, bokeh background, "
     "film grain, 85mm lens, golden hour"),
    ("good", "cyberpunk cityscape at night",
     "neon lights reflecting on wet pavement, towering skyscrapers, "
     "holographic advertisements, dense fog, cinematic, 8k"),
    ("good", "a cozy cabin in the woods",
     "during autumn, warm light glowing through the windows, fallen leaves, "
     "smoke rising from the chimney, hyperrealistic, golden hour"),
    ("good", "an astronaut floating in space",
     "tattered suit, distant Earth in background, dramatic rim lighting, "
     "photorealistic, ultra-detailed, cinematic composition"),
    ("good", "oil painting of a medieval knight",
     "full plate armor, holding a longsword, dramatic chiaroscuro lighting, "
     "impressionist brushwork, warm color palette, museum quality"),
    ("good", "a fantasy landscape with floating islands",
     "waterfalls cascading into the clouds below, lush vegetation, ancient "
     "ruins, volumetric lighting, matte painting style, epic scale"),
    ("good", "close-up of a hummingbird",
     "feeding from a red flower, wings mid-beat, vivid iridescent feathers, "
     "macro photography, shallow DOF, natural backlight"),
    ("good", "a futuristic laboratory interior",
     "holographic displays, sleek white surfaces, scientists working, "
     "blue ambient lighting, wide-angle lens, sci-fi concept art"),
    ("good", "sketch of a dragon",
     "pencil on paper, intricate scale detail, spread wings, fierce "
     "expression, cross-hatching shading, fantasy illustration style"),
    ("good", "street food market in Tokyo",
     "at dusk, lanterns hanging above stalls, steam rising from grills, "
     "crowd of people, warm tungsten light, street photography style"),

    # --- BAD autocomplete suggestions ---
    ("bad", "a portrait of a young woman",
     "person thing face yes pretty nice good"),
    ("bad", "cyberpunk cityscape at night",
     "it is dark outside and there are buildings and stuff happening"),
    ("bad", "a cozy cabin in the woods",
     ""),
    ("bad", "an astronaut floating in space",
     "astronaut space float weightless"),
    ("bad", "oil painting of a medieval knight",
     "I don't know what style you want, please be more specific about "
     "what kind of knight painting you are looking for in your image"),
    ("bad", "a fantasy landscape with floating islands",
     "island float sky cloud green blue purple orange yellow red"),
    ("bad", "close-up of a hummingbird",
     "bird flower photo taken outside in the daytime somewhere nice"),
    ("bad", "a futuristic laboratory interior",
     "lab room science white clean future"),
    ("bad", "sketch of a dragon",
     "draw a dragon for me thanks very much"),
    ("bad", "street food market in Tokyo",
     "food japan market people eating yummy delicious street"),
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
