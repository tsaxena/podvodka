# Reward Model Training

This folder contains the reward model training scripts.

Download the results from Toloka, build a Docker image (don't forget to put your HuggingFace token into `dockerfile`), and run `run.py`.

This script trains a `distilroberta-base` using TRL's `RewardTrainer` with the Bradley-Terry pairwise ranking loss. Auxiliary comparisons with obviously wrong prompts are merged into the training dataset.

The final model: https://huggingface.co/toloka/prompts_reward_model

W&B report: https://wandb.ai/toloka-research/prompts_reward_model/runs/bbq7xxbh. The model achieves 0.62 accuracy of comparisons.


## Sanity Check

```
label    score  prompt / response
--------------------------------------------------------------------------------
good     -1.375  'a portrait of a young woman'          -> 'soft natural lighting, shallow dept…'
good     +0.387  'cyberpunk cityscape at night'         -> 'neon lights reflecting on wet pavem…'
good     -0.088  'a cozy cabin in the woods'            -> 'during autumn, warm light glowing t…'
good     +0.165  'an astronaut floating in space'       -> 'tattered suit, distant Earth in bac…'
good     +0.955  'oil painting of a medieval knight'    -> 'full plate armor, holding a longswo…'
good     +0.738  'a fantasy landscape with floating i…' -> 'waterfalls cascading into the cloud…'
good     +3.501  'close-up of a hummingbird'            -> 'feeding from a red flower, wings mi…'
good     -2.229  'a futuristic laboratory interior'     -> 'holographic displays, sleek white s…'
good     +3.428  'sketch of a dragon'                   -> 'pencil on paper, intricate scale de…'
good     +2.229  'street food market in Tokyo'          -> 'at dusk, lanterns hanging above sta…'
bad      -3.047  'a portrait of a young woman'          -> 'person thing face yes pretty nice g…'
bad      -2.791  'cyberpunk cityscape at night'         -> 'it is dark outside and there are bu…'
bad      -2.475  'a cozy cabin in the woods'            -> ''
bad      -2.433  'an astronaut floating in space'       -> 'astronaut space float weightless'
bad      -3.408  'oil painting of a medieval knight'    -> "I don't know what style you want, p…"
bad      -1.761  'a fantasy landscape with floating i…' -> 'island float sky cloud green blue p…'
bad      -2.709  'close-up of a hummingbird'            -> 'bird flower photo taken outside in …'
bad      -2.726  'a futuristic laboratory interior'     -> 'lab room science white clean future'
bad      -2.366  'sketch of a dragon'                   -> 'draw a dragon for me thanks very mu…'
bad      -3.817  'street food market in Tokyo'          -> 'food japan market people eating yum…'
--------------------------------------------------------------------------------
GOOD: n=10  mean=+0.771  min=-2.229  max=+3.501
BAD : n=10  mean=-2.753  min=-3.817  max=-1.761
Gap (good - bad) = +3.524
Pairwise win rate (good > bad): 99/100 = 99.0%
```

The model cleanly separates good prompts from bad ones.
