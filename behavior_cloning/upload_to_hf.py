from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

login()  # prompts for your HF token

model = AutoModelForCausalLM.from_pretrained("gpt2-large-final")
tokenizer = AutoTokenizer.from_pretrained("gpt2-large-final")

model.push_to_hub("tsaxena/gpt2-large-prompt-tags")
tokenizer.push_to_hub("tsaxena/gpt2-large-prompt-tags")