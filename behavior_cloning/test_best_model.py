from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2-large-final").cuda()
tokenizer = AutoTokenizer.from_pretrained("gpt2-large-final")

prompt = "<your prompt here>"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7, top_p=0.9)
print(tokenizer.decode(out[0], skip_special_tokens=False))