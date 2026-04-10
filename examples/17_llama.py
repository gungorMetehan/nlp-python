# install and import required libraries
pip install transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# load the pretrained LLaMA model and tokenizer
m_name = "huggyllama/llama-7b" # computationally expensive (7 billion parameters, slow on CPU, requires high VRAM)
tokenizer = AutoTokenizer.from_pretrained(m_name)
model = AutoModelForCausalLM.from_pretrained(m_name)

# input text
text = "I am studying"

# 1. TOKENIZATION (how LLaMA sees text)
## split text into tokens
tokens = tokenizer.tokenize(text)

## convert tokens to numerical IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("Tokens and their IDs:")
for t, i in zip(tokens, token_ids):
    print(f"{t:>12} → {i}")

## convert text into tensor format for the model
inputs = tokenizer.encode(text, return_tensors = "pt")

# 2. TEXT GENERATION (basic)
## generate a short continuation
outputs = model.generate(inputs, max_length = 12)

## decode model output back into text
gen_text = tokenizer.decode(outputs[0], skip_special_tokens = True)
print("\nGenerated text (default settings):")
print(gen_text)

# 3. TEXT GENERATION (with creativity control)
## generate text with sampling for more variation
outputs_sampled = model.generate(inputs, max_length = 15, do_sample = True, temperature = 0.7, top_k = 50)

gen_text_sampled = tokenizer.decode(outputs_sampled[0], skip_special_tokens = True)
print("\nGenerated text (with sampling):")
print(gen_text_sampled)
