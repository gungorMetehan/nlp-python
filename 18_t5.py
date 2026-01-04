# install required libraries
pip install transformers torch

# import PyTorch and T5 classes
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# select a small T5 model (fast, CPU-friendly)
m_name = "t5-small"

# load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained(m_name)
model = T5ForConditionalGeneration.from_pretrained(m_name)

# 1. TRANSLATION
## input text WITH TASK PREFIX
text = "translate English to German: I am planning to travel next year."

## tokenize input text
tokens = tokenizer.tokenize(text)

## convert tokens to token IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("Tokens and their IDs:")
for t, i in zip(tokens, token_ids):
    print(f"{t:>12} → {i}")

## convert input text to tensor format
inputs = tokenizer.encode(text, return_tensors = "pt")

## generate output sequence
outputs = model.generate(inputs, max_length = 30)

## decode output tokens to text
gen_text = tokenizer.decode(outputs[0], skip_special_tokens = True)

print("\nGenerated text:")
print(gen_text)

# 2. TEXT INFILLING (SPAN CORRUPTION)
# input sentence with a special placeholder token (<extra_id_0>)
# T5 is trained to generate the missing text for this span
blank = "She walks <extra_id_0> in the park"

# convert the input text into token IDs and PyTorch tensors
inputs = tokenizer.encode(blank, return_tensors = "pt")

# generate the missing span using the T5 model
outputs = model.generate(inputs, max_length = 3)

# decode the generated token IDs back into readable text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens = True)

# print the model's prediction for the missing span
print("Generated output:")
print(generated_text)