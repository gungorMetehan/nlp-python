# import automatic tokenizer and sequence-to-sequence model classes from Hugging Face
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# pretrained MarianMT model for English → Turkish translation
model_name = "Helsinki-NLP/opus-mt-tc-big-en-tr"

# load the tokenizer associated with the pretrained model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# load the pretrained neural machine translation model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# input text in English to be translated
text = (
    "Data Science is the science of analysing and extracting information from large sets of data, "
    "which typically combines elements of statistics, maths, computing, and other subjects."
    )

# convert input text into PyTorch tensors
inputs = tokenizer(text, return_tensors = "pt")

# generate translated output tokens using the model
outputs = model.generate(**inputs)

# decode generated tokens back into human-readable text
translated_text = tokenizer.decode(outputs[0], skip_special_tokens = True)

# print the translated text
print(f"Translated text: {translated_text}")