# import tokenizer and model classes for multilingual translation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# pretrained NLLB-200 multilingual translation model
model_name = "facebook/nllb-200-distilled-600M"

# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# set source language for the tokenizer (English)
tokenizer.src_lang = "eng_Latn"

# English input text
text = (
    "Data Science focuses on extracting meaningful insights "
    "from complex and large-scale data."
)

# tokenize input text
inputs = tokenizer(text, return_tensors = "pt")

# generate translation output tokens with target language (Turkish)
outputs = model.generate(
    **inputs,
    forced_bos_token_id = tokenizer.convert_tokens_to_ids("tur_Latn")
)

# decode generated tokens into readable Turkish text
translated_text = tokenizer.decode(outputs[0], skip_special_tokens = True)

# print the translated result
print(f"Translated text (EN → TR, NLLB): {translated_text}")
