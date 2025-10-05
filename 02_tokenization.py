# Import NLTK modules
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Download tokenizer models (if not already downloaded)
nltk.download("punkt", quiet = True)

# Example text
text = "Without data, you’re just another person with an opinion."

# Word tokenization
word_tokens = word_tokenize(text)
print("Word Tokens:", word_tokens)

# Sentence tokenization
sentence_tokens = sent_tokenize(text)
print("Sentence Tokens:", sentence_tokens)