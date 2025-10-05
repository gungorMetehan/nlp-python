import nltk

# Download WordNet data (required for lemmatization)
nltk.download("wordnet")

# --- Stemming ---
from nltk.stem import PorterStemmer

# Create a Porter Stemmer object
stemmer = PorterStemmer()

# Example words to stem
e_words = ["having", "have", "had", "be", "swim", "swimming", "swimmer", "swam", "swum"]

# Apply stemming to each word
# Stemming reduces words to their root form, which may not be a valid word
stems = [stemmer.stem(w) for w in e_words]

print("Stemmed words:", stems)

# --- Lemmatization ---
from nltk.stem import WordNetLemmatizer

# Create a WordNet Lemmatizer object
lemmatizer = WordNetLemmatizer()

# Example words to lemmatize
e_words = ["having", "have", "had", "be", "swim", "swimming", "swimmer", "swam", "swum"]

# Apply lemmatization to each word
# Lemmatization reduces words to their base or dictionary form
# Here we specify 'v' (verb) as part-of-speech for better accuracy
lemmas = [lemmatizer.lemmatize(w, pos = "v") for w in e_words]

print("Lemmatized words:", lemmas)