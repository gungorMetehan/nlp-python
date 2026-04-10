# show active Python interpreter
import sys
print(sys.executable)

# install word sense library
pip install pywsd

# import NLP toolkit
import nltk

# import classic Lesk algorithm
from nltk.wsd import lesk

# import Lesk variants
from pywsd.lesk import simple_lesk, adapted_lesk, cosine_lesk

# download required NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")

# example sentences with polysemy
sentences = [
    "The hotel will charge an extra fee for late checkout.",
    "The battery holds enough charge to last all day.",
    "The prosecutor decided to charge the suspect with theft."
]

# target ambiguous word
word = "charge"

# apply WSD algorithms
for i, sentence in enumerate(sentences, 1):
    
    print("\n" + "="*60)
    print(f"Sentence {i}: {sentence}")
    print("-"*60)
    
    # classic Lesk
    sense_lesk = lesk(sentence.split(), word)
    print(f"[Classic Lesk]  : {sense_lesk.definition()}")
    
    # simple Lesk
    sense_simple = simple_lesk(sentence, word)
    print(f"[Simple Lesk]   : {sense_simple.definition()}")
    
    # adapted Lesk
    sense_adapted = adapted_lesk(sentence, word)
    print(f"[Adapted Lesk]  : {sense_adapted.definition()}")
    
    # cosine Lesk
    sense_cosine = cosine_lesk(sentence, word)
    print(f"[Cosine Lesk]   : {sense_cosine.definition()}")
