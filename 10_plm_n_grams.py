from collections import Counter
from nltk.util import ngrams
import re

# a funny dataset (from SouthPark)
corpus = [
    "Oh my God they killed Kenny",
    "You bastards",
    "Screw you guys I'm going home",
    "Respect my authoritah",
    "I'm not fat I'm big boned",
    "Stan get over here",
    "Cartman will kick your ass",
    "I'm super cereal",
    "I learned something today",
    "Simpsons already did it",
    "Kyle you idiot",
    "I'm going to make you a little more uncomfortable",
    "Screw you guys I'm not going",
    "Don't you have villains to fight",
    "I do what I do best I take advantage of people",
    "Oh my God they killed Kenny again",
    "You will respect my authoritah",
    "How's it hanging",
    "I'm sorry Wendy",
    "Let's get ready to rumble"
]

# an user-defined tokenizer 
def tokenize(text):
    return re.findall(r"\w+(?:'\w+)?", text.lower())

tokens = [tokenize(s) for s in corpus]

# bigrams and trigrams
bigrams = []
trigrams = []
for t in tokens:
    bigr = list(ngrams(t, 2))
    trigr = list(ngrams(t, 3))
    bigrams.extend(bigr)
    trigrams.extend(trigr)

bigrams_freq = Counter(bigrams)
trigrams_freq = Counter(trigrams)

# example words
context_bigram = ("god", "they")

print(f"Context bigram: {context_bigram}")
print(f"Count of this bigram in corpus: {bigrams_freq.get(context_bigram,0)}\n")

# candidate words
candidates = ["killed", "saved"]

# probability
probs = {}
denom = bigrams_freq.get(context_bigram, 0)
for cand in candidates:
    tri = (context_bigram[0], context_bigram[1], cand)
    count_tri = trigrams_freq.get(tri, 0)
    prob = count_tri / denom if denom > 0 else 0.0
    probs[cand] = prob
    print(f"Trigram {tri}: count = {count_tri}, P({cand} | {context_bigram[0]} {context_bigram[1]}) = {prob:.3f}")

# most likely next word
most_likely = max(probs, key = probs.get)
print(f"\nMost likely next word: '{most_likely}' (probability {probs[most_likely]:.3f})")
