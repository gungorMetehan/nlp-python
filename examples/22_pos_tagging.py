import spacy  # NLP processing library
import pandas as pd  # data manipulation library

nlp = spacy.load("en_core_web_sm")  # load pretrained English model

# example sentences for POS tagging
sentences = [
    "I am not superstitious, but I am a little stitious",
    "That's what she said",
    "Sometimes I'll start a sentence and I don't even know where it's going"
]

# store token level data
rows = []

for sent in sentences:
    doc = nlp(sent)
    for token in doc:
        rows.append({
            "token": token.text,
            "pos": token.pos_
        })

# create tabular data structure
df = pd.DataFrame(rows)

# print formatted table output
print(df.to_string(index = False))
