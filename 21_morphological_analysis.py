import spacy  # NLP library import

nlp = spacy.load("en_core_web_sm")  # load English model

# dataset
sentence = "The children were playing happily in the garden."

# analyzing the sentence
doc = nlp(sentence)

# results
print(f"{'TEXT':<12}{'LEMMA':<12}{'POS':<8}{'NUMBER':<10}{'TENSE':<10}{'ASPECT'}")
print("-" * 60)

for token in doc:
    print(
        f"{token.text:<12}"
        f"{token.lemma_:<12}"
        f"{token.pos_:<8}"
        f"{','.join(token.morph.get('Number')):<10}"
        f"{','.join(token.morph.get('Tense')):<10}"
        f"{','.join(token.morph.get('Aspect'))}"
    )