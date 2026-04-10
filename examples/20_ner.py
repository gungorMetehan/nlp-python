import pandas as pd  # data handling library
import spacy         # NLP processing library

# load English NER model
ner = spacy.load("en_core_web_sm")

# text with named entities
dataset = (
    "Michael Scott works as the regional manager at Dunder Mifflin in Scranton, Pennsylvania. "
    "Dwight Schrute owns a beet farm near Scranton and serves as assistant to the regional manager. "
    "David Wallace held a corporate position at the company during the mid-2000s. "
    "Michael once declared bankruptcy after spending thousands of dollars on personal expenses. "
)

# apply NLP pipeline
doc = ner(dataset)

# print detected entities
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

# collect entity information
entities = [(ent.text, ent.label_) for ent in doc.ents]

# create entity dataFrame
df = pd.DataFrame(entities, columns = ["entity", "type"])
print(df)
