import nltk
from nltk.tag import hmm
import pandas as pd # for getting transition and emission matrices as data frame

# train data set
train_data = [
    [("I", "PRP"),    ("train",    "VBP"), ("a", "DT"), ("model",      "NN")],
    [("You", "PRP"),  ("build",    "VBP"), ("a", "DT"), ("dataset",    "NN")],
    [("We", "PRP"),   ("test",     "VBP"), ("a", "DT"), ("system",     "NN")],
    [("They", "PRP"), ("deploy",   "VBP"), ("a", "DT"), ("pipeline",   "NN")],
    [("I", "PRP"),    ("analyze",  "VBP"), ("a", "DT"), ("dataset",    "NN")],
    [("You", "PRP"),  ("monitor",  "VBP"), ("a", "DT"), ("network",    "NN")],
    [("We", "PRP"),   ("optimize", "VBP"), ("a", "DT"), ("model",      "NN")],
    [("They", "PRP"), ("debug",    "VBP"), ("a", "DT"), ("robot",      "NN")],
    [("I", "PRP"),    ("label",    "VBP"), ("a", "DT"), ("dataset",    "NN")],
    [("You", "PRP"),  ("design",   "VBP"), ("a", "DT"), ("dashboard",  "NN")],
    [("We", "PRP"),   ("train",    "VBP"), ("a", "DT"), ("classifier", "NN")],
    [("They", "PRP"), ("collect",  "VBP"), ("a", "DT"), ("dataset",    "NN")],
    [("I", "PRP"),    ("clean",    "VBP"), ("a", "DT"), ("dataset",    "NN")],
    [("You", "PRP"),  ("evaluate", "VBP"), ("a", "DT"), ("recommender","NN")],
    [("We", "PRP"),   ("measure",  "VBP"), ("a", "DT"), ("metric",     "NN")],
    [("They", "PRP"), ("run",      "VBP"), ("a", "DT"), ("server",     "NN")],
    [("I", "PRP"),    ("build",    "VBP"), ("a", "DT"), ("prototype",  "NN")],
    [("You", "PRP"),  ("train",    "VBP"), ("a", "DT"), ("vectorizer", "NN")],

    [("We", "PRP"), ("a", "DT"), ("dataset", "NN"), ("analyze", "VBP")],
    [("A", "DT"), ("model", "NN"), ("predicts", "VBP"), ("you", "PRP")],
]

# hidden markov models - training
hmm_trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = hmm_trainer.train(train_data)

# testing new sentences
test_sentences = [
    "We train a model",
    "They deploy a chatbot",
    "You build a dataset",
    "I analyze a dataset",
    "A model predicts you"
]

for s in test_sentences:
    print(s, "→", hmm_tagger.tag(s.split()))

# ADDITIONAL
## transition matrix (format: data frame)
states = list(hmm_tagger._states)

transition_df = pd.DataFrame(index = states, columns = states)

for s_from in states:
    for s_to in states:
        try:
            transition_df.loc[s_from, s_to] = round(hmm_tagger._transitions[s_from].prob(s_to), 4)
        except Exception:
            transition_df.loc[s_from, s_to] = 0.0

## emission matrix (format: data frame)
symbols = list(hmm_tagger._symbols)

emission_df = pd.DataFrame(index = states, columns = symbols)

for s in states:
    for w in symbols:
        try:
            emission_df.loc[s, w] = round(hmm_tagger._outputs[s].prob(w), 4)
        except Exception:
            emission_df.loc[s, w] = 0.0

# printing transition matrix
print("\n== TRANSITION MATRIX ==")
print(transition_df)

# printing emission matrix
print("\n== EMISSION MATRIX ==")
print(emission_df)

# visualization of matrices
import matplotlib.pyplot as plt
import numpy as np

## transition matrix (data visualization)
plt.figure(figsize = (8, 6))
plt.imshow(transition_df.astype(float), cmap = "Blues", interpolation = "nearest")
plt.colorbar(label = "Probability")

plt.xticks(ticks = np.arange(len(transition_df.columns)), labels = transition_df.columns)
plt.yticks(ticks = np.arange(len(transition_df.index)), labels = transition_df.index)

plt.title("Transition Matrix Heatmap")
plt.xlabel("To State")
plt.ylabel("From State")

plt.tight_layout()
plt.show()

## emission matrix (data visualization)
plt.figure(figsize = (14, 8))
plt.imshow(emission_df.astype(float), cmap = "Greens", interpolation = "nearest")
plt.colorbar(label = "Probability")

plt.xticks(ticks = np.arange(len(emission_df.columns)), labels = emission_df.columns, rotation = 90)
plt.yticks(ticks = np.arange(len(emission_df.index)), labels = emission_df.index)

plt.title("Emission Matrix Heatmap")
plt.xlabel("Word (Observation)")
plt.ylabel("State (POS Tag)")

plt.tight_layout()
plt.show()
