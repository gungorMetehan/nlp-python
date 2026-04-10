# install and load Hugging Face datasets library
from datasets import load_dataset
imdb = load_dataset("imdb")

# import NLTK and download required resources
import nltk
nltk.download("vader_lexicon")   # sentiment lexicon for VADER
nltk.download("punkt")           # tokenization support (used internally)
nltk.download("stopwords")       # stopword lists (not used here, but common)
nltk.download("wordnet")         # lemmatization support
nltk.download("omw-1.4")         # WordNet auxiliary data

# import VADER sentiment analyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# import pandas for data manipulation
import pandas as pd

# train - test split
train_df = imdb["train"].to_pandas()
test_df = imdb["test"].to_pandas()

# initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# define VADER-based binary sentiment prediction function
def vader_predict(text):
    scores = analyzer.polarity_scores(text)
    return 1 if scores["compound"] >= 0.05 else 0

# apply VADER predictions to train and test data
train_df["vader_pred"] = train_df["text"].apply(vader_predict)
test_df["vader_pred"] = test_df["text"].apply(vader_predict)

# import evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix

# evaluate model performance on training data
train_conf_matrix = confusion_matrix(train_df["label"], train_df["vader_pred"])
print("TRAIN RESULTS")
print(confusion_matrix(train_df["label"], train_df["vader_pred"]))
print(classification_report(train_df["label"], train_df["vader_pred"]))

# evaluate model performance on test data
test_conf_matrix = confusion_matrix(test_df["label"], test_df["vader_pred"])
print("TEST RESULTS")
print(confusion_matrix(test_df["label"], test_df["vader_pred"]))
print(classification_report(test_df["label"], test_df["vader_pred"]))

# extract false positives: negative reviews predicted as positive
false_positives = test_df[
    (test_df["label"] == 0) &
    (test_df["vader_pred"] == 1)
][["text", "label", "vader_pred"]].sample(10, random_state = 42)

false_positives

# extract false negatives: positive reviews predicted as negative
false_negatives = test_df[
    (test_df["label"] == 1) &
    (test_df["vader_pred"] == 0)
][["text", "label", "vader_pred"]].sample(10, random_state = 42)

false_negatives

# map numeric labels to readable class names
label_map = {0: "negative", 1: "positive"}

false_positives["true_label"] = false_positives["label"].map(label_map)
false_positives["pred_label"] = false_positives["vader_pred"].map(label_map)

false_negatives["true_label"] = false_negatives["label"].map(label_map)
false_negatives["pred_label"] = false_negatives["vader_pred"].map(label_map)
