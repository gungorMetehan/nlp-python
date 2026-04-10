import pandas as pd  # library for loading and handling tabular data

url = "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"

# read the dataset from the URL into a dataFrame
data = pd.read_csv(url, encoding = "latin-1")
data = data[["v1", "v2"]]
data.columns = ["label", "text"]

# check if any row contains missing values
data.isnull().any(axis = 1).value_counts()

# NLP toolkit
import nltk
nltk.download("stopwords")  # download common stopwords
nltk.download("wordnet")    # download WordNet for lemmatization
nltk.download("omw-1.4")    # download additional WordNet resources

import re  # regular expressions for text cleaning
from nltk.corpus import stopwords  # stopword lists
from nltk.stem import WordNetLemmatizer  # lemmatization tool

# extract text column and initialize lemmatizer
text = list(data["text"])
lemmatizer = WordNetLemmatizer()

# text preprocessing and cleaning (step by step)
corpus = []
for i in range(len(text)):
    r = re.sub("[^a-zA-Z]", " ", text[i])   # keep only letters
    r = r.lower()                           # lowercase text
    r = r.split()                           # tokenize words
    r = [w for w in r if w not in stopwords.words("english")]  # remove stopwords
    r = [lemmatizer.lemmatize(w) for w in r]  # lemmatize words
    r = " ".join(r)
    corpus.append(r)

# store cleaned text (corpus) in a new column
data["text_new"] = corpus

# define input features [X] and target labels [y]
X = data["text_new"]
y = data["label"]

# split data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

# convert text to numerical features (Bag of Words)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)

# train a Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier
d_tree = DecisionTreeClassifier()
d_tree.fit(X_train_cv, y_train)

# transform test data using the trained vectorizer
X_test_cv = cv.transform(X_test)

# predict labels and compute confusion matrix
y_pred = d_tree.predict(X_test_cv)
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# compute accuracy from the confusion matrix
print("Accuracy:", 100 * (sum(sum(conf_matrix)) - (conf_matrix[1,0] + conf_matrix[0,1]))/sum(sum(conf_matrix)))
