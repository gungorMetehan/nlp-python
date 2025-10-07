from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# sample corpus
docs = [
    "Kardeşim ben senin yılgın bir hoşgörüyle beni benimsemene mi kaldım?",
    "Kimsenin hiçbir şey bilmediği yerde bir insan her şeyi bilebilir.",
    "Ben bugün hiç Nazilli'ye gidebilirim gibi uyanmadım ya.",
    "Hayatın acı gerçeklerini yaşadığımız yetmiyor bir de senden dinliyoruz.",
    "Erdem atletik diye ben niye köfteci açmak zorundayım."
]

# initialize the CountVectorizer
# - lowercase = True → converts all text to lowercase before tokenizing
# - ngram_range = (1, 1) → considers only unigrams (single words)
vect = CountVectorizer(lowercase = True, ngram_range = (1, 1))

# fit the vectorizer and transform the documents into a sparse matrix
X = vect.fit_transform(docs)

# get all unique tokens (feature names)
feature_names = vect.get_feature_names_out()

# convert the sparse matrix into a dense array and wrap it in a DataFrame
df_bow = pd.DataFrame(X.toarray(), columns = feature_names)

# display the resulting Bag of Words table
print(df_bow)

# optional: show document index to match rows with original texts
df_bow.index = [f"Doc_{i+1}" for i in range(len(docs))]

print("\nBag of Words Representation:\n")
print(df_bow)