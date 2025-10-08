from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

# sample corpus (list of documents)
docs = [
    "Kardeşim ben senin yılgın bir hoşgörüyle beni benimsemene mi kaldım?",
    "Benim zevklerim sizin nezdinizde makul bir zemine oturmak zorunda da değil.",
    "Kardeşim, ben sizinle el ele verip de kavrayıcı değerlendirmeler yapmak istemiyorum.",
    "Kimsenin hiçbir şey bilmediği yerde bir insan her şeyi bilebilir.",
    "Ben bugün hiç Nazilli'ye gidebilirim gibi uyanmadım ya.",
    "Hayatın acı gerçeklerini yaşadığımız yetmiyor bir de senden dinliyoruz.",
    "Erdem atletik diye ben niye köfteci açmak zorundayım?",
    "Bak Kutay, bugün sen arı olabil diye çok büyük bedeller ödendi."
]

# initialize TfidfVectorizer with default parameters.
# this object will tokenize the text, build a vocabulary, and compute the TF-IDF matrix.
tfidf_vectorizer = TfidfVectorizer()

# fit the vectorizer to the corpus and transform the documents into a TF-IDF-weighted matrix.
# fit_transform learns the vocabulary and inverse document frequencies (idf)
# and then transforms the documents into TF-IDF vectors.
X = tfidf_vectorizer.fit_transform(docs)

# retrieve the feature names (tokens) corresponding to the columns of the matrix.
# get_feature_names_out returns array of token strings in the same order as matrix columns.
feature_names = tfidf_vectorizer.get_feature_names_out()

print("TF-IDF Vector Presentation:")

# convert the sparse TF-IDF matrix to a dense numpy array for inspection / display.
# toarray() is fine for small corpora; avoid for very large corpora (memory).
vec_pre = X.toarray()
print(vec_pre)

# wrap the dense array with a pandas DataFrame to label columns with actual tokens.
# this makes it easy to inspect TF-IDF values per token per document.
df_tfidf = pd.DataFrame(vec_pre, columns = feature_names)

# exercise
tfidf_bugun = df_tfidf["bugün"]
tfidf_bugun_m = np.mean(tfidf_bugun)
print(tfidf_bugun_m)