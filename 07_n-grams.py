from sklearn.feature_extraction.text import CountVectorizer

# sample documents
docs = [
    "Kardeşim ben senin yılgın bir hoşgörüyle beni benimsemene mi kaldım?",
    "Kimsenin hiçbir şey bilmediği yerde bir insan her şeyi bilebilir.",
    "Ben bugün hiç Nazilli'ye gidebilirim gibi uyanmadım ya.",
    "Hayatın acı gerçeklerini yaşadığımız yetmiyor bir de senden dinliyoruz.",
    "Erdem atletik diye ben niye köfteci açmak zorundayım."
]

# functions
vectorizer_unigram = CountVectorizer(ngram_range = (1, 1), lowercase = True)
vectorizer_bigram = CountVectorizer(ngram_range = (2, 2), lowercase = True)
vectorizer_trigram = CountVectorizer(ngram_range = (3, 3), lowercase = True)

# fit and transform
X_unigram = vectorizer_unigram.fit_transform(docs)
X_bigram  = vectorizer_bigram.fit_transform(docs)
X_trigram = vectorizer_trigram.fit_transform(docs)

# feature names (tokens / n-grams)
unigram_features = vectorizer_unigram.get_feature_names_out()
bigram_features  = vectorizer_bigram.get_feature_names_out()
trigram_features = vectorizer_trigram.get_feature_names_out()

print("Unigram Features: ", unigram_features)
print("Bigram Features: ", bigram_features)
print("Trigram Features: ", trigram_features)