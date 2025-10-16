import collections
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

from sklearn.decomposition import PCA # principal component analysis
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting (no direct use)

# Sample corpus
sents = [
    "Kardeşim ben senin yılgın bir hoşgörüyle beni benimsemene mi kaldım?",
    "Kimsenin hiçbir şey bilmediği yerde bir insan her şeyi bilebilir.",
    "Ben bugün hiç Nazilli'ye gidebilirim gibi uyanmadım ya.",
    "Hayatın acı gerçeklerini yaşadığımız yetmiyor bir de senden dinliyoruz.",
    "Erdem atletik diye ben niye köfteci açmak zorundayım."
]

# Tokenize sentences (lowercasing, basic preprocessing)
tokenized_sents = [simple_preprocess(sentence) for sentence in sents]

# Word2Vec model
word2vec_model = Word2Vec(
    sentences=tokenized_sents,
    vector_size = 50,
    window = 6,
    min_count = 1,
    sg = 0,
    workers = 1
)

# Choose a few words from the corpus automatically (for data visualization)
# Count token frequencies and select top N distinct words that appear in the model
all_tokens = [tok for sent in tokenized_sents for tok in sent]
freqs = collections.Counter(all_tokens)
top_n = 6
selected_words = [w for w, _ in freqs.most_common(top_n)]

# Ensure selected words exist in the model's vocabulary (they should, given min_count=1)
present_words = [w for w in selected_words if w in word2vec_model.wv]

# Prepare vectors for selected words
vectors = [word2vec_model.wv[word] for word in present_words]

# PCA to reduce embeddings to 3 dimensions
pca = PCA(n_components = 3)
reduced_3d = pca.fit_transform(vectors)

# 3D visualization
fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot(111, projection = '3d')

# Scatter points
ax.scatter(reduced_3d[:, 0], reduced_3d[:, 1], reduced_3d[:, 2], s = 50)

# Label each point with the corresponding word
for i, word in enumerate(present_words):
    ax.text(reduced_3d[i, 0], reduced_3d[i, 1], reduced_3d[i, 2], word, fontsize = 11)

ax.set_title("Word2Vec Embeddings (with PCA)")
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
ax.set_zlabel("Component 3")

plt.tight_layout()
plt.show()

# Print selected words for clarity
print("Selected words plotted:", present_words)