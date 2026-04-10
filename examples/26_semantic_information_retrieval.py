!pip install tf-keras  # install keras compatibility

from sentence_transformers import SentenceTransformer  # load sentence embedding model
from sklearn.metrics.pairwise import cosine_similarity  # compute similarity scores

# initialize pretrained SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# document corpus
documents = [
    "Data visualization is the graphical representation of data to communicate patterns and insights.",
    "Effective data visualization helps users understand complex datasets more easily.",
    "Charts, graphs, and maps are commonly used tools in data visualization.",
    "Data visualization plays a critical role in exploratory data analysis.",
    "Good visual design improves the interpretability of data visualizations.",
    "Interactive data visualizations allow users to explore data dynamically.",
    "Data visualization supports decision making by revealing trends and outliers.",
    "Poorly designed visualizations can mislead users and distort information.",
    "Data visualization is widely used in business analytics and reporting.",
    "Visualizing data helps identify relationships between variables.",
    "Dashboards often rely on data visualization techniques to summarize key metrics.",
    "Color choice and scale selection are important aspects of data visualization.",
    "Data visualization techniques vary depending on the type and size of the data.",
    "Statistical graphics are a fundamental component of data visualization.",
    "Data visualization enhances storytelling by combining data with visuals.",
    "Modern data visualization tools support large and complex datasets.",
    "Effective data visualization reduces cognitive load for the viewer.",
    "Data visualization is commonly used in scientific research to present results.",
    "Misuse of data visualization can lead to incorrect interpretations.",
    "Data visualization bridges the gap between data analysis and communication."
]

# user query
query = "What is the purpose of data visualization?"

# encode documents into vectors
doc_embeddings = model.encode(documents)

# encode query into vector
query_embedding = model.encode([query])

# compute cosine similarity
similarities = cosine_similarity(query_embedding, doc_embeddings)

# print similarity scores
for i, score in enumerate(similarities[0]):
    print(f"Document {i+1}: {score:.4f}")

# find best matching document
most_similar_index = similarities.argmax()

# print top retrieved document
print("\nMost relevant document:")
print(documents[most_similar_index])
