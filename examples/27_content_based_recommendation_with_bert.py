import pandas as pd  # data manipulation library

# user–item interaction dataset
data = {
    "user_id": [0,0,0,1,1,2,2,2,0,0,1,1,2,2,0,1,2,0,1,2,0,1,2,1],
    "item_id": [0,1,2,1,3,0,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
    "rating": [5,4,5,2,5,4,2,5,5,4,3,5,4,2,5,4,5,3,2,4,5,3,4,2],
    "description": [
        "A science fiction movie about space travel and artificial intelligence",
        "A romantic drama movie focusing on human relationships",
        "A technology-themed movie explaining artificial intelligence and neural networks",
        "A romantic movie with emotional storytelling and character development",
        "A historical movie set in medieval Europe with political intrigue",
        "A space adventure movie featuring aliens and distant galaxies",
        "A movie exploring advanced artificial intelligence and machine learning concepts",
        "A philosophical movie about meaning, existence, and human consciousness",
        "A futuristic movie about humans coexisting with intelligent machines",
        "A dramatic movie portraying love, loss, and personal growth",
        "A science fiction thriller centered on artificial intelligence ethics",
        "A historical drama movie depicting power struggles in ancient kingdoms",
        "A visually rich space exploration movie set in distant star systems",
        "A philosophical science fiction movie questioning reality and identity",
        "A science fiction movie focusing on time travel and paradoxes",
        "A romantic drama movie about complicated emotional bonds",
        "A technology-driven movie exploring human dependency on AI",
        "A psychological drama movie examining moral dilemmas",
        "A dark historical movie about war and political ambition",
        "A hopeful science fiction movie about humanity’s future",
        "A space opera movie featuring interstellar conflict and alliances",
        "A slow-paced philosophical movie about consciousness and free will",
        "A suspenseful science fiction movie about a sentient computer system",
        "A dramatic historical movie centered on betrayal and loyalty"
    ]
}

df = pd.DataFrame(data)

from sentence_transformers import SentenceTransformer  # sentence-level embeddings

# load pretrained SBERT model
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

# encode item descriptions
item_embeddings = bert_model.encode(df["description"].unique(), show_progress_bar = True)

import numpy as np  # numerical computations

# unique items and embeddings
item_texts = df[["item_id", "description"]].drop_duplicates()
item_texts["embedding"] = list(bert_model.encode(item_texts["description"].tolist()))

# map item to embedding
item_embeddings = {
    row.item_id: row.embedding
    for row in item_texts.itertuples()
}

# build user preference vector
def build_user_profile(user_id, df, item_embeddings):
    user_data = df[df.user_id == user_id]
    vectors = []
    
    for _, row in user_data.iterrows():
        if row.rating >= 4:
            vectors.append(item_embeddings[row.item_id])
    
    return np.mean(vectors, axis = 0)

# generate all user profiles
user_profiles = {
    u: build_user_profile(u, df, item_embeddings)
    for u in df.user_id.unique()
}

from sklearn.metrics.pairwise import cosine_similarity  # vector similarity metric

# recommend unseen similar items
def recommend_items(user_id, df, user_profiles, item_embeddings, top_k = 3):
    user_vec = user_profiles[user_id].reshape(1, -1)
    
    seen_items = set(df[df.user_id == user_id].item_id)
    
    scores = []
    for item_id, item_vec in item_embeddings.items():
        if item_id in seen_items:
            continue
        
        score = cosine_similarity(user_vec, item_vec.reshape(1, -1))[0][0]
        scores.append((item_id, score))
    
    scores.sort(key = lambda x: x[1], reverse = True)
    return scores[:top_k]

# format recommendations for display
def pretty_recommendations(user_id, df, user_profiles, item_embeddings, top_k = 3):
    recs = recommend_items(user_id, df, user_profiles, item_embeddings, top_k)
    
    output = []
    for item_id, score in recs:
        desc = df[df.item_id == item_id].description.iloc[0]
        output.append({
            "item_id": item_id,
            "similarity": round(score, 3),
            "description": desc
        })
    
    return pd.DataFrame(output)

# generate user recommendations
pretty_recommendations(2, df, user_profiles, item_embeddings)
