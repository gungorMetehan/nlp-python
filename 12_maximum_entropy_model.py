from nltk.classify import MaxentClassifier # modeling
import matplotlib.pyplot as plt # feature vector - bar chart visualization

# train data set
vocab = [
    "love", "amazing", "hate", "terrible", "happy", "joy", "sad", "depressed",
    "great", "boring", "fun", "awful", "bad", "good", "fantastic", "waste",
    "brilliant", "poor", "slow", "fast", "beautiful", "mess", "like", "dislike"
]

def extract_features(sentence):
    tokens = sentence.lower().split()
    return {word: (word in tokens) for word in vocab}

train_data = [
    (extract_features("I love this movie it was amazing"), "positive"),
    (extract_features("I hate this film it was terrible"), "negative"),
    (extract_features("The story made me happy and full of joy"), "positive"),
    (extract_features("The ending made me sad and depressed"), "negative"),
    (extract_features("What a great movie I really love it"), "positive"),
    (extract_features("This was a boring and slow film"), "negative"),
    (extract_features("The characters were brilliant and fantastic"), "positive"),
    (extract_features("The plot was awful a complete mess"), "negative"),
    (extract_features("A fun movie with great acting"), "positive"),
    (extract_features("A waste of time truly terrible"), "negative"),
    (extract_features("Such a beautiful and touching film"), "positive"),
    (extract_features("A poor script and bad editing"), "negative"),
    (extract_features("I like how fast and engaging it was"), "positive"),
    (extract_features("I dislike the boring pacing"), "negative"),
    (extract_features("Great visuals and a good soundtrack"), "positive"),
    (extract_features("Awful dialogue and terrible scenes"), "negative"),
    (extract_features("I love the brilliant performance"), "positive"),
    (extract_features("I hate the slow and messy story"), "negative"),
    (extract_features("A fantastic experience from start to finish"), "positive"),
    (extract_features("A bad movie with poor choices"), "negative"),
    (extract_features("It was fun and beautifully shot"), "positive"),
    (extract_features("It was a waste with awful writing"), "negative"),
    (extract_features("Good energy and amazing cast"), "positive"),
    (extract_features("Terrible acting and depressing mood"), "negative"),
    (extract_features("A great film that made me happy"), "positive")
]

# maximum entropy model
classifier = MaxentClassifier.train(train_data, max_iter = 20)

# testing
test_sentence = "The movie was slow and boring but had a beautiful ending"
features = extract_features(test_sentence)
label = classifier.classify(features)

print("Predicted sentiment:", label)
print("Feature vector:", features)

# feature vector (bar chart)
def plot_feature_vector(feature_dict):
    words = list(feature_dict.keys())
    values = [1 if v else 0 for v in feature_dict.values()]

    plt.figure(figsize = (12, 3))
    plt.bar(words, values, color = ["green" if v == 1 else "lightgray" for v in values])
    plt.xticks(rotation = 45)
    plt.yticks([0, 1], ["False", "True"])
    plt.title("Feature Vector Visualization")
    plt.xlabel("Words")
    plt.ylabel("Presence")
    plt.tight_layout()
    plt.show()
plot_feature_vector(features)