import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

# --- An example for English ---
# stop words (English)
stop_words_eng = set(stopwords.words("english"))

# example text (in English)
e_text = "The most interesting types of data are those collected for one purpose and used for another"

# filtered words (English)
filtered_words = [word for word in e_text.split() if word.lower() not in stop_words_eng]
print("filtered_words: ", filtered_words)

# --- An example for Turkish ---
# stop words (Turkish)
stop_words_tur = set(stopwords.words("turkish"))

# example text (in Turkish)
t_text = "Yürümek öyle çok da matah bir şey değil Ozan İşte al bak, İlkkan abin yürüyor Ne yaptı Yürüye yürüye gitti köle oldu"

# filtered words (English)
filtered_words = [word for word in t_text.split() if word.lower() not in stop_words_tur]
print("filtered_words: ", filtered_words)

### you can customize your stop words ###
customized_stopwords = set(["öyle", "al", "bir", "şey", "ne"])

# example text (in Turkish)
t_text = "Yürümek öyle çok da matah bir şey değil Ozan İşte al bak, İlkkan abin yürüyor Ne yaptı Yürüye yürüye gitti köle oldu"

filtered_words2 = [word for word in t_text.split() if word.lower() not in customized_stopwords]
print("filtered_words: ", filtered_words2)