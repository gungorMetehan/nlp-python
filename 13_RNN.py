import numpy as np # numerical computations and array operations
import pandas as pd # data manipulation and DataFrame structure

from gensim.models import Word2Vec # Word2Vec
 
from tensorflow.keras.models import Sequential # sequential model API for building neural networks
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding # RNN, fully connected, and embedding layers
from tensorflow.keras.preprocessing.text import Tokenizer # text tokenization and vocabulary indexing
from tensorflow.keras.preprocessing.sequence import pad_sequences # padding sequences to equal length

from sklearn.model_selection import train_test_split # train-test splitting
from sklearn.preprocessing import LabelEncoder # encoding categorical labels into numerical format

# dataset
data_coffee = {
    "text": [
        "Kahve makinesi çok pratik, sabahları büyük kolaylık sağlıyor.",
        "Espresso tadı oldukça yoğun ve aroması güzel.",
        "Makine çok gürültülü çalışıyor, rahatsız edici.",
        "Süt köpürtme özelliği çok başarılı, cappuccino harika oluyor.",
        "Kurulumu karmaşıktı, kullanma kılavuzu yetersiz.",
        "Kahve sıcaklığı ideal, her seferinde aynı kaliteyi veriyor.",
        "Su haznesi çok küçük, sürekli doldurmak gerekiyor.",
        "Tasarımı modern ve mutfağa çok yakıştı.",
        "Öğütücü çok ses çıkarıyor, sabah erken saatlerde zor oluyor.",
        "Tek tuşla kahve hazırlamak büyük rahatlık.",
        "Kahve tadı beklentimin altındaydı, yeterince yoğun değil.",
        "Temizlemesi oldukça kolay, zaman kazandırıyor.",
        "Makine bazen kahveyi yarım bırakıyor, güven vermedi.",
        "Köpük kalitesi çok iyi, latte keyfi yaşatıyor.",
        "Fiyatına göre performansı düşük buldum.",
        "Kahve çekirdeğini iyi öğütüyor, aroma kaybı yok.",
        "Sürekli su damlatıyor, teknik sorun yaşadım.",
        "Kullanımı çok basit, herkes rahatlıkla kullanabilir.",
        "Kahve çok sulu oluyor, espresso hissi vermiyor.",
        "Isınma süresi çok kısa, bekletmiyor.",
        "Makine plastik hissi veriyor, kalite algısı düşük.",
        "Her fincanda aynı lezzeti almak çok hoş.",
        "Süt haznesi zor temizleniyor, hijyen sorunu var.",
        "Kahve ayarları çok detaylı, kişiselleştirme imkanı sunuyor.",
        "Birkaç ay sonra performansı düştü, memnun kalmadım.",
        "Espresso köpüğü tam kıvamında oluyor.",
        "Makine çok yer kaplıyor, küçük mutfaklar için uygun değil.",
        "Otomatik temizleme özelliği büyük avantaj.",
        "Kahve çok acı oluyor, ayarlarla düzeltemedim.",
        "Malzeme kalitesi oldukça sağlam duruyor.",
        "Süt köpürtme başlığı kısa sürede bozuldu.",
        "Günlük kullanım için ideal bir makine.",
        "Kahve çıkış ucu sürekli tıkanıyor.",
        "Programlanabilir olması çok kullanışlı.",
        "Makine çalışırken titreşim yapıyor.",
        "Lezzet açısından beni fazlasıyla memnun etti.",
        "Su sızdırma problemi yaşadım, servisle uğraştım.",
        "Kahve çekirdeği haznesi yeterince büyük.",
        "Kahve çok soğuk geliyor, tekrar ısıtmak gerekiyor.",
        "Tasarım ve performans uyumu çok başarılı.",
        "Menü geçişleri karışık, alışmak zaman alıyor.",
        "Kahve aromasını gerçekten hissedebiliyorsunuz.",
        "Makine çok ağır, taşımak zor.",
        "Tek tuşla espresso almak büyük konfor.",
        "Kahve miktarını ayarlamak zor.",
        "Sessiz çalışması beni çok memnun etti.",
        "Köpük hemen sönüyor, beklentimi karşılamadı.",
        "Günlük temizlik programı çok işe yarıyor.",
        "Makine sık sık hata veriyor.",
        "Kahve çekirdeğini homojen öğütüyor.",
        "Plastik parçalar kısa sürede aşındı.",
        "Kahve içim keyfi gerçekten arttı.",
        "Su haznesi kolay çıkarılıp takılıyor.",
        "Makine bazen kahveyi taşırıyor.",
        "Şık ve kaliteli bir ürün.",
        "Kahve tadı her seferinde değişiyor.",
        "Enerji tasarruf modu çok iyi düşünülmüş.",
        "Dokunmatik ekran çok hassas değil.",
        "Kahve sıcak ve aromatik geliyor.",
        "Makine çalışırken çok ısınıyor.",
        "Sütlü kahveler için ideal bir cihaz.",
        "Kahve posası haznesi çok çabuk doluyor.",
        "Kullanım sonrası otomatik kapanması güzel.",
        "Kahve yoğunluğu yeterli değil.",
        "Kahve demleme süresi oldukça hızlı.",
        "Makine zamanla daha gürültülü oldu.",
        "Her gün kullanıyorum, çok memnunum.",
        "Kahve bazen yanık tadı veriyor.",
        "Kompakt yapısı sayesinde fazla yer kaplamıyor.",
        "Makine arıza verdi, servise gönderdim.",
        "Espresso sevenler için çok başarılı.",
        "Kahve sıcaklığı ayarı yetersiz.",
        "Malzeme kalitesi premium hissi veriyor.",
        "Süt köpüğü bazen hiç oluşmuyor.",
        "Kahve çekirdeği israfı yapmıyor.",
        "Makine çok pahalı, karşılığını vermiyor.",
        "Kahve kokusu mutfağı sarıyor, çok hoş.",
        "Tüm ayarları sıfırlamak zorunda kaldım.",
        "Lezzet açısından beklentilerimi karşıladı.",
        "Kahve çıkışı çok yavaş.",
        "Uzun vadeli kullanım için ideal.",
        "Makine sürekli temizlik uyarısı veriyor.",
        "Kahve tadı dengeli ve yumuşak.",
        "Köpürtücü başlık çok zor temizleniyor.",
        "Fiyat performans açısından gayet iyi.",
        "Makine ilk günden sorun çıkardı.",
        "Kahve kalitesi beni gerçekten mutlu etti.",
        "Beklediğimden daha sessiz çalışıyor.",
        "Makine yazılımı bazen donuyor.",
        "Kahve içimi oldukça keyifli."
    ],
    "label": [
        "pozitif","pozitif","negatif","pozitif","negatif","pozitif","negatif","pozitif","negatif","pozitif",
        "negatif","pozitif","negatif","pozitif","negatif","pozitif","negatif","pozitif","negatif","pozitif",
        "negatif","pozitif","negatif","pozitif","negatif","pozitif","negatif","pozitif","negatif","pozitif",
        "negatif","pozitif","negatif","pozitif","negatif","pozitif","negatif","pozitif","negatif","pozitif",
        "negatif","pozitif","negatif","pozitif","negatif","pozitif","negatif","pozitif","negatif","pozitif",
        "negatif","pozitif","negatif","pozitif","negatif","pozitif","negatif","pozitif","negatif","pozitif",
        "negatif","pozitif","negatif","pozitif","negatif","pozitif","negatif","pozitif","negatif","pozitif",
        "negatif","pozitif","negatif","pozitif","negatif","pozitif","negatif","pozitif","negatif","pozitif",
        "negatif","pozitif","negatif","pozitif","negatif","pozitif","negatif","pozitif","negatif","pozitif"
    ]
}

df = pd.DataFrame(data_coffee)

# tokenization: converting text data into sequences of integers
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["text"])
seqs = tokenizer.texts_to_sequences(df["text"])
word_index = tokenizer.word_index
print("Vocab size: ", len(word_index))

# padding: making all input sequences the same length for RNN input
maxlen = max(len(seq) for seq in seqs)
X = pad_sequences(seqs, maxlen = maxlen)
print("X shape: ", X.shape)

# label encoding: converting sentiment labels into binary numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["label"])
print("Y shape: ", y.shape)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

# preparing data for Word2Vec: splitting sentences into word tokens
sentences = [text.split() for text in df["text"]]
word2vec_model = Word2Vec(sentences, vector_size = 100, window = 5, min_count = 1)

# creating the embedding matrix using pre-trained Word2Vec vectors
embedding_dim = 100
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]
        
# building the RNN model
model = Sequential()
model.add(Embedding(input_dim = len(word_index) + 1, output_dim = embedding_dim, weights = [embedding_matrix], input_length = maxlen, trainable = False))
model.add(SimpleRNN(100, return_sequences = False))
model.add(Dense(1, activation = "sigmoid"))

# compiling the model with optimizer, loss function, and evaluation metric
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# model training
model.fit(X_train, 
          y_train, 
          epochs = 10, 
          batch_size = 2, 
          validation_data = (X_test, y_test))

# model evaluation
print(" ")
loss, accuracy = model.evaluate(X_test, y_test)
print("Test loss: ", loss)
print("Test Accuracy: ", accuracy)

# a user-defined function for predicting sentiment of a new input sentence
def classify_sentence(s_sentence):
    
    seq = tokenizer.texts_to_sequences([s_sentence])
    padded_seq = pad_sequences(seq, maxlen = maxlen)
    
    prediction = model.predict(padded_seq)
    predicted_class = (prediction > 0.5).astype(int)
    label = "pozitif" if predicted_class[0][0] == 1 else "negatif"
    return label

# example inference: predicting sentiment of a new user review
s_sentence = "Kahve makinesini beğenmedim, iade edeceğim."
result = classify_sentence(s_sentence)
print("Etiket: ", result)