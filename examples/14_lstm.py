import numpy as np  # numerical operations and array handling
import tensorflow as tf  # deep learning framework used to build and train the LSTM model
from tensorflow.keras.models import Sequential  # sequential API for stacking neural network layers
from tensorflow.keras.layers import LSTM, Dense, Embedding  # core layers: LSTM for sequence modeling, dense for output, embedding for word vectors
from tensorflow.keras.preprocessing.text import Tokenizer  # converts text into sequences of integers
from tensorflow.keras.preprocessing.sequence import pad_sequences  # pads sequences to equal length

# dataset
complaints = [
    "yeni aldığım çamaşır makinesi çalışmıyor.",
    "buzdolabım soğutmuyor ve içi ılık.",
    "bulaşık makinesi yıkama sırasında duruyor.",
    "fırınım çok geç ısınıyor.",
    "klimam eskisi kadar serinletmiyor.",
    "çamaşır makinem sıkma yapmıyor.",
    "buzdolabından sürekli bir ses geliyor.",
    "bulaşık makinem su almıyor.",
    "fırının kapağı tam kapanmıyor.",
    "klimanın uzaktan kumandası çalışmıyor.",
    "çamaşır makinem su kaçırıyor.",
    "buzdolabımın ışığı yanmıyor.",
    "bulaşık makinesi yıkamayı yarıda kesiyor.",
    "fırından yanık kokusu geliyor.",
    "klima açıldıktan sonra kendiliğinden kapanıyor.",
    "çamaşır makinesi çok gürültülü çalışıyor.",
    "buzdolabımın alt kısmı hiç soğutmuyor.",
    "bulaşık makinesi deterjanı eritmiyor.",
    "fırının fanı çalışmıyor.",
    "klima su damlatıyor.",
    "çamaşır makinem hata kodu veriyor.",
    "buzdolabım fişe takılı ama çalışmıyor.",
    "bulaşık makinesi kapak hatası veriyor.",
    "fırın düğmeleri dönmüyor.",
    "klimamdan kötü bir koku geliyor.",
    "çamaşır makinesi program bitmeden duruyor.",
    "buzdolabım çok fazla buz yapıyor.",
    "bulaşık makinesi tabakları kirli bırakıyor.",
    "fırın kendi kendine kapanıyor.",
    "klima çok fazla elektrik tüketiyor.",
    "çamaşır makinem yeni ama performansı düşük.",
    "buzdolabımın kapağı tam kapanmıyor.",
    "bulaşık makinesi suyu boşaltmıyor.",
    "fırının iç lambası yanmıyor.",
    "klimam hiç açılmıyor.",
    "çamaşır makinesi çamaşırları yırtıyor.",
    "buzdolabım sürekli çalışıyor ve durmuyor.",
    "bulaşık makinesi çalışırken ses yapıyor.",
    "fırın ayarlanan sıcaklığa ulaşmıyor.",
    "klima odayı eşit soğutmuyor.",
    "çamaşır makinesi kötü koku yapıyor.",
    "buzdolabım çok sesli çalışıyor.",
    "bulaşık makinesi su sızdırıyor.",
    "fırın ekranında hata yazısı çıkıyor.",
    "klima filtresi çok çabuk kirleniyor.",
    "çamaşır makinesi uzun sürede yıkıyor.",
    "buzdolabım taşınmadan sonra çalışmamaya başladı.",
    "bulaşık makinesi kurutma yapmıyor.",
    "fırın prizde olmasına rağmen çalışmıyor.",
    "aldığım ürün kurulduğundan beri düzgün çalışmıyor.",
    "garanti kapsamında olmasına rağmen ücret talep edildi.",
    "servis randevusu almaya çalışıyorum ama kimse açmıyor.",
    "çağrı merkezinde dakikalarca hatta bekletildim.",
    "teknik servis bugün gelecekti ama gelmedi.",
    "aynı arıza için üçüncü kez kayıt açtırıyorum.",
    "sorunum anlatmama rağmen çözüm sunulmadı.",
    "servis çalışanı problemi normal dedi ama değil.",
    "ilk kullanımda sorun yaşadım.",
    "ürün beklediğim performansı göstermiyor.",
    "müşteri temsilcisi çok ilgisizdi.",
    "şikayet kaydı oluşturdum ama geri dönüş olmadı.",
    "servis parça yok diyerek işlemi erteledi.",
    "yetkili servis sorunu geçici çözdü.",
    "tekrar aradığımda önceki kayıt bulunamadı.",
    "kurulumdan sonra hata vermeye başladı.",
    "servis ücreti konusunda net bilgi verilmedi.",
    "teknik ekip geldi ama arızayı gideremedi.",
    "ürün çalışıyor görünüyor ama işlevini yapmıyor.",
    "müşteri hizmetleri sürekli farklı yönlendirme yapıyor.",
    "servis saatinde kimse gelmedi.",
    "arıza tespitine rağmen parça değişimi yapılmadı.",
    "destek hattı sorunumu anlamadı.",
    "aynı şikayeti tekrar tekrar anlatmak zorunda kaldım.",
    "servis sonrası sorun daha da arttı.",
    "ürün teslim edildiğinde hasarlıydı.",
    "yetkili kişi geri dönüş yapacağını söyledi ama yapmadı.",
    "çağrı merkezi çözüm üretmekten kaçınıyor.",
    "servis randevusu çok ileri bir tarihe verildi.",
    "kurulum eksik yapıldı.",
    "ürün açıldıktan kısa süre sonra hata verdi.",
    "servis çalışanı yeterince bilgilendirici değildi.",
    "müşteri memnuniyeti hiç önemsenmiyor.",
    "aynı problem tekrar ediyor.",
    "servis sonrası ürün eskisinden kötü çalışıyor.",
    "destek almak bu kadar zor olmamalı.",
    "şikayet oluşturmak için defalarca aradım.",
    "servis talebi oluşturuldu ama iptal edildi.",
    "teknik ekip geç kaldı.",
    "ürün beklenen şekilde çalışmıyor.",
    "müşteri temsilcisi kaba davrandı.",
    "servis raporu eksik doldurulmuş.",
    "sorunum hala çözülmedi.",
    "garanti süreciyle ilgili yanlış bilgi verildi.",
    "servis ziyareti çok kısa sürdü.",
    "ürünle ilgili net açıklama yapılmadı.",
    "teknik destek yetersiz kaldı.",
    "müşteri hizmetleri beni oyalıyor.",
    "aynı arıza tekrar yaşanıyor.",
    "servis randevusu oluşturuldu ama kimse gelmedi.",
    "teknik ekip geç geldiği için işlem yarım kaldı.",
    "servis çalışanı arızayı yeterince incelemedi.",
    "aynı servis kaydı defalarca açıldı.",
    "servis sonrası problem devam etti.",
    "teknik ekip sorunu geçici olarak çözüp gitti.",
    "servis personeli cihazı kontrol etmeden rapor yazdı.",
    "randevu saatinde evde olmama rağmen gelinmedi.",
    "servis ziyareti çok kısa sürdü.",
    "teknik servis parça değişimini erteledi.",
    "servis çalışanı yaptığı işlemi açıklamadı.",
    "aynı arıza için tekrar servis çağırmak zorunda kaldım.",
    "teknik ekip arızayı yanlış tespit etti.",
    "servis formunda yapılan işlemler eksik yazılmış.",
    "servis sonrası farklı bir sorun ortaya çıktı.",
    "teknik ekip yeterli ekipmanla gelmedi.",
    "servis randevusu son anda iptal edildi.",
    "servis çalışanı sorunu kullanıcı hatası olarak değerlendirdi.",
    "teknik servis garanti kapsamını yanlış yorumladı.",
    "servis süreci gereğinden uzun sürdü.",
    "teknik ekip parça olmadan geldi.",
    "servis sonrası test yapılmadı.",
    "servis personeli aceleci davrandı.",
    "teknik ekip sorunu net şekilde anlatmadı.",
    "servis ziyaretinden sonra tekrar arıza yaşandı.",
    "servis kaydı kapatıldı ama sorun çözülmedi.",
    "teknik ekip önceki servis notlarını okumamış.",
    "servis çalışanı geçici çözüm önerdi.",
    "servis sonrası ürün eskisi gibi çalışmadı.",
    "teknik servis sorunu hafife aldı.",
    "servis randevusu için çok uzun süre bekledim.",
    "teknik ekip sorunu görmezden geldi.",
    "servis personeli yetersiz bilgiye sahipti.",
    "servis sonrası tekrar çağrı açmam gerekti.",
    "teknik ekip arızayı tam olarak gideremedi.",
    "servis çalışanı detaylı kontrol yapmadı.",
    "servis raporu gerçeği yansıtmıyor.",
    "teknik servis yönlendirmesi yetersizdi.",
    "servis sonrası performans düşüklüğü yaşandı.",
    "teknik ekip aynı gün içinde çözüm sunmadı.",
    "servis personeli iletişim kurmakta zorlandı.",
    "servis işlemi yarım bırakıldı.",
    "teknik ekip sorunun kaynağını bulamadı.",
    "servis ziyareti plansız gerçekleşti.",
    "servis çalışanı sorunla ilgilenmedi.",
    "teknik servis yanlış parça değiştirdi.",
    "servis sonrası tekrar arıza oluştu.",
    "teknik ekip çözüm üretmeden ayrıldı.",
    "servis süreci profesyonel değildi.",
    "yıkama sırasında aşırı ses çıkarmaya başladı.",
    "sıkma aşamasında durup hata veriyor.",
    "program bitmesine rağmen kapak açılmıyor.",
    "iç kısmı yeterince soğuk kalmıyor.",
    "alt bölümü serinletmiyor ama üst taraf normal.",
    "kapak lastiğinden su sızıyor.",
    "çalışırken titreyip yerinden oynuyor.",
    "ısı ayarı yapılmasına rağmen istenen sıcaklığa ulaşmıyor.",
    "uzun süredir kullanmama rağmen ilk günden beri sorunlu.",
    "çalışma esnasında yanık kokusu geliyor.",
    "ekran üzerinde anlamsız bir hata kodu görünüyor.",
    "içerideki raflar beklenenden hızlı kırıldı.",
    "program seçimi yapıldıktan sonra kendiliğinden kapanıyor.",
    "fişe takılı olmasına rağmen tepki vermiyor.",
    "çalışma süresi eskisine göre çok uzadı.",
    "iç lambası bir süredir yanmıyor.",
    "ayar düğmeleri düzgün tepki vermiyor.",
    "yıkama sonunda çamaşırlar ıslak kalıyor.",
    "soğutma performansı gün geçtikçe düşüyor.",
    "yüksek sıcaklıkta kullanıldığında kendini kapatıyor.",
    "ilk çalıştırmada anormal sesler çıkarıyor.",
    "kapak tam kapalı olmasına rağmen uyarı veriyor.",
    "program ayarları kendi kendine değişiyor.",
    "iç kısımda buzlanma oluşmaya başladı.",
    "yüksek ısıda pişirme sırasında kapanıyor.",
    "çalışırken aşırı elektrik tüketiyor.",
    "temizlik sonrası performansı düştü.",
    "ekran parlaklığı zamanla azaldı.",
    "kapak menteşesi gevşedi.",
    "uzun süre çalışmasına rağmen beklenen sonucu vermiyor.",
    "çalışma sırasında metal sürtme sesi geliyor.",
    "ayarlar sıfırlanmış gibi davranıyor.",
    "alt kısımdan su birikiyor.",
    "önceden olmayan bir titreşim oluştu.",
    "düşük ayarda bile fazla ısınıyor.",
    "ilk programdan sonra hata vermeye başladı.",
    "iç hacim eskisi kadar verimli kullanılmıyor.",
    "kapak açıldığında koku yayılıyor.",
    "çalışma esnasında ani duraksamalar oluyor.",
    "yüksek devirde dengesiz çalışıyor.",
    "ısı dağılımı eşit değil.",
    "çalışırken ışıklar gidip geliyor.",
    "program tamamlanmadan kapanıyor.",
    "iç yüzeylerde beklenmedik deformasyon oluştu.",
    "ayar paneli dokunuşlara geç tepki veriyor.",
    "daha önce olmayan bir performans kaybı var.",
    "kullanım sırasında sürekli uyarı sesi veriyor.",
    "çalışma sonrası ortam aşırı ısınıyor.",
    "düşük ayarlarda bile yeterli soğutma sağlamıyor."]

# initialize the tokenizer and build the vocabulary based on the dataset
tokenizer = Tokenizer()
tokenizer.fit_on_texts(complaints)
total_words = len(tokenizer.word_index) + 1  # total number of unique tokens (+1 for padding)

# generate input sequences using n-gram approach for next-word prediction
input_sequences = []

for text in complaints:
    token_list = tokenizer.texts_to_sequences([text])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# find the maximum sequence length and pad all sequences to the same length
max_sl = max(len(x) for x in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen = max_sl, padding = "pre")

# split sequences into input features (X) and target word (y)
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes = total_words) # one-hot encode target words (y)

# build the LSTM-based language model 
lstm = Sequential()
lstm.add(Embedding(input_dim = total_words, output_dim = 100, input_length = X.shape[1]))
lstm.add(LSTM(units = 128, return_sequences = False))
lstm.add(Dense(total_words, activation = "softmax"))

# compile the model with optimizer and loss function suitable for multi-class classification
lstm.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

# train the model on the prepared input-output sequences
lstm.fit(X, y, epochs = 50, verbose = 1)

# function to generate text by predicting the next word iteratively
def prediction_func(seed_text, next_words):
    
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0] # convert current text to token sequence
        token_list = pad_sequences([token_list], maxlen = max_sl - 1, padding = "pre") # pad to required input length
        predicted_probs = lstm.predict(token_list, verbose = 0) # predict next-word probabilities
        predicted_word_index = np.argmax(predicted_probs, axis = -1) # select most probable word's index
        predicted_word = tokenizer.index_word[predicted_word_index[0]] # map index back to word
        seed_text = seed_text + " " + predicted_word # append predicted word to input text
    
    return seed_text

# examples
seed_text = "teknik servis"
print(prediction_func(seed_text, 3))

seed_text = "teknik ekip"
print(prediction_func(seed_text, 5))

seed_text = "klimam"
print(prediction_func(seed_text, 2))

seed_text = "fırın"
print(prediction_func(seed_text, 2))
