from django.shortcuts import render
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from django.conf import settings  # Django ayarlarını kullanmak için
#from tensorflow.keras.utils import custom_object_scope
import os


# Dosya yollarını ayarla
vocab_path = os.path.join(settings.BASE_DIR, 'vocab.pkl')  # 'vocab.pkl' dosyasının tam yolu
models_path = os.path.join(settings.BASE_DIR, '')  # Modellerin bulunduğu dizin
    
# Modelleri ve kelime dağarcığını yükleyin
vocab = pickle.load(open('vocab.pkl', 'rb'))  # Kelime dağarcığı dosyası
max_len = 1000  # Modelin eğitildiği maksimum metin uzunluğu
models = {
    'cnn': load_model('cnn_model.h5'),
    'hybrid': load_model('hybrid_model.h5'),
    #'transformer': load_model('transformer_model.h5'),
    'lstm': load_model('lstm_model.h5'),
}
print(models.keys())  # Çıktı: dict_keys(['cnn', 'hybrid', 'lstm'])

categories = ['BUSINESS', 'ENTERTAINMENT', 'FOOD & DRINK', 'PARENTING', 'POLITICS', 'SPORTS', 'STYLE & BEAUTY', 'TRAVEL', 'WELLNESS', 'WORLD NEWS']

def preprocess_text(text, vocab, max_len):
    """
    Kullanıcının girdiği metni model için ön işleme.
    """
    tokens = text.lower().split()
    sequences = [[vocab.get(word, 0) for word in tokens]]
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    print("Giriş Metni Tokenize:", tokens)
    print("Kelime İndeksleri:", sequences)
    print("Padded Input:", padded_sequences)
    
    return padded_sequences

def predict_category(request):
    prediction = None

    if request.method == "POST":
        news_text = request.POST.get('news_text')  # Kullanıcının girdiği metin
        selected_model = request.path.strip('/')  # URL'deki model ismini al

        if news_text and selected_model in models:
            processed_text = preprocess_text(news_text, vocab, max_len)
            model = models[selected_model]
            predictions = model.predict(processed_text)
            
            print("Ham Tahmin Değerleri:", predictions)
            
            prediction_idx = predictions.argmax(axis=1)[0]
            prediction = categories[prediction_idx]
            print(f"Model: {selected_model}, Prediction Index: {prediction_idx}, Prediction: {prediction}")

    return render(request, 'index.html', {'prediction': prediction})


def index(request):
    """
    Ana sayfa.
    """
    return render(request, 'index.html')

