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
max_len = 100  # Modelin eğitildiği maksimum metin uzunluğu
models = {
    'cnn': load_model('cnn_model.h5'),
    'hybrid': load_model('hybrid_model.h5'),
    #'transformer': load_model('transformer_model.h5'),
    'lstm': load_model('lstm_model.h5'),
}
print(models.keys())  # Çıktı: dict_keys(['cnn', 'hybrid', 'lstm'])
categories = [
    'STYLE & BEAUTY', 
    'POLITICS', 
    'FOOD & DRINK', 
    'TRAVEL', 
    'PARENTING', 
    'WORLD NEWS', 
    'BUSINESS', 
    'ENTERTAINMENT', 
    'SPORTS', 
    'WELLNESS'
]
def preprocess_text(text, vocab, max_len):
    """
    Kullanıcının girdiği metni model için ön işleme.
    """
    tokens = text.lower().split()
    sequences = [[vocab.get(word, 0) for word in tokens]]
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded_sequences

def predict_category(request):
    """
    Model tahmin işlemini yapar ve sonucu döner.
    """
    prediction = None

    if request.method == "POST":
        news_text = request.POST.get('news_text')  # Kullanıcının girdiği metin
        selected_model = request.path.strip('/')  # URL'deki model ismini al

        if news_text and selected_model in models:
            processed_text = preprocess_text(news_text, vocab, max_len)
            model = models[selected_model]
            print(model)
            prediction_idx = model.predict(processed_text).argmax(axis=1)[0]
            prediction = categories[prediction_idx]
            print(prediction)
            print("Prediction Index:", prediction_idx)

    # Her model aynı şablona yönlenir.
    return render(request, 'index.html', {'prediction': prediction})

def index(request):
    """
    Ana sayfa.
    """
    return render(request, 'index.html')

