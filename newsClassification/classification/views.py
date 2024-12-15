from django.shortcuts import render
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from django.conf import settings
import sys
import os
import numpy as np
from utils.data_preprocessing.LowerCaseConverter import LowerCaseConverter
from utils.data_preprocessing.URLRemover import URLRemover
from utils.data_preprocessing.RemovePunctuation import RemovePunctuation
from utils.data_preprocessing.RemoveStopwords import RemoveStopwords
from utils.data_preprocessing.Lemmatizer import Lemmatizer
from utils.data_preprocessing.Tokenizer import Tokenizer
from utils.model_training.Transformer import TokenAndPositionEmbedding, TransformerBlock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Dosya yollarını ayarla
vocab_path = os.path.join(settings.BASE_DIR, 'vocab.pkl')  # Kelime dağarcığı dosyası
vocab = pickle.load(open(vocab_path, 'rb'))
max_len = 100  # Modelin eğitildiği maksimum metin uzunluğu

# Custom katmanları tanımlayın
custom_objects = {
    "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
    "TransformerBlock": TransformerBlock,
}

categories = ['BUSINESS', 'ENTERTAINMENT', 'FOOD & DRINK', 'PARENTING', 'POLITICS', 'SPORTS', 'STYLE & BEAUTY', 'TRAVEL', 'WELLNESS', 'WORLD NEWS']

def preprocess_text_for_prediction(text, vocab, max_len):
    """
    Eğitimdeki ön işleme adımlarını tahmin sürecine uygular.
    """
    # 1. Küçük harfe çevirme
    text = LowerCaseConverter.to_lowercase(text)

    # 2. URL'leri kaldırma
    text = URLRemover.remove_urls(text)

    # 3. Noktalama işaretlerini kaldırma
    text = RemovePunctuation.remove_punctuation(text)

    # 4. Stopwords kaldırma
    text = RemoveStopwords.remove_stopwords(text)

    # 5. Lemmatization
    lemmatizer = Lemmatizer()
    text = lemmatizer.lemmatize(text)

    # 6. Tokenization
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(text)

    # 7. Kelimeleri indekse dönüştürme
    sequences = [[vocab.get(token, 0) for token in tokens]]

    # 8. Padding
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded_sequences

def load_model_for_prediction(model_name):
    """
    Modeli her tahmin için sıfırdan yükler.
    """
    model_path = f"{model_name}_model.h5"
    if model_name == 'transformer':
        model = load_model(model_path, custom_objects=custom_objects)
    else:
        model = load_model(model_path)
    return model

def predict_category(request):
    prediction = None

    if request.method == "POST":
        news_text = request.POST.get('news_text')  # Kullanıcının girdiği metin
        print(f"Input Text Received: {news_text}")
        selected_model = request.path.strip('/')  # URL'deki model ismini al

        if news_text and selected_model in ['cnn', 'hybrid', 'transformer', 'lstm']:
            # Tahmin öncesi ön işleme
            processed_text = preprocess_text_for_prediction(news_text, vocab, max_len)
            processed_text = np.repeat(processed_text, 32, axis=0)  # Batch boyutunu ayarla

            # Modeli yükle
            model = load_model_for_prediction(selected_model)
            print(f"Model loaded: {model}")
            # Model tahmini
            predictions = model.predict(processed_text)
                        
            # En yüksek olasılığı bul ve kategoriye dönüştür
            prediction_idx = predictions.mean(axis=0).argmax()
            prediction = categories[prediction_idx]
            print(f"Model: {selected_model}, Prediction Index: {prediction_idx}, Prediction: {prediction}")

    return render(request, 'index.html', {'prediction': prediction})


def index(request):
    """
    Ana sayfa.
    """
    return render(request, 'index.html')