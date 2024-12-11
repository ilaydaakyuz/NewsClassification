from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model

import pandas as pd
import numpy as np
import os
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.layers import Input
from utils.data_preprocessing.Tokenizer import Tokenizer
from utils.data_preprocessing.Lemmatizer import Lemmatizer
from utils.data_preprocessing.RemoveStopwords import RemoveStopwords
from utils.data_preprocessing.RepeatedCharRemover import RepeatedCharRemover
from utils.data_preprocessing.TextExpander import TextExpander
from utils.data_preprocessing.TextCorrector import TextCorrector
from utils.data_preprocessing.LowerCaseConverter import LowerCaseConverter
from utils.data_preprocessing.URLRemover import URLRemover
from utils.data_preprocessing.EmojiRemover import EmojiRemover
from utils.data_preprocessing.HashtagMentionRemover import HashtagMentionRemover
from utils.data_preprocessing.FillMissingValues import FillMissingValues
from utils.data_preprocessing.RemovePunctuation import RemovePunctuation
from utils.model_training.CNN import CNN
from utils.feature_preparation.PaddingHandler import PaddingHandler
from utils.feature_preparation.TextToSequence import TextToSequence
from utils.feature_preparation.VocabularyBuilder import VocabularyBuilder
from utils.model_training.Hybrid import Hybrid
from utils.visualization.ComparisonVisualizer import ComparisonVisualizer
from utils.visualization.LearningCurve import LearningCurve
from utils.model_training.Transformer import TokenAndPositionEmbedding, TransformerBlock
from utils.model_training.LSTM import LSTMModel



def main():
    

    # Stopwords'leri yükle
    load_stopwords()

    # Projenin konumunu bul
    project_root = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(project_root, 'data', 'NewsCategorizer.csv')
    processed_path = os.path.join(project_root, 'data', 'Processed_NewsCategorizer.csv')

    # Veriyi yükle
    df = load_dataset(dataset_path)
  
    # bilgisayarın yorulmasını önlemek için örnek 500 veri ile işlem yapıyoruz
    df = df.sample(n=500, random_state=42)  # random_state ile aynı veriyi seçmek için sabitlik sağlanır

    # her seferinde ön işlem adımları gerçekleşmesin diye direkt işlenmiş veriyi çekiyoruz
    #df= load_dataset(processed_path)
    
    # Veriyi ön işle
    df = preprocess_dataframe(df,project_root)
    
    # Özellik hazırlama işlemleri
    X, y = feature_preparation(df)

    # Metinleri sayısallaştır ve CNN modeli ile eğit
    train_models(X, y)

    # İşlenmiş veriyi kaydet
    save_preprocessed_data(df, processed_path)


def load_stopwords():
    try:
        stop_words = set(stopwords.words('english'))
        print("Stopwords başarıyla yüklendi!")
    except LookupError:
        print("Stopwords bulunamadı. Şimdi indiriliyor...")
        nltk.download('stopwords')


def load_dataset(file_path):
    """Veri setini yükler."""
    return pd.read_csv(file_path)


def preprocess_dataframe(df,project_root):
    """DataFrame üzerinde tüm ön işleme adımlarını uygular."""
    df = FillMissingValues.fill_missing_values(df)
    FillMissingValues.verify_missing_values(df)

    df = remove_urls_from_dataframe(df)
    df = remove_hashtags_from_dataframe(df)
    df = remove_emojis_from_dataframe(df)
    df = remove_punctuation_from_dataframe(df)
        
    df = convert_to_lowercase_in_dataframe(df)

    corrector = TextCorrector(project_root)
    df = apply_text_correction(df, ['headline', 'short_description'], corrector)

    df = apply_text_expansion(df, ['headline', 'short_description'])
    df = apply_repeated_char_removal(df, ['headline', 'short_description'])
    df = apply_stopword_removal(df, ['headline', 'short_description'])

    lemmatizer = Lemmatizer()
    df = apply_lemmatization(df, ['headline', 'short_description'], lemmatizer)

    tokenizer = Tokenizer()
    df = apply_tokenization(df, ['headline', 'short_description','keywords'], tokenizer)

    return df

# Veri sayısallaştırma için fonksiyonlar
def feature_preparation(df):
    """
    Veriyi eğitim için hazırlar. Tokenize edilmiş verilerle işlem yapar.
    """
    # Kategorileri sayısallaştır
    df = encode_labels(df)

    # One-hot encoding
    y = encode_one_hot(df)

    # Giriş metinlerini birleştir
    df = create_input_text_column(df)

    # Kelime dağarcığını oluştur
    vocab = build_vocabulary(df)

    # Metinleri indekslere dönüştür
    sequences = convert_text_to_sequences(df, vocab)

    # Dizileri padding ile sabit uzunlukta yap
    X = apply_padding_to_sequences(sequences)


    # NumPy dizilerine dönüştür
    X, y = convert_to_numpy_arrays(X, y)

    print("Feature preparation tamamlandı.")
    return X, y


def encode_one_hot(df):
    """Kategori etiketlerini One-hot encode eder."""
    y = to_categorical(df['category_encoded'])
    return y


def create_input_text_column(df):
    """Tokenize edilmiş sütunları birleştirip giriş metni oluşturur."""
    df['input_text'] = df['headline'] + df['short_description'] + df['keywords']
    return df


def build_vocabulary(df):
    """Kelime dağarcığını oluşturur."""
    vocab = VocabularyBuilder.build_vocab(df['headline'] + df['short_description'] + df['keywords'])
    print("Kelime dağarcığı oluşturuldu.")
    return vocab


def convert_text_to_sequences(df, vocab):
    """Metinleri kelime indekslerine dönüştürür."""
    sequences = TextToSequence.convert_to_sequence(df['input_text'].tolist(), vocab)
    print("Metinler dizilere dönüştürüldü.")
    return sequences


def apply_padding_to_sequences(sequences, max_length=100):
    """Dizileri padding ile sabit uzunlukta yapar."""
    padded_sequences = PaddingHandler.apply_padding(sequences, max_length=max_length)
    print("Padding işlemi tamamlandı.")
    return padded_sequences


def convert_to_numpy_arrays(X, y):
    """Verileri NumPy dizilerine dönüştürür."""
    X = np.array(X)
    y = np.array(y)
    print("Veriler NumPy dizilerine dönüştürüldü.")
    return X, y

def encode_labels(df):
    """Kategorik etiketleri sayısal değerlere dönüştürür."""
    label_encoder = LabelEncoder()
    df['category_encoded'] = label_encoder.fit_transform(df['category'])
    print(f"Kategoriler: {label_encoder.classes_}")
    return df


def save_preprocessed_data(df, save_path):
    """İşlenmiş veriyi kaydeder."""
    df.to_csv(save_path, index=False)
    print(f"İşlenmiş veri kaydedildi: {save_path}")


# Ön işleme adımları için fonksiyonlar
def remove_urls_from_dataframe(df):
    for col in ['headline', 'short_description']:
        if col in df.columns:
            df[col] = df[col].apply(URLRemover.remove_urls)
    return df


def remove_hashtags_from_dataframe(df):
    columns_to_clean = ['headline', 'short_description', 'keywords']
    for col in columns_to_clean:
        if col in df.columns:
            df[col] = df[col].apply(HashtagMentionRemover.remove_hashtags)
    return df


def remove_emojis_from_dataframe(df):
    columns_to_clean = ['headline', 'short_description', 'keywords']
    for col in columns_to_clean:
        if col in df.columns:
            df[col] = df[col].apply(EmojiRemover.remove_emojis)
    return df


def remove_punctuation_from_dataframe(df):
    columns_to_clean = ['headline', 'short_description', 'keywords']
    for col in columns_to_clean:
        if col in df.columns:
            df[col] = df[col].apply(RemovePunctuation.remove_punctuation)
    return df


def convert_to_lowercase_in_dataframe(df):
    columns_to_clean = ['headline', 'short_description', 'keywords']
    for col in columns_to_clean:
        if col in df.columns:
            df[col] = df[col].apply(LowerCaseConverter.to_lowercase)
    return df


def apply_text_correction(df, columns, corrector):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(corrector.correct_text)
    return df


def apply_text_expansion(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(TextExpander.expand_text)
    return df


def apply_repeated_char_removal(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(RepeatedCharRemover.remove_repeated_chars)
    return df


def apply_stopword_removal(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(RemoveStopwords.remove_stopwords)
    return df


def apply_lemmatization(df, columns, lemmatizer):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(lemmatizer.lemmatize)
    return df


def apply_tokenization(df, columns, tokenizer):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(tokenizer.tokenize)
    return df

def visualize(history):
   LearningCurve().plot_learning_curves(history)
   
   

def train_cnn(X, y):
    """
    Mevcut tokenize edilmiş verilerle CNN modelini eğitir.
    """
   
    # CNN'i oluştur ve eğit
    cnn = CNN(max_words=10000, max_len=100, num_classes=y.shape[1])
    cnn.build_model()

    # Modeli eğit
    history = cnn.train(X, y, validation_split=0.2, epochs=5, batch_size=32)
    print("Model eğitimi tamamlandı.")
    
    visualize(history)
    
    return history

def train_hybrid(X, y):
    """
    Mevcut tokenize edilmiş verilerle hibrit CNN + LSTM modelini eğitir.
    """
    # Hibrit modeli oluştur ve eğit
    hybrid_model = Hybrid(max_words=10000, max_len=100, num_classes=y.shape[1])
    hybrid_model.build_model()

    # Modeli eğit
    history = hybrid_model.train(X, y, validation_split=0.2, epochs=5, batch_size=32)
    print("Hibrit model eğitimi tamamlandı.")
    
    # Öğrenme eğrilerini görselleştir
    visualize(history)
    
    return history

def train_transformer(X, y):
    """
    Mevcut tokenize edilmiş verilerle Transformer modelini eğitir.
    """
    # Model parametreleri
    maxlen = X.shape[1]  # Giriş dizisinin uzunluğu
    vocab_size = 10000  # Kelime dağarcığı büyüklüğü
    embed_dim = 128  # Gömme boyutu
    num_heads = 4  # Çoklu başlık sayısı
    ff_dim = 128  # Beslemeli ileri katman boyutu
    num_classes = y.shape[1]  # Sınıf sayısı

    # Token ve pozisyon gömme
    inputs = Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)

    # Transformer bloğu
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    
    # Global Average Pooling ekleyelim
    x = GlobalAveragePooling1D()(x)
    
    # Sınıflandırma katmanları
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Modeli oluştur
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Modeli eğit
    history = model.fit(X, y, validation_split=0.2, epochs=5, batch_size=32)
    print("Transformer model eğitimi tamamlandı.")

    visualize(history)
    return history

def train_lstm(X, y):
    """
    Tokenize edilmiş verilerle LSTM modelini eğitir.
    """
    lstm_model = LSTMModel(max_words=10000, max_len=100, num_classes=y.shape[1])
    lstm_model.build_model()

    # Modeli eğit
    history = lstm_model.train(X, y, validation_split=0.2, epochs=5, batch_size=32)
    print("LSTM modeli eğitimi tamamlandı.")
    
    visualize(history)
    return history

def train_models(X, y):
    """
    CNN ve Hibrit modeli art arda eğitir ve sonuçlarını aynı grafikte karşılaştırır.
    """
    # CNN Eğitim
    print("CNN modeli eğitiliyor...")
    history_cnn = train_cnn(X, y)

    # Hibrit Model Eğitim
    print("Hibrit modeli eğitiliyor...")
    history_hybrid = train_hybrid(X, y)

    # Transformer Model Eğitim
    print("Transformer modeli eğitiliyor...")
    history_transformer = train_transformer(X, y)

    # LSTM Model Eğitim
    print("LSTM modeli eğitiliyor...")
    history_lstm = train_lstm(X, y)

    # Sonuçları görselleştir
    ComparisonVisualizer.visualize_comparison(history_cnn, history_hybrid,history_transformer,history_lstm)

    #LearningCurve().plot_learning_curves(history_transformer)


if __name__ == "__main__":
    main()