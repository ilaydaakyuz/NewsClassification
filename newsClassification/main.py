from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.backend import clear_session
import pandas as pd
import numpy as np
import os
import pickle
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
from utils.model_training.Transformer import Transformer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def main():
    
    # Stopwords'leri yükle
    load_stopwords()

    # Projenin konumunu bul
    project_root = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(project_root, 'data', 'NewsCategorizer.csv')
    processed_path = os.path.join(project_root, 'data', 'Processed_NewsCategorizer.csv')
    vocab_path = os.path.join(project_root, 'vocab.pkl')  # Kelime dağarcığı dosyası

    # Veriyi yükle
    #df = load_dataset(dataset_path)
    
    # her seferinde ön işlem adımları gerçekleşmesin diye direkt işlenmiş veriyi çekiyoruz
    df= load_dataset(processed_path)
    
    # bilgisayarın yorulmasını önlemek için örnek 500 veri ile işlem yapıyoruz
    #df = df.sample(n=20000, random_state=42)  # random_state ile aynı veriyi seçmek için sabitlik sağlanır

    # Veriyi ön işle
    #df = preprocess_dataframe(df,project_root)
    
    # Metinleri sayısallaştırma işlemleri
    X, y = feature_preparation(df,vocab_path)
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Modelleri ile eğit
    train_models(X_train, y_train, X_val, y_val)

    evaluate_models(X_test, y_test) 
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
def feature_preparation(df,vocab_path):
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
    vocab = build_vocabulary(df, vocab_path)

    # Metinleri indekslere dönüştür
    sequences = convert_text_to_sequences(df, vocab)

    # Dizileri padding ile sabit uzunlukta yap
    X = apply_padding_to_sequences(sequences)
    
    # NumPy dizilerine dönüştür
    X, y = convert_to_numpy_arrays(X, y)
    print("Eğitim sırasında X şekli:", X.shape)
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

def build_vocabulary(df,vocab_path):
    """Kelime dağarcığını oluşturur."""
    vocab = VocabularyBuilder.build_vocab(df['input_text'])
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
   
   
def train_cnn(X_train, y_train, X_val, y_val):
    """
    Mevcut tokenize edilmiş verilerle CNN modelini eğitir.
    """
    vocab_path="vocab.pkl"
    
    # Callback'ler
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    cnn = CNN(max_words=20000, max_len=100, num_classes=10, dropout_rate=0.05)
    cnn.build_model()
    
    # Modeli Eğit
    history = cnn.train(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=32,
        callbacks=[early_stopping, lr_scheduler]
    )
    
    # Modeli kaydet
    cnn.model.save('cnn_model.h5')

    with open('cnn_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    print("CNN Model eğitimi tamamlandı.")
    visualize(history)
    return history


def train_hybrid(X_train, y_train, X_val, y_val):
    """
    Mevcut tokenize edilmiş verilerle hibrit CNN + LSTM modelini eğitir.
    """
    # Hibrit modeli oluştur ve eğit
    hybrid_model = Hybrid(max_words=20000, max_len=100, num_classes=y_train.shape[1])
    hybrid_model.build_model()
    
    # Modeli eğit
    history = hybrid_model.train(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)
    print("Hibrit model eğitimi tamamlandı.")
    hybrid_model.model.save('hybrid_model.h5')
    # History'yi kaydet
    with open('hybrid_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    
    # Öğrenme eğrilerini görselleştir
    visualize(history)
    
    return history

def train_transformer(X_train, y_train, X_val, y_val):
    """
    Tokenize edilmiş verilerle Transformer modelini eğitir.
    """
    transformer_model = Transformer(max_words=20000, max_len=100, num_classes=y_train.shape[1])
    transformer_model.build_model()

    # Modeli eğit
    history = transformer_model.train(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)
    transformer_model.model.save('transformer_model.h5')

    # History'yi kaydet
    with open('transformer_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    print("Transformer modeli eğitimi tamamlandı.")

    # Öğrenme eğrilerini görselleştir
    visualize(history)

    return history

def train_lstm(X_train, y_train, X_val, y_val):
    """
    Tokenize edilmiş verilerle LSTM modelini eğitir.
    """
    lstm_model = LSTMModel(max_words=20000, max_len=100, num_classes=y_train.shape[1])
    lstm_model.build_model()

    # Modeli eğit
    history = lstm_model.train(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)
    print("LSTM modeli eğitimi tamamlandı.")

    lstm_model.model.save('lstm_model.h5')
    
    # History'yi kaydet
    with open('lstm_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    
    visualize(history)
    
    return history

def train_models(X_train, y_train, X_val, y_val):
    """
    Tüm modelleri eğitir ve history'lerini karşılaştırır.
    """
    # CNN Eğitim
    print("CNN modeli eğitiliyor...")
    train_cnn(X_train, y_train, X_val, y_val)
    clear_session()

    # Hibrit Model Eğitim
    print("Hibrit modeli eğitiliyor...")
    train_hybrid(X_train, y_train, X_val, y_val)
    clear_session()

    # Transformer Model Eğitim
    print("Transformer modeli eğitiliyor...")
    train_transformer(X_train, y_train, X_val, y_val)
    clear_session()

    # LSTM Model Eğitim
    print("LSTM modeli eğitiliyor...")
    train_lstm(X_train, y_train, X_val, y_val)
    clear_session()

    # Karşılaştırma
    ComparisonVisualizer.visualize_comparison()

def split_data(X, y, test_size=0.2, validation_size=0.2):
    # Test setini ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Doğrulama setini ayır
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test    

def evaluate_models(X_test, y_test):
    """
    Eğitilmiş modelleri test verisi üzerinde değerlendirir.
    """
    # CNN Modeli Yükleme ve Değerlendirme
    cnn_model = load_model('cnn_model.h5')
    cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test, y_test)
    print(f"CNN Model Test Loss: {cnn_loss}, Test Accuracy: {cnn_accuracy}")

    # Hibrit Model Yükleme ve Değerlendirme
    hybrid_model = load_model('hybrid_model.h5')
    hybrid_loss, hybrid_accuracy = hybrid_model.evaluate(X_test, y_test)
    print(f"Hybrid Model Test Loss: {hybrid_loss}, Test Accuracy: {hybrid_accuracy}")

    # Transformer Modeli Yükleme ve Değerlendirme (Özel Katmanlarla)
    custom_objects = {
        "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
        "TransformerBlock": TransformerBlock,
    }
    with custom_object_scope({'TokenAndPositionEmbedding': TokenAndPositionEmbedding, 'TransformerBlock': TransformerBlock}):
        transformer_model = load_model('transformer_model.h5')
    transformer_loss, transformer_accuracy = transformer_model.evaluate(X_test, y_test)
    print(f"Transformer Model Test Loss: {transformer_loss}, Test Accuracy: {transformer_accuracy}")

    # LSTM Modeli Yükleme ve Değerlendirme
    lstm_model = load_model('lstm_model.h5')
    lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test, y_test)
    print(f"LSTM Model Test Loss: {lstm_loss}, Test Accuracy: {lstm_accuracy}")

    test_results = {
        'cnn': {'loss': cnn_loss, 'accuracy': cnn_accuracy},
        'hybrid': {'loss': hybrid_loss, 'accuracy': hybrid_accuracy},
        'transformer': {'loss': transformer_loss, 'accuracy': transformer_accuracy},
        'lstm': {'loss': lstm_loss, 'accuracy': lstm_accuracy},
    }
    ComparisonVisualizer.visualize_comparison(test_results)

if __name__ == "__main__":
    main()
