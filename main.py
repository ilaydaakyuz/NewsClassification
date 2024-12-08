from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
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
from utils.model_training.CNN.CNNModel import CNNModel
from utils.feature_preparation.PaddingHandler import PaddingHandler
from utils.feature_preparation.TextToSequence import TextToSequence
from utils.feature_preparation.VocabularyBuilder import VocabularyBuilder

def main():
    

    # Stopwords'leri yükle
    load_stopwords()

    # Projenin konumunu bul
    project_root = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(project_root, 'data', 'NewsCategorizer.csv')
    processed_path = os.path.join(project_root, 'data', 'Processed_NewsCategorizer.csv')

    # Veriyi yükle
    df = load_dataset(dataset_path)

    # Her seferinde ön işlem adımları gerçekleşmesin diye direkt işlenmiş veriyi çekiyoruz
    #df= load_dataset(processed_path)
    
    # Veriyi ön işle
    df = preprocess_dataframe(df,project_root)
    
    # Özellik hazırlama işlemleri
    X, y = feature_preparation(df)

    # Metinleri sayısallaştır ve CNN modeli ile eğit
    train_cnn_model(X, y)

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
    y = to_categorical(df['category_encoded'])

    # Tokenize edilmiş headline, short_description ve keywords verilerini birleştiriyoruz
    df['input_text'] = df['headline'] + df['short_description'] + df['keywords']

    X = df['input_text']
  
    # Kelime dağarcığını oluştur
    vocab = VocabularyBuilder.build_vocab(df['headline'] + df['short_description'] + df['keywords'])

    # TextToSequence sınıfını kullanarak kelimeleri indekslere dönüştür
    sequences = TextToSequence.convert_to_sequence(df['input_text'].tolist(), vocab)


    # PaddingHandler ile sabit uzunlukta dizilere dönüştür
    X = PaddingHandler.apply_padding(sequences, max_length=100)

    # NumPy dizisine dönüştür
    X = np.array(X)
    y = np.array(y)


    print("Feature preparation tamamlandı.")
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

def train_cnn_model(X, y):
    """
    Mevcut tokenize edilmiş verilerle CNN modelini eğitir.
    """
   
    # CNNModel'i oluştur ve eğit
    cnn_model = CNNModel(max_words=10000, max_len=100, num_classes=y.shape[1])
    cnn_model.build_model()

    # Modeli eğit
    history = cnn_model.train(X, y, validation_split=0.2, epochs=10, batch_size=32)
    print("Model eğitimi tamamlandı.")
    return history


if __name__ == "__main__":
    main()