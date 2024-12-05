import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
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


try:
    stop_words = set(stopwords.words('english'))
    print("Stopwords başarıyla yüklendi!")
except LookupError:
    print("Stopwords bulunamadı. Şimdi indiriliyor...")
    nltk.download('stopwords')
    
def main():
    # Excel dosyasının doğru yolunu tanımlıyoruz
    dataset = os.path.join('data', 'NewsCategorizer.csv')  # Excel dosyasının adı ve yolu
    
    df = pd.read_csv('data/NewsCategorizer.csv')

# Eksik değerleri doldurma
    df = FillMissingValues.fill_missing_values(df)

# Eksik değerleri kontrol etme
    FillMissingValues.verify_missing_values(df)

    df = remove_urls_from_dataframe(df)
    df = remove_hashtags_from_dataframe(df)
    df = remove_emojis_from_dataframe(df)
    df = remove_punctuation_from_dataframe(df)
    df = convert_to_lowercase_in_dataframe(df)

    
    corrector = TextCorrector()

    # Yazım düzeltme işlemi
    columns_to_correct = ['headline', 'short_description']
    for col in columns_to_correct:
        if col in df.columns:
            df = apply_text_correction(df, col, corrector)

    # Metin genişletme işlemi
    columns_to_expand = ['headline', 'short_description']
    for col in columns_to_expand:
        if col in df.columns:
            df = apply_text_expansion(df, col)

     # Tekrar eden karakterleri temizleme işlemi
    columns_to_clean = ['headline', 'short_description']
    for col in columns_to_clean:
        if col in df.columns:
            df = apply_repeated_char_removal(df, col)

    # Stopword temizleme işlemi
    columns_to_clean = ['headline', 'short_description']
    for col in columns_to_clean:
        if col in df.columns:
            df = apply_stopword_removal(df, col)  

    # Lemmatization işlemi
    lemmatizer = Lemmatizer()
    columns_to_lemmatize = ['headline', 'short_description']
    for col in columns_to_lemmatize:
        if col in df.columns:
            df = apply_lemmatization(df, col, lemmatizer)     

     # Tokenizer oluştur
    tokenizer = Tokenizer()

    # Tokenization işlemi
    columns_to_tokenize = ['headline', 'short_description']
    for col in columns_to_tokenize:
        if col in df.columns:
            df = apply_tokenization(df, col, tokenizer)           

   

    processed_path = 'data/Processed_NewsCategorizer.csv'
    df.to_csv(processed_path, index=False)
    print(f"İşlenmiş veri kaydedildi: {processed_path}")
    
    # İlk 5 satırı kontrol etmek için yazdırıyoruz
    #print(df.head())

def remove_urls_from_dataframe(df):
    """
    DataFrame'in 'headline' ve 'short_description' sütunlarından URL'leri kaldırır.
    """
    for col in ['headline', 'short_description']:
        if col in df.columns:
            df[col] = df[col].apply(URLRemover.remove_urls)  # URL'leri temizle
    print("URL temizlemesi başarılı")
    return df  

def remove_hashtags_from_dataframe(df):
    """
    DataFrame'in 'headline', 'short_description' ve 'keywords' sütunlarından hashtag'leri kaldırır.
    """
    # İşlenecek sütunlar
    columns_to_clean = ['headline', 'short_description', 'keywords']

    for col in columns_to_clean:
        if col in df.columns:  # Sütunun varlığını kontrol et
            df[col] = df[col].apply(HashtagMentionRemover.remove_hashtags)
            print(f"{col} sütunundan hashtag'ler temizlendi.")
    
    print("Hashtag temizleme işlemi tamamlandı.")
    return df

def remove_emojis_from_dataframe(df):
    """
    DataFrame'in 'headline', 'short_description' ve 'keywords' sütunlarından emojileri kaldırır.
    """
    # İşlenecek sütunlar
    columns_to_clean = ['headline', 'short_description', 'keywords']

    for col in columns_to_clean:
        if col in df.columns:  # Sütunun varlığını kontrol et
            df[col] = df[col].apply(EmojiRemover.remove_emojis)
            print(f"{col} sütunundan emojiler temizlendi.")
    
    print("Emoji temizleme işlemi tamamlandı.")
    return df

def remove_punctuation_from_dataframe(df):
    """
    DataFrame'in 'headline', 'short_description' ve 'keywords' sütunlarından noktalama işaretlerini kaldırır.
    """
    # İşlenecek sütunlar
    columns_to_clean = ['headline', 'short_description', 'keywords']

    for col in columns_to_clean:
        if col in df.columns:  # Sütunun varlığını kontrol et
            df[col] = df[col].apply(RemovePunctuation.remove_punctuation)
            print(f"{col} sütunundan noktalama işaretleri temizlendi.")
    
    print("Noktalama işaretleri temizleme işlemi tamamlandı.")
    return df

def convert_to_lowercase_in_dataframe(df):
    """
    DataFrame'in 'headline', 'short_description' ve 'keywords' sütunlarını küçük harfe dönüştürür.
    """
    # İşlenecek sütunlar
    columns_to_clean = ['headline', 'short_description', 'keywords']

    for col in columns_to_clean:
        if col in df.columns:  # Sütunun varlığını kontrol et
            df[col] = df[col].apply(LowerCaseConverter.to_lowercase)
            print(f"{col} sütunundaki metinler küçük harfe dönüştürüldü.")
    
    print("Küçük harfe dönüştürme işlemi tamamlandı.")
    return df

def apply_text_correction(df, column, corrector):
    """
    Sadece belirtilen sütunda yazım düzeltmesi uygular ve işlemi parçalara böler.
    """
    batch_size = 1000  # Parça boyutu
    for i in range(0, len(df), batch_size):
        df.iloc[i:i+batch_size, df.columns.get_loc(column)] = (
            df.iloc[i:i+batch_size][column].apply(corrector.correct_text)
        )
    print(f"{column} sütunundaki yazım hataları düzeltildi.")
    return df


def apply_text_expansion(df, column):
    """
    DataFrame'in belirtilen sütununda kısaltmaları genişletir.
    """
    df[column] = df[column].apply(TextExpander.expand_text)
    print(f"{column} sütunundaki kısaltmalar genişletildi.")
    return df

def apply_repeated_char_removal(df, column):
    """
    DataFrame'in belirtilen sütunundaki tekrar eden karakterleri kaldırır.
    """
    df[column] = df[column].apply(RepeatedCharRemover.remove_repeated_chars)
    print(f"{column} sütunundaki tekrar eden karakterler temizlendi.")
    return df

def apply_stopword_removal(df, column):
    """
    DataFrame'in belirtilen sütunundaki stopword'leri kaldırır.
    """
    df[column] = df[column].apply(RemoveStopwords.remove_stopwords)
    print(f"{column} sütunundaki stopword'ler kaldırıldı.")
    return df

def apply_lemmatization(df, column, Lemmatizer):
    """
    DataFrame'in belirtilen sütununa lemmatization uygular.
    """
    df[column] = df[column].apply(Lemmatizer.lemmatize)
    print(f"{column} sütununa lemmatization uygulandı.")
    return df

def apply_tokenization(df, column, Tokenizer):
    """
    DataFrame'in belirtilen sütunundaki metinlere tokenization uygular.
    """
    df[column] = df[column].apply(Tokenizer.tokenize)
    print(f"{column} sütununa tokenization uygulandı.")
    return df






  
if __name__ == "__main__":
    main()
