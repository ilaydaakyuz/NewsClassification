import pandas as pd
import os
from utils.data_preprocessing.URLRemover import URLRemover
from utils.data_preprocessing.HashtagMentionRemover import HashtagMentionRemover
from utils.data_preprocessing.FillMissingValues import FillMissingValues

def main():
    # Excel dosyasının doğru yolunu tanımlıyoruz
    dataset = os.path.join('data', 'NewsCategorizer.csv')  # Excel dosyasının adı ve yolu
    
    df = pd.read_csv('data/NewsCategorizer.csv')

# Eksik değerleri doldurma
    df = FillMissingValues.fill_missing_values(df)

# Eksik değerleri kontrol etme
    FillMissingValues.verify_missing_values(df)


    print("Eksik veri doldurma işlemi tamamlandı!")

    # URL'leri kaldırma işlemi
    df = remove_urls_from_dataframe(df)

    # Hashtag'leri kaldırma işlemi
    df = remove_hashtags_from_dataframe(df)

    processed_path = 'data/Processed_NewsCategorizer.csv'
    df.to_csv(processed_path, index=False)
    print(f"İşlenmiş veri kaydedildi: {processed_path}")
    
    # İlk 5 satırı kontrol etmek için yazdırıyoruz
    print(df.head())

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
  
if __name__ == "__main__":
    main()
