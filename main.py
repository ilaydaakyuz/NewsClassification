import pandas as pd
import json
import os
from utils.data_preprocessing.URLRemover import URLRemover
from utils.data_preprocessing.HashtagMentionRemover import HashtagMentionRemover
from utils.data_preprocessing.FillMissingValues import FillMissingValues

def main():
    # JSON dosyasının doğru yolunu tanımlıyoruz
    dataset = os.path.join('data', 'News_Category_Dataset_v3.json')  
    
    # JSON dosyasını açma
    with open(dataset, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # Veriyi DataFrame'e dönüştürme
    df = pd.DataFrame(data)

    # Eksik verileri doldurma işlemi
    df = FillMissingValues.fill_missing_values(df)
    # Eksik değerlerin kontrol edilmesi ve mesaj yazdırılması
    FillMissingValues.verify_missing_values(df)

    # URL'leri kaldırma işlemi
    df = remove_urls_from_dataframe(df)

    # Hashtag'leri kaldırma işlemi
    df = remove_hashtags_from_dataframe(df)
    
    # İlk 5 satırı kontrol etmek için yazdırıyoruz
    print(df.head())

def remove_urls_from_dataframe(df):
    """
    DataFrame'in 'headline' ve 'short_description' sütunlarından URL'leri kaldırır.
    """
    df['headline'] = df['headline'].apply(URLRemover.remove_urls)
    df['short_description'] = df['short_description'].apply(URLRemover.remove_urls)
    print("URL temizlemesi başarılı")
    return df    

def remove_hashtags_from_dataframe(df):
    """
    DataFrame'in 'headline' ve 'short_description' sütunlarından hashtag'leri kaldırır.
    """
    df['headline'] = df['headline'].apply(HashtagMentionRemover.remove_hashtags)
    df['short_description'] = df['short_description'].apply(HashtagMentionRemover.remove_hashtags)
    print("Hashtag temizlemesi başarılı")
    return df  
if __name__ == "__main__":
    main()
