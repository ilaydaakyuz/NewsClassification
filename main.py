from utils.data_preprocessing.EmojiRemover import EmojiRemover
import pandas as pd
import json

def main():
    # JSON verisini satır satır okuma
  with open('C:\\Users\\Huawei\\OneDrive\\Masaüstü\\Haber Sınıflandırma\\NewsClassification\\data\\News_Category_Dataset_v3.json', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

    # DataFrame'e dönüştürme
    df = pd.DataFrame(data)
    print(df.head())  # İlk 5 satırı yazdırarak doğru şekilde okunduğunu kontrol edin

if __name__ == "__main__":
    main()

