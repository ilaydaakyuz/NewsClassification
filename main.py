import pandas as pd
import json
import os

def main():
    # JSON dosyasının doğru yolunu tanımlıyoruz
    dataset_path = os.path.join('data', 'News_Category_Dataset_v3.json')  # Dosya adı gereksinime göre düzenlenebilir
    
    # JSON dosyasını açma
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # Veriyi DataFrame'e dönüştürme
    df = pd.DataFrame(data)
    
    # İlk 5 satırı kontrol etmek için yazdırıyoruz
    print(df.head())

if __name__ == "__main__":
    main()
