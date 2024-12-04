import pandas as pd
import json
import os

def main():
    # JSON dosyasının doğru yolunu tanımlıyoruz
    dataset = os.path.join('data', 'News_Category_Dataset_v3.json')  
    
    # JSON dosyasını açma
    with open(dataset, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # Veriyi DataFrame'e dönüştürme
    df = pd.DataFrame(data)
    
    # İlk 5 satırı kontrol etmek için yazdırıyoruz
    #daha sonra silinebilir
    #print(df.head())

if __name__ == "__main__":
    main()
