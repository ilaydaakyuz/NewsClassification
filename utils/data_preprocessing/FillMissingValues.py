import pandas as pd
import numpy as np

class FillMissingValues:
    @staticmethod
    def fill_missing_values(df):
        """
        DataFrame'deki eksik veya boş alanları, önceki ve sonraki 10 habere bakarak doldurur.
        """
        # Tüm sütunları kontrol et
        for col in df.columns:
            if col in ['headline', 'short_description', 'category', 'keywords']:
                # Metin sütunları için boş değerleri NaN yap
                df[col] = df[col].fillna("").replace("", np.nan)  # Boş değerleri NaN yap
                df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)  # Fazladan boşlukları temizle

                # Eksik olan metinleri önceki ve sonraki 10 haberin metniyle doldur
                for i in range(len(df)):
                    if pd.isna(df[col].iloc[i]):
                        # İlk 10 haberi almak için pencere kullanımı
                        previous_text = " ".join(df[col].iloc[max(0, i-10):i].dropna())  # Önceki 10 haber
                        next_text = " ".join(df[col].iloc[i+1:i+11].dropna())  # Sonraki 10 haber
                        
                        # Önceki ve sonraki metni birleştirip doldur
                        combined_text = f"{previous_text} {next_text}".strip()
                        df.iloc[i, df.columns.get_loc(col)] = combined_text if combined_text else "Unknown"

            elif col == 'date':
                # Tarih sütunundaki eksik değerleri önceki ve sonraki tarihlerle doldur
                df[col] = pd.to_datetime(df[col], errors='coerce')  # Geçerli olmayan tarihleri NaT yap
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')  # İleri ve geri doldurma
            
            elif col == 'category':
                # Kategori için eksik değerleri en sık görülen kategori ile doldur
                most_common = df[col].mode()[0]
                df[col] = df[col].fillna(most_common)

            else:
                # Numerik sütunlar için önceki ve sonraki 10 değerin ortalamasını kullan
                df[col] = pd.to_numeric(df[col], errors='coerce')  # Numerik değer yap
                df[col] = df[col].fillna(df[col].rolling(window=21, min_periods=1, center=True).mean())
        
        return df

    @staticmethod
    def verify_missing_values(df):
        """
        DataFrame'deki eksik değerlerin kontrol edilmesi ve doğrulama mesajı.
        """
        missing_data = df.isnull().sum().sum()
        if missing_data == 0:
            print("Veri doldurma işlemi başarıyla tamamlandı.")
        else:
            print(f"Veri doldurma işlemi tamamlanamadı. Hala {missing_data} eksik değer bulunuyor.")
