from nltk.corpus import stopwords
import re

class RemoveStopwords:
    # Stopword listesini yükleme (İngilizce için)
    stop_words = set(stopwords.words('english'))

    @staticmethod
    def remove_stopwords(text):
        """
        Metindeki stopword'leri kaldırır.
        """
        if not isinstance(text, str):
            return text  # NaN veya None değerlerini olduğu gibi döndür

        # Metni kelimelerine ayır ve stopword olmayanları seç
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in RemoveStopwords.stop_words]

        # Stopword'ler temizlenmiş metni birleştir
        return " ".join(filtered_words)
