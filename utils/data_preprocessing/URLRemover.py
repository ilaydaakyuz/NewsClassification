import re

class URLRemover:
    @staticmethod
    def remove_urls(text):
        """
        Metindeki URL'leri kaldırır. Eğer metin NaN veya None ise, olduğu gibi döndürür.
        """
        if not isinstance(text, str):
            return text  # NaN veya metin olmayan değerleri olduğu gibi döndür
        return re.sub(r'http[s]?://\S+', '', text)
