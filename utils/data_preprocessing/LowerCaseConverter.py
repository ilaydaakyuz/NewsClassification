class LowerCaseConverter:
    @staticmethod
    def to_lowercase(text):
        """
        Metni küçük harfe dönüştürür. Eğer metin NaN veya None ise olduğu gibi döndürür.
        """
        if not isinstance(text, str):
            return text  # NaN veya None değerlerini olduğu gibi döndür
        return text.lower()
