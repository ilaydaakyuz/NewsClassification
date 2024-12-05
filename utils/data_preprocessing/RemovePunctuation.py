import re
import string

class RemovePunctuation:
    @staticmethod
    def remove_punctuation(text):
        """
        Metindeki noktalama işaretlerini kaldırır. Eğer metin NaN veya None ise olduğu gibi döndürür.
        """
        if not isinstance(text, str):
            return text  # NaN veya None değerlerini olduğu gibi döndür

        # Noktalama işaretlerini kaldırma
        return re.sub(f"[{re.escape(string.punctuation)}]", '', text).strip()
