import re

class RepeatedCharRemover:
    @staticmethod
    def remove_repeated_chars(text):
        """
        Metindeki art arda tekrar eden karakterleri kaldırır.
        Örneğin: "hellooo" → "hello", "yeees" → "yes"
        """
        if not isinstance(text, str):
            return text  # NaN veya None değerlerini olduğu gibi döndür

        # Tekrar eden karakterleri yakalayarak temizleme
        return re.sub(r'(.)\1{2,}', r'\1', text)
