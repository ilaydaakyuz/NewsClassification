import re

class HashtagMentionRemover:
    @staticmethod
    def remove_hashtags(text):
        """
        Metindeki hashtag (#) ve mention (@) işaretlerini kaldırır.
        Eğer metin NaN veya None ise olduğu gibi döndürür.
        """
        if not isinstance(text, str):
            return text  # NaN veya None değerlerini olduğu gibi döndür
        return re.sub(r'[#@]\w+', '', text).strip()

