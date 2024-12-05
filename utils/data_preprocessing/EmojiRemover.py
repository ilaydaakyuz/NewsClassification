import re

class EmojiRemover:
    @staticmethod
    def remove_emojis(text):
        """
        Metindeki emojileri kaldırır. Eğer metin NaN veya None ise olduğu gibi döndürür.
        """
        if not isinstance(text, str):
            return text  # NaN veya None değerlerini olduğu gibi döndür

        # Unicode emoji karakterlerini temizlemek için regex
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # Smileys & Emotion
            u"\U0001F300-\U0001F5FF"  # Symbols & Pictographs
            u"\U0001F680-\U0001F6FF"  # Transport & Map
            u"\U0001F700-\U0001F77F"  # Alchemical Symbols
            u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            u"\U0001FA00-\U0001FA6F"  # Chess Symbols
            u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            u"\U00002702-\U000027B0"  # Dingbats
            u"\U000024C2-\U0001F251"  # Enclosed Characters
            "]+", 
            flags=re.UNICODE
        )
        return emoji_pattern.sub(r'', text).strip()
