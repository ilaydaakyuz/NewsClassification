class TextExpander:
    # Kısaltma-tam anlam eşleme tablosu
    abbreviation_map = {
        "idk": "I don't know",
        "btw": "by the way",
        "omg": "oh my god",
        "brb": "be right back",
        "lol": "laugh out loud",
        "smh": "shaking my head",
        "fyi": "for your information",
        "asap": "as soon as possible",
        "tbh": "to be honest",
        "imho": "in my humble opinion"
    }

    @staticmethod
    def expand_text(text):
        """
        Metindeki kısaltmaları tam anlamlarına dönüştürür.
        """
        if not isinstance(text, str):
            return text  # NaN veya None değerlerini olduğu gibi döndür

        words = text.split()  # Metni kelimelerine ayır
        expanded_words = [TextExpander.abbreviation_map.get(word.lower(), word) for word in words]
        return " ".join(expanded_words)  # Dönüştürülmüş kelimeleri birleştir
