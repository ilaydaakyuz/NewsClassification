import spacy

class Tokenizer:
    def __init__(self):
        # İngilizce dil modeli
        self.nlp = spacy.load("en_core_web_sm")

    def tokenize(self, text):
        """
        Metni tokenlere ayırır. Noktalama işaretlerini, boşlukları ve stopword'leri çıkarır.
        """
        if not isinstance(text, str):
            return []
        
        # spaCy'de metni işlemden geçir
        doc = self.nlp(text)

        # Noktalama işaretleri, boşluklar ve stopword'leri çıkar
        tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
        return tokens
