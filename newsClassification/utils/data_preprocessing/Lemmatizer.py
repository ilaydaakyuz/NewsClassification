import spacy

class Lemmatizer:
    def __init__(self):
        # İngilizce modeli yükle
        self.nlp = spacy.load("en_core_web_sm")

    def lemmatize(self, text):
        """
        Metni lemmatization işlemiyle kök forma dönüştürür.
        """
        if not isinstance(text, str):
            return text  # NaN veya None değerlerini olduğu gibi döndür

        doc = self.nlp(text)
        return " ".join([token.lemma_ for token in doc])
