import spacy
class Tokenizer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def tokenize(self, text):
        if not isinstance(text, str):
            return []

        # Diğer metinler için standart işlem
        doc = self.nlp(text)
        return [token.text for token in doc if not token.is_punct and not token.is_space]
