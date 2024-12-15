class TextToSequence:
    @staticmethod
    def convert_to_sequence(text_list, vocab):
        """
        Tokenize edilmiş metinleri kelime dağarcığına göre indekslere çevirir.
        """
        return [[vocab.get(word, 0) for word in tokens] for tokens in text_list]