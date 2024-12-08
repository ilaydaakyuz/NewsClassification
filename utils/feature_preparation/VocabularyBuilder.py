class VocabularyBuilder:
    @staticmethod
    def build_vocab(tokenized_texts, max_vocab_size=10000):
        """
        Kelime dağarcığı oluşturur ve her kelimeye bir indeks atar.
        """
        all_words = [word for tokens in tokenized_texts for word in tokens]
        vocab = {word: idx + 1 for idx, word in enumerate(set(all_words))}
        print(f"Kelime dağarcığı büyüklüğü: {len(vocab)}")
        return vocab