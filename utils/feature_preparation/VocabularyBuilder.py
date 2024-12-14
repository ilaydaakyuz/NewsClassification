import pickle

class VocabularyBuilder:
    @staticmethod
    def build_vocab(series, max_vocab_size=10000):
        """
        Kelime dağarcığı oluşturur ve her kelimeye bir indeks atar.
        """
        vocab_set = set()  # Tekrarlı kelimeleri engellemek için set kullanıyoruz.

        # Series içerisindeki her bir satırı işleyip kelimeleri toplama
        for text in series:
            # Her metni temizle ve kelimelere ayır
            text_cleaned = text.replace('[', '').replace(']', '').replace("'", "")
            tokens = text_cleaned.split()
            vocab_set.update(tokens)  # Kelimeleri vocab_set'e ekle

        # Set'i listeye çevir ve boyutunu sınırla
        vocab_list = list(vocab_set)[:max_vocab_size]

        # Kelimelere indeks ata
        vocab = {word: i + 1 for i, word in enumerate(vocab_list)}  # İndeksler 1'den başlıyor
        print("Kelime Dağarcığından Örnekler:", list(vocab.items())[:10])
        
        # Kaydet ve yükle
        VocabularyBuilder.save_vocab(vocab)
        vocab = VocabularyBuilder.load_vocab()
        return vocab
    
    @staticmethod
    def save_vocab(vocab, file_path='vocab.pkl'):
        """
        Kelime dağarcığını dosyaya kaydeder.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(vocab, f)
        print(f"Kelime dağarcığı kaydedildi: {file_path}")

    @staticmethod
    def load_vocab(file_path='vocab.pkl'):
        """
        Kelime dağarcığını dosyadan yükler.
        """
        with open(file_path, 'rb') as f:
            vocab = pickle.load(f)
        print(f"Kelime dağarcığı yüklendi: {file_path}")
        return vocab