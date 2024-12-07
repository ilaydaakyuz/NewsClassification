from tensorflow.keras.preprocessing.sequence import pad_sequences

class PaddingHandler:
    @staticmethod
    def apply_padding(sequences, max_length=50):
        """
        Verilen dizilere padding uygular ve belirtilen uzunluğa getirir.
        """
        return pad_sequences(sequences, maxlen=max_length, padding='post').tolist()