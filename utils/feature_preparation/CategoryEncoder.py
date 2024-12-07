from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

class CategoryEncoder:
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def encode_categories(self, categories):
        """
        Kategorileri sayısal değerlere dönüştürür.
        """
        encoded = self.label_encoder.fit_transform(categories)
        return encoded

    def one_hot_encode(self, encoded_categories):
        """
        One-hot encoding uygular.
        """
        return to_categorical(encoded_categories)