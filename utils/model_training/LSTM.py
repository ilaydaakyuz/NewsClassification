from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPooling1D

class LSTMModel:  # Sınıf adını değiştirin
    def __init__(self, max_words=100000, max_len=100, num_classes=10):
        self.max_words = max_words
        self.max_len = max_len
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        # Doğrudan Keras katmanlarını kullanan bir model oluşturun
        model = Sequential([
            Embedding(self.max_words, 256, input_length=self.max_len),
            LSTM(64, return_sequences=True),  # Keras'ın LSTM katmanı
            GlobalMaxPooling1D(),
            Dense(64, activation='relu'),
            Dropout(0.6),
            Dense(self.num_classes, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model
        return model

    def train(self, X_train, y_train,validation_data=None, validation_split=0.2, epochs=5, batch_size=32):
        return self.model.fit(X_train, y_train,validation_data=validation_data, validation_split=validation_split, epochs=epochs, batch_size=batch_size)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)