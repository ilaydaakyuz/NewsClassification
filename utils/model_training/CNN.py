from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

class CNN:
    def __init__(self, max_words=100000, max_len=100, num_classes=10):
        self.max_words = max_words
        self.max_len = max_len
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        model = Sequential([
            Embedding(self.max_words, 128, input_length=self.max_len),
            Conv1D(128, 5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model
        return model

    def train(self, X_train, y_train, validation_split=0.2, epochs=10, batch_size=32):
        return self.model.fit(X_train, y_train, validation_split=validation_split, epochs=epochs, batch_size=batch_size)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
