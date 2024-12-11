from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, LSTM, Dense, Dropout, Concatenate

class Hybrid:
    def __init__(self, max_words=10000, max_len=100, num_classes=10):
        self.max_words = max_words
        self.max_len = max_len
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        # Giriş katmanı
        input_layer = Input(shape=(self.max_len,), name='input_layer')

        # Embedding katmanı
        embedding = Embedding(input_dim=self.max_words, output_dim=256, input_length=self.max_len)(input_layer)

        # CNN katmanı
        cnn = Conv1D(64, kernel_size=3, activation='relu', name='cnn_layer')(embedding)
        cnn = GlobalMaxPooling1D()(cnn)

        # LSTM katmanı
        lstm = LSTM(64, return_sequences=False, name='lstm_layer')(embedding)

        # CNN ve LSTM'yi birleştirme
        merged = Concatenate()([cnn, lstm])

        # Dense katmanları
        dense_1 = Dense(64, activation='relu')(merged)
        dropout = Dropout(0.3)(dense_1)
        output = Dense(self.num_classes, activation='softmax', name='output_layer')(dropout)
        


        # Modeli tanımlama
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model
        return model

    def train(self, X_train, y_train, validation_split=0.2, epochs=5, batch_size=32):
        history = self.model.fit(X_train, y_train, validation_split=validation_split, epochs=epochs, batch_size=batch_size)
        return history

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
