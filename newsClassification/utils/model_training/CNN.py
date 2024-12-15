from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class CNN:
    def __init__(self, max_words=20000, max_len=100, num_classes=10, optimizer=None, dropout_rate=0.05, l2_reg=0.001):
        """
        CNN sınıfı oluşturur.
        :param max_words: Kelime dağarcığı boyutu
        :param max_len: Maksimum metin uzunluğu
        :param num_classes: Çıkış sınıfı sayısı
        :param optimizer: Kullanılacak optimizer
        :param dropout_rate: Dropout oranı
        :param l2_reg: L2 regularization oranı
        """
        self.max_words = max_words
        self.max_len = max_len
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate  # Dropout oranını parametre olarak al
        self.l2_reg = l2_reg  # L2 Regularization parametresi
        self.optimizer = optimizer if optimizer else Adam(learning_rate=0.0002)  # Varsayılan olarak Adam optimizer kullan
        self.model = None

    def build_model(self):
        """
        CNN modelini oluşturur.
        """
        model = Sequential([
            Embedding(self.max_words, 300, input_length=self.max_len),
            Conv1D(128, 5, activation='relu'),
            Conv1D(64, 5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.05),
            Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.05),
            Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.05),
            Dense(self.num_classes, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        self.model = model
        return model

    def train(self, X_train, y_train, validation_data=None, validation_split=0.2, epochs=20, batch_size=32, callbacks=None):
        """
        Modeli eğitir.
        """
        if callbacks is None:
            # Varsayılan callbackler: EarlyStopping ve ReduceLROnPlateau
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
            ]

        return self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )

