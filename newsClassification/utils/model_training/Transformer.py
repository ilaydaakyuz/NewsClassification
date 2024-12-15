import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dense, Dropout, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class Transformer:
    def __init__(self, max_words=10000, max_len=100, num_classes=10):
        self.max_words = max_words
        self.max_len = max_len
        self.num_classes = num_classes
        self.model = None
        self.embed_dim = 128
        self.num_heads = 4
        self.ff_dim = 64

    def build_model(self):
        inputs = tf.keras.Input(shape=(self.max_len,))
        
        # Embedding katmanı
        embedding_layer = TokenAndPositionEmbedding(
            maxlen=self.max_len,
            vocab_size=self.max_words,
            embed_dim=self.embed_dim
        )
        x = embedding_layer(inputs)
        
        # Transformer katmanı
    
        transformer_block = TransformerBlock(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
                 
        )
        x = transformer_block(x)
        
        
        
        # Global pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Dense katmanları
        x = Dense(64, activation="relu", kernel_regularizer=l2(0.001))(x)
        x = Dropout(0.05)(x)  
        outputs = Dense(self.num_classes, activation="softmax")(x)

        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Model oluşturma
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        return self.model

    def train(self, X_train, y_train, validation_data=None, validation_split=0.2, epochs=5, batch_size=32):
        return self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size
        )

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

