import matplotlib.pyplot as plt

class ComparisonVisualizer:
    @staticmethod
    def visualize_comparison(history_cnn, history_hybrid,history_transformer):

        # Eğitim kaybı (loss) karşılaştırması
        plt.figure()
        plt.plot(history_cnn.history['loss'], label="CNN - Training Loss")
        plt.plot(history_cnn.history['val_loss'], label="CNN - Validation Loss")
        plt.plot(history_hybrid.history['loss'], label="Hybrid - Training Loss")
        plt.plot(history_hybrid.history['val_loss'], label="Hybrid - Validation Loss")
        plt.plot(history_transformer.history['loss'], label="Transformer - Training Loss")
        plt.plot(history_transformer.history['val_loss'], label="Transformer - Validation Loss")
        plt.title("Training vs Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        # Eğitim doğruluğu (accuracy) karşılaştırması
        plt.figure()
        plt.plot(history_cnn.history['accuracy'], label="CNN - Training Accuracy")
        plt.plot(history_cnn.history['val_accuracy'], label="CNN - Validation Accuracy")
        plt.plot(history_hybrid.history['accuracy'], label="Hybrid - Training Accuracy")
        plt.plot(history_hybrid.history['val_accuracy'], label="Hybrid - Validation Accuracy")
        plt.plot(history_transformer.history['accuracy'], label="Transformer - Training Accuracy")
        plt.plot(history_transformer.history['val_accuracy'], label="Transformer - Validation Accuracy")
        plt.title("Training vs Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
