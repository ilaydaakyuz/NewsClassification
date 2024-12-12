import matplotlib.pyplot as plt
import pickle

class ComparisonVisualizer:
    @staticmethod
    def load_history(file_path):
        """
        Belirtilen dosya yolundan history verisini yükler.
        """
        with open(file_path, 'rb') as f:
            history = pickle.load(f)
        return history

    @staticmethod
    def visualize_comparison(test_results=None):
        """
        Kaydedilmiş history dosyalarını yükler ve karşılaştırır.
        Test sonuçlarını da dahil eder.
        """
        # History dosyalarını yükle
        history_cnn = ComparisonVisualizer.load_history('cnn_history.pkl')
        history_hybrid = ComparisonVisualizer.load_history('hybrid_history.pkl')
        history_transformer = ComparisonVisualizer.load_history('transformer_history.pkl')
        history_lstm = ComparisonVisualizer.load_history('lstm_history.pkl')

        # Eğitim kaybı (loss) karşılaştırması
        plt.figure()
        plt.plot(history_cnn['loss'], label="CNN - Training Loss")
        plt.plot(history_cnn['val_loss'], label="CNN - Validation Loss")
        plt.plot(history_hybrid['loss'], label="Hybrid - Training Loss")
        plt.plot(history_hybrid['val_loss'], label="Hybrid - Validation Loss")
        plt.plot(history_transformer['loss'], label="Transformer - Training Loss")
        plt.plot(history_transformer['val_loss'], label="Transformer - Validation Loss")
        plt.plot(history_lstm['loss'], label="LSTM - Training Loss")
        plt.plot(history_lstm['val_loss'], label="LSTM - Validation Loss")

        # Test kayıpları (loss) varsa ekle
        if test_results:
            plt.axhline(y=test_results['cnn']['loss'], color='r', linestyle='--', label="CNN - Test Loss")
            plt.axhline(y=test_results['hybrid']['loss'], color='g', linestyle='--', label="Hybrid - Test Loss")
            plt.axhline(y=test_results['transformer']['loss'], color='b', linestyle='--', label="Transformer - Test Loss")
            plt.axhline(y=test_results['lstm']['loss'], color='m', linestyle='--', label="LSTM - Test Loss")

        plt.title("Training vs Validation vs Test Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        # Eğitim doğruluğu (accuracy) karşılaştırması
        plt.figure()
        plt.plot(history_cnn['accuracy'], label="CNN - Training Accuracy")
        plt.plot(history_cnn['val_accuracy'], label="CNN - Validation Accuracy")
        plt.plot(history_hybrid['accuracy'], label="Hybrid - Training Accuracy")
        plt.plot(history_hybrid['val_accuracy'], label="Hybrid - Validation Accuracy")
        plt.plot(history_transformer['accuracy'], label="Transformer - Training Accuracy")
        plt.plot(history_transformer['val_accuracy'], label="Transformer - Validation Accuracy")
        plt.plot(history_lstm['accuracy'], label="LSTM - Training Accuracy")
        plt.plot(history_lstm['val_accuracy'], label="LSTM - Validation Accuracy")

        # Test doğrulukları (accuracy) varsa ekle
        if test_results:
            plt.axhline(y=test_results['cnn']['accuracy'], color='r', linestyle='--', label="CNN - Test Accuracy")
            plt.axhline(y=test_results['hybrid']['accuracy'], color='g', linestyle='--', label="Hybrid - Test Accuracy")
            plt.axhline(y=test_results['transformer']['accuracy'], color='b', linestyle='--', label="Transformer - Test Accuracy")
            plt.axhline(y=test_results['lstm']['accuracy'], color='m', linestyle='--', label="LSTM - Test Accuracy")

        plt.title("Training vs Validation vs Test Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
