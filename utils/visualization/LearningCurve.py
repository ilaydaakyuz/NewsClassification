import matplotlib.pyplot as plt

class LearningCurve:
    def plot_learning_curves(self, history):
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.show()