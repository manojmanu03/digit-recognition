import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from src.load_data import get_data

def predict_digit(index=0):
    _, _, X_test, y_test = get_data()
    model = load_model("model/digit_model.h5")
    pred = model.predict(X_test)
    predicted_class = np.argmax(pred[index])
    actual_class = np.argmax(y_test[index])
    print(f"Predicted: {predicted_class}, Actual: {actual_class}")
    plt.imshow(X_test[index].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_class}")
    plt.show()

if __name__ == "__main__":
    predict_digit()
