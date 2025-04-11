from src.load_data import get_data
from tensorflow.keras.models import load_model

def evaluate_model():
    _, _, X_test, y_test = get_data()
    model = load_model("model/digit_model.h5")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    evaluate_model()
