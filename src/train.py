from src.load_data import get_data
from src.build_model import create_model

def train_and_save():
    X_train, y_train, X_test, y_test = get_data()
    model = create_model()
    model.fit(X_train, y_train, epochs=5, batch_size=32)
    model.save("model/digit_model.h5")

if __name__ == "__main__":
    train_and_save()
