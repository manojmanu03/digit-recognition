import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the model
model = load_model("model/digit_model.h5")

# Load and preprocess image (make sure the image is 28x28 and grayscale)
img_path = "input/WhatsApp Image 2025-04-11 at 22.25.20_8b4bdb2d.jpg"  # path to your test image
img = Image.open(img_path).convert("L").resize((28, 28))
img_array = np.array(img)
img_array = 255 - img_array  # Invert if needed
img_array = img_array / 255.0  # Normalize
img_array = img_array.reshape(1, 28, 28, 1)

# Predict
prediction = model.predict(img_array)
predicted_digit = np.argmax(prediction)
print(f"Predicted digit: {predicted_digit}")
