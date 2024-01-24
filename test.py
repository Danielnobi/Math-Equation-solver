import cv2
import numpy as np
import tensorflow as tf

# Load the ResNet model with 19 classes
import cv2
import numpy as np

# Assuming model is already loaded
model = tf.keras.models.load_model('./model_files/modelX.h5')

def preprocess_image(img):
    img = cv2.resize(img, (28,28))
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    img = img.reshape(1, 28,28, 1)
    return img

def predict_symbol(img):
    img = preprocess_image(img)
    img = np.concatenate([img, img, img], axis=-1)
    prediction = model.predict(img)
    symbol_index = np.argmax(prediction)
    print(symbol_index)

    if symbol_index < 10:
        return str(symbol_index)
    elif symbol_index == 10:
        return "+"
    elif symbol_index == 11:
        return "-"
    elif symbol_index == 12:
        return "*"
    elif symbol_index == 13:
        return "/"
    elif symbol_index == 14:
        return "="
    elif symbol_index == 15:
        return "."
    elif symbol_index == 16:
        return "x"
    elif symbol_index == 17:
        return "y"
    elif symbol_index == 18:
        return "z"

def process_image(img_path):
    img = cv2.imread(f'./images/6.png', cv2.IMREAD_GRAYSCALE)
    img = ~img  # Invert the image

    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ctrs, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    equation = ''
    for c in cnt:
        x, y, w, h = cv2.boundingRect(c)
        im_crop = thresh[y:y + h + 10, x:x + w + 10]
        symbol = predict_symbol(im_crop)
        equation += symbol

    return equation

# Example usage:
image_path = 'path_to_your_equation_image.jpg'
detected_equation = process_image(image_path)
print("Detected Equation:", detected_equation)
