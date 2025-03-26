import tensorflow as tf
import numpy as np
from config import MODEL_DIR, IMAGE_SIZE
from data_preprocessing.preprocess import preprocess_image

class HandwritingRecognizer:
    def __init__(self, model_path=f"{MODEL_DIR}/cnn_model.h5"):
        self.model = tf.keras.models.load_model(model_path)
        self.char_map = {i: chr(i + 65) for i in range(26)}  # A-Z
        self.char_map.update({i+26: chr(i + 97) for i in range(26)})  # a-z
        self.char_map.update({52+i: str(i) for i in range(10)})  # 0-9

    def predict_char(self, image):
        processed = preprocess_image(image)
        pred = self.model.predict(np.array([processed]))
        char_code = np.argmax(pred)
        return self.char_map.get(char_code, '?')