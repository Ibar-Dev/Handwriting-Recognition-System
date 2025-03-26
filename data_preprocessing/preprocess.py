import cv2
import numpy as np
from config import IMAGE_SIZE

def preprocess_image(img, is_binary=True):
    """Preprocesamiento básico para imágenes de escritura"""
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.resize(img, IMAGE_SIZE)
    
    if is_binary:
        img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
    
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=-1)

def augment_image(img):
    """Aumentación de datos en tiempo real"""
    import random
    from tensorflow.keras.preprocessing.image import apply_affine_transform
    
    # Rotación aleatoria
    if random.random() > 0.5:
        img = apply_affine_transform(
            img, theta=random.uniform(-15, 15)
        )
    
    # Desplazamiento aleatorio
    if random.random() > 0.5:
        img = apply_affine_transform(
            img,
            tx=random.uniform(-0.1, 0.1) * IMAGE_SIZE[0],
            ty=random.uniform(-0.1, 0.1) * IMAGE_SIZE[1]
        )
    
    return img