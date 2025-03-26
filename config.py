import os

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')

# Parámetros del modelo
IMAGE_SIZE = (28, 28)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001

# Configuración de datasets
DATASET_CONFIG = {
    'emnist': {
        'name': 'emnist/letters',
        'num_classes': 62
    },
    'iam': {
        'path': os.path.join(DATA_DIR, 'iam_dataset'),
        'word_level': True
    }
}