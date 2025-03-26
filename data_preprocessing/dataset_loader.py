import tensorflow_datasets as tfds
from config import DATASET_CONFIG, BATCH_SIZE
from .preprocess import preprocess_image, augment_image

def load_emnist():
    """Cargar dataset EMNIST"""
    (train, test), info = tfds.load(
        DATASET_CONFIG['emnist']['name'],
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )
    
    def preprocess_emnist(image, label):
        image = tf.image.rot90(image, k=3)  # Corregir rotación EMNIST
        image = preprocess_image(image.numpy())
        return image, label
    
    train = train.map(
        lambda x, y: tf.py_function(
            preprocess_emnist, [x, y], [tf.float32, tf.int32]
        )
    ).batch(BATCH_SIZE)
    
    test = test.map(
        lambda x, y: tf.py_function(
            preprocess_emnist, [x, y], [tf.float32, tf.int32]
        )
    ).batch(BATCH_SIZE)
    
    return train, test, info