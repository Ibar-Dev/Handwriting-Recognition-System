import tensorflow as tf
from datetime import datetime
from config import MODEL_DIR, EPOCHS, LEARNING_RATE
from .cnn_model import build_cnn_model

def train_model(train_data, val_data, model_type='cnn'):
    if model_type == 'cnn':
        model = build_cnn_model()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        patience=5, restore_best_weights=True
    )
    
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=[tensorboard_callback, early_stopping]
    )
    
    model.save(f"{MODEL_DIR}/{model_type}_model.h5")
    return history