# ============================================================
# NSL Recognition System - Training Pipeline
# CV/NLP Engineer: Bibek
# Date: July 31, 2026
# ============================================================

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import numpy as np

# ============================================================
# 1. DATA PREPROCESSING
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# ============================================================
# 2. MODEL ARCHITECTURE
# ============================================================

def build_mobilenetv2_model(num_classes=36):
    """Build MobileNetV2 model with transfer learning"""
    
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling='avg'
    )
    base_model.trainable = False
    
    # Custom classification head
    model = models.Sequential([
        base_model,
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model

# ============================================================
# 3. TRAINING FUNCTION
# ============================================================

def train_model(model, train_generator, val_generator, epochs=15):
    """Train the model with checkpoints"""
    
    callbacks = [
        ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

# ============================================================
# 4. EVALUATION METRICS
# ============================================================

def evaluate_model(model, val_generator):
    """Evaluate model performance"""
    
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np
    
    val_generator.reset()
    predictions = model.predict(val_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_generator.classes[:len(predicted_classes)]
    
    # Classification Report
    class_names = list(val_generator.class_indices.keys())
    report = classification_report(true_classes, predicted_classes, 
                                   target_names=class_names, 
                                   digits=4)
    
    return report

# ============================================================
# 5. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("🚀 NSL Recognition System - Training Pipeline")
    print("="*60)
    print("🔹 CV/NLP Engineer: Bibek")
    print("🔹 Date: July 31, 2026")
