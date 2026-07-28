"""
NSL Recognition System - Training Pipeline
Author: Bibek
Date: July 28, 2026
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import numpy as np
import os

print("✅ Training Pipeline Loaded Successfully!")

def build_model(num_classes=36):
    """Build MobileNetV2 model"""
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    base_model.trainable = False
    
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

print("✅ Model function defined!")