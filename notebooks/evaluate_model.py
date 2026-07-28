"""
Model Evaluation Script
Author: Bibek
Date: July 28, 2026
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model_path, class_names):
    """Evaluate the trained model"""
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded from: {model_path}")
    print(f"📊 Model summary:")
    model.summary()
    return model

print("✅ Evaluation script loaded!")