cat > notebooks/evaluate.py << 'EOF'

# Model Evaluation Script
# Author: Bibek


import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model_path, validation_generator):
    """Evaluate the trained model"""
    
    print("📊 Loading model...")
    model = tf.keras.models.load_model(model_path)
    
    print("🔄 Making predictions...")
    val_generator.reset()
    predictions = model.predict(val_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_generator.classes[:len(predicted_classes)]
    
    # Accuracy
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"\n📈 Overall Accuracy: {accuracy*100:.4f}%")
    
    # Classification Report
    class_names = list(val_generator.class_indices.keys())
    report = classification_report(true_classes, predicted_classes, 
                                   target_names=class_names, digits=4)
    print("\n📋 Classification Report:")
    print(report)
    
    # Confusion Matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm[:20, :20], annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names[:20], yticklabels=class_names[:20])
    plt.title('Confusion Matrix (First 20 Classes)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("✅ Confusion matrix saved!")
    
    return accuracy, report

print("✅ Evaluation script ready!")
EOF