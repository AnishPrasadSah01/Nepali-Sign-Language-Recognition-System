cat > notebooks/lstm_model.py << 'EOF'

# LSTM Model for Dynamic Gesture Recognition
# Author: Bibek


import tensorflow as tf
from tensorflow.keras import layers, models

def build_lstm_model(input_shape=(30, 63), num_classes=36):
    """
    LSTM model for dynamic gesture recognition
    Input: 30 frames × 63 landmarks (21 hand landmarks × 3 coordinates)
    """
    model = models.Sequential([
        layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.3),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

print("📊 LSTM Model Architecture")
print("="*60)
lstm_model = build_lstm_model()
lstm_model.summary()

print("\n✅ LSTM model ready for dynamic gesture recognition!")
EOF