# CV Pipeline Documentation

## Overview
The CV pipeline processes video frames and extracts hand gestures for recognition.

## Pipeline Steps

### 1. Frame Extraction
- Captures video frames at 30 FPS
- Resizes to 224×224 for MobileNetV2
- Converts BGR to RGB

### 2. Data Preprocessing
- **Normalization**: Scales pixels to [0, 1]
- **Augmentation**:
  - Rotation: ±20°
  - Width/Height Shift: ±20%
  - Zoom: ±20%
  - Horizontal Flip

### 3. Model Inference
- **MobileNetV2**: 99.9167% accuracy
- **LSTM**: Sequence modeling (dynamic gestures)

### 4. Post-Processing
- Confidence threshold filtering (0.7)
- History tracking (last 20 predictions)
- TTS output

## Technologies Used
| Component | Technology |
|-----------|------------|
| Video Processing | OpenCV 4.8.0 |
| Model Framework | TensorFlow 2.18.0 |
| Landmark Detection | MediaPipe 0.10.8 |
| UI Framework | Streamlit 1.29.0 |

## Performance
| Metric | Value |
|--------|-------|
| Inference Time | < 50ms per frame |
| Accuracy | 99.9167% |
| Processing Speed | 30 FPS |