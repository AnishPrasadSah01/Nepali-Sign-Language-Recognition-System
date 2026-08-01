# Model Summary - NSL Recognition System

## MobileNetV2 (Baseline)

### Architecture
- **Base Model**: MobileNetV2 (ImageNet pre-trained)
- **Input Shape**: 224 × 224 × 3
- **Parameters**: 3,057,508 trainable
- **Classes**: 36 NSL Characters

### Training Details
- **Phase 1**: 15 epochs (Frozen Base)
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-Entropy
- **Data Augmentation**: Rotation, Shift, Zoom, Flip

### Results
- **Validation Accuracy**: 99.9167%
- **Validation Loss**: 0.0024
- **Best Epoch**: 14
- **Precision**: 99.92%
- **Recall**: 99.92%
- **F1-Score**: 99.92%

### Comparison
| Model | Accuracy |
|-------|----------|
| MobileNetV2 (Published) | 90.45% |
| **Our Model** | **99.9167%** |
| **Improvement** | **+9.47%** |

### Error Analysis
**Common Misclassifications:**
1. KA ↔ KHA (visually similar hand positions)
2. MA ↔ M_SHA (close hand orientation)
3. PA ↔ PHA (subtle finger differences)

**Root Causes:**
- Lighting variations
- Hand orientation differences
- Limited edge cases in training
- Absence of temporal smoothing