# Model Architecture - MobileNetV2

## Model Summary

| Layer | Output Shape | Parameters |
|-------|--------------|------------|
| MobileNetV2 | (None, 1280) | 2,257,984 |
| Dense 1 | (None, 512) | 655,872 |
| BatchNormalization | (None, 512) | 2,048 |
| Dropout (0.5) | (None, 512) | 0 |
| Dense 2 | (None, 256) | 131,328 |
| BatchNormalization | (None, 256) | 1,024 |
| Dropout (0.3) | (None, 256) | 0 |
| Dense 3 (Output) | (None, 36) | 9,252 |
| **Total** | | **3,057,508** |

## Training Configuration

- **Optimizer:** Adam
- **Learning Rate (Phase 1):** 0.001
- **Learning Rate (Phase 2):** 0.0001
- **Loss Function:** Categorical Cross-Entropy
- **Batch Size:** 32
- **Epochs (Phase 1):** 15
- **Epochs (Phase 2):** 15