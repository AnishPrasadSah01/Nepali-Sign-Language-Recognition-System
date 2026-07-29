cat > docs/model_architecture.md << 'EOF'
# MobileNetV2 Architecture for NSL Recognition

## Model Summary

| Layer | Output Shape | Parameters |
|-------|--------------|------------|
| MobileNetV2 | 1280 | 2,257,984 |
| Dense 1 | 512 | 655,872 |
| BatchNormalization | 512 | 2,048 |
| Dropout (0.5) | 512 | 0 |
| Dense 2 | 256 | 131,328 |
| BatchNormalization | 256 | 1,024 |
| Dropout (0.3) | 256 | 0 |
| Dense 3 (Output) | 36 | 9,252 |
| **Total** | | **3,057,508** |

## Training Parameters

- Batch Size: 32
- Learning Rate: 0.001 (Phase 1)
- Optimizer: Adam
- Loss: Categorical Cross-Entropy
- Epochs: 15

## Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | 99.9167% |
| Precision | 99.92% |
| Recall | 99.92% |
| F1-Score | 99.92% |
EOF