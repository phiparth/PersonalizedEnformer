We fastly iterate through 4 architectures by training and testing on only HG00152 Geuvadis dataset:

| Model | Features | MSE |
|---|---|---|
| A | Same as original Enformer | 0.30166 |
| B | GELU swapped with SwiGLU, Layer normalization and gradient clipping used, 5-fold cross fold validation (contrary to fixed split) | 0.13774 |
| C | Multi-Head Attention Pooling (4 Heads), replacing Linear Pooling to capture diverse regulatory regions, LayerNorm & GradClipping, 5-fold CV | 1.38315 |
| D | Inserts a 1D Convolutional Refinement block before pooling to mix local context, Swaps GELU with Mish (Self-regularized non-monotonic), 5-Fold CV. | 0.79096 |
