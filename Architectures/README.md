We fastly iterate through 4 architectures by training and testing on only HG00152 Geuvadis dataset:

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Features</th>
      <th>MSE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>A</td>
      <td>Same as original Enformer</td>
      <td>0.30166</td>
    </tr>

    <!-- highlighted row B -->
    <tr style="background-color:#00FF33;">
      <td>B</td>
      <td>
        GELU swapped with SwiGLU, Layer normalization and gradient
        clipping used, 5-fold cross fold validation (contrary to fixed split)
      </td>
      <td>0.13774</td>
    </tr>

    <tr>
      <td>C</td>
      <td>
        Multi-Head Attention Pooling (4 Heads), replacing Linear
        Pooling to capture diverse regulatory regions, LayerNorm &amp;
        GradClipping, 5-fold CV
      </td>
      <td>1.38315</td>
    </tr>

    <tr>
      <td>D</td>
      <td>
        Inserts a 1D Convolutional Refinement block before pooling to
        mix local context, Swaps GELU with Mish (Self-regularized
        non-monotonic), 5-Fold CV.
      </td>
      <td>0.79096</td>
    </tr>
  </tbody>
</table>
