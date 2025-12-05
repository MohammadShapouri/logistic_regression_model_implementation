# Multi-Class Logistic Regression (Softmax) for MNIST: A Deep Dive

This project implements a **Multi-Class Logistic Regression** model from scratch using `numpy`. Unlike the binary version which asks "Is this a 6?", this model asks: **"Which digit from 0 to 9 is this?"**

This document details the mathematical theory, code implementation, and the dimensional flow of data through the network.

---

## 1. Theoretical Foundation

### 1.1 The Linear Model (The Scores)
In Binary Regression, we calculated a single score ($z$) to denote "positiveness". In Multi-Class Regression, we must generate **10 distinct scores**, one for each digit (0 through 9).

The equation looks identical to the binary case, but the **dimensions** are significantly different:
$$ Z = X \cdot W + b $$

Here is the detailed breakdown of how this math translates directly into the code structure:

#### A. The Input Matrix ($X$) - `batch_images`
*   **Concept:** The input data remains the same. Flattened images.
*   **Shape:** `(N, 784)`.
    *   $N$: Batch size (e.g., 256 images).
    *   $784$: The number of pixels (features).

#### B. The Weight Matrix ($W$) - `self.weight_matrix`
*   **Concept:** Instead of one template, we now have **10 Templates**.
*   **Structure:**
    *   Column 0: The weights that recognize the number "0".
    *   Column 1: The weights that recognize the number "1".
    *   ...
    *   Column 9: The weights that recognize the number "9".
*   **Shape:** `(784, 10)`.
    *   Rows: 784 pixels.
    *   Columns: 10 Classes.
*   **Interpretation:** `W[55, 3]` is the weight connecting Pixel #55 to the score for Digit #3.

#### C. The Dot Product ($X \cdot W$)
*   **The Operation:** `batch_images.dot(self.weight_matrix)`
*   **Mechanism:** Matrix multiplication.
    *   For every image (row in $X$), we calculate the dot product against **every column** in $W$.
*   **Dimensional Result:** `(N, 784) \cdot (784, 10) \rightarrow (N, 10)`.
    *   We now have $N$ rows (images).
    *   Each row has 10 numbers (raw scores).

#### D. The Bias ($b$) - `self.bias`
*   **Concept:** We need a separate threshold (bias) for every digit.
    *   Example: Writing a "1" requires very little ink, so its base probability might differ from an "8".
*   **Shape:** `(1, 10)`. A row vector containing 10 separate bias values.
*   **Broadcasting:** When adding `(N, 10) + (1, 10)`, numpy adds the bias row to every single row in the scores matrix.

#### E. The Logits ($Z$) - `logits`
*   **Shape:** `(N, 10)`.
*   **Example Row:** `[-10.5, 4.2, -2.1, 15.6, ...]`
    *   This row implies the model thinks the image is likely a "3" (score 15.6) and definitely not a "0" (score -10.5).

---

### 1.2 The Activation Function (Softmax)
We have 10 raw scores (logits). We cannot use Sigmoid because Sigmoid treats each score independently (it could output 0.9 for class 0 *and* 0.8 for class 1). We need a probability distribution that **sums to 1.0** across the 10 classes.

We use the **Softmax Function**:
$$ \text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{10} e^{z_j}} $$

#### A. Numerical Stability (The "Safe" Shift)
*   **Problem:** $e^{1000}$ causes a computer overflow (returns `inf`).
*   **Solution:** `safe_raw_model_data = raw_model_data - np.max(...)`
*   **Math:** Softmax is shift-invariant. $\frac{e^{z_i}}{e^{z_{total}}} = \frac{e^{z_i - C}}{e^{z_{total} - C}}$.
*   **Effect:** By subtracting the max value in the row, the largest number becomes 0 (and $e^0 = 1$), preventing overflow.

#### B. The Exponentiation
*   **Code:** `exp = np.exp(safe_raw_model_data)`
*   **Operation:** Converts scores to positive numbers. Negative scores become small positive fractions ($0 < x < 1$), positive scores become large numbers.

#### C. Normalization
*   **Code:** `exp / np.sum(exp, axis=1, keepdims=True)`
*   **Mechanism:** We sum the row to find the "Total Energy" of the prediction. We divide each element by this total.
*   **Result:** A row `[0.1, 0.05, 0.8, 0.05 ...]` which sums to exactly 1.0.

---

### 1.3 The Loss Function (Cross-Entropy)
We need to measure the error for 10 classes. We use **Categorical Cross-Entropy**.

$$ J = - \sum_{k=1}^{10} y_k \log(\hat{y}_k) $$

#### A. One-Hot Encoding
*   **Concept:** The label is a single number, e.g., `3`. The output is a vector of 10 probabilities. We must convert the label `3` into a format matching the output.
*   **Transformation:** `3` $\to$ `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`.
*   **Code:** `_generate_one_hot_label_data` uses fancy indexing `one_hot[rows, labels] = 1`.

#### B. The Selection Mechanism
*   **Code:** `one_hot_labels * np.log(probabilities + eps)`
*   **Logic:**
    *   The `one_hot_labels` vector has 0s everywhere except the target class.
    *   When we multiply, the error contribution from all incorrect classes is multiplied by 0 (ignored).
    *   We **only** calculate the log loss of the probability assigned to the *correct* class.
*   **Batch Mean:** `np.sum(...) / batch_size` averages the error across all images.

---

### 1.4 Regularization (The Constraint)
We apply L2 Regularization to the Weight Matrix to prevent overfitting.

$$ J_{reg} = \frac{\lambda}{2N} \sum_{i} \sum_{j} W_{ij}^2 $$

*   **Matrix Math:** `np.sum(self.weight_matrix * self.weight_matrix)`.
    *   This squares every element in the `(784, 10)` matrix and sums them up.
*   **Purpose:** It forces the model to distribute "importance" across many pixels rather than relying on a few pixels having massive weights.

---

### 1.5 Gradient Descent (The Learning)
We update $W$ and $b$ to minimize Loss.

#### A. The Gradient Calculation
The derivative of the Cross-Entropy loss combined with Softmax is surprisingly simple:
$$ \frac{\partial J}{\partial Z} = \text{Probabilities} - \text{Labels (OneHot)} $$

*   **Code:** `dlogits = (probabilities - batch_one_hot_labels) / number_of_batch_items`
*   **Interpretation:**
    *   If True Label is Class 3 (OneHot `[...0, 1, 0...]`)
    *   And Prediction is Class 3 (Prob `[...0, 0.99, 0...]`)
    *   Diff is `-0.01`. Gradient is small. No major change needed.
    *   If Prediction is Class 3 (Prob `[...0, 0.10, 0...]`)
    *   Diff is `-0.90`. Gradient is large negative. Push weights to increase this score!

#### B. Backpropagation
1.  **Weights (`dW`):** `batch_images.T.dot(dlogits)`.
    *   Shapes: `(784, N) · (N, 10) = (784, 10)`.
    *   This calculates how to adjust every pixel weight for every class.
2.  **Bias (`db`):** `np.sum(dlogits, axis=0)`.
    *   Shapes: Sum `(N, 10)` vertically $\to$ `(1, 10)`.

#### C. Regularization Update
*   **Code:** `dW += (lambda / N) * W`.
*   **Effect:** Weight Decay. We shrink the weights slightly at every step.

---

## 2. Dimensional Analysis Example

**Scenario:** Batch size 256. 10 Classes.

| Variable | Code Name | Matrix Shape | Description |
| :--- | :--- | :--- | :--- |
| **Input Batch** | `batch_images` | `(256, 784)` | 256 images, flattened. |
| **Weights** | `self.weight_matrix` | `(784, 10)` | 10 filters, each 784 pixels long. |
| **Bias** | `self.bias` | `(1, 10)` | 10 separate bias terms. |
| **Logits** | `logits` | `(256, 10)` | The raw scores. 256 images x 10 scores. |
| **Probabilities** | `probabilities` | `(256, 10)` | Sum of each row is 1.0. |
| **One Hot Labels**| `batch_one_hot` | `(256, 10)` | All zeros, except one '1' per row. |
| **Gradient** | `dlogits` | `(256, 10)` | The error signal. |
| **Weight Grad** | `dW` | `(784, 10)` | Update for the template matrix. |

---
