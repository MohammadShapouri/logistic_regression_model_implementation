# Binary Logistic Regression for MNIST: A Deep Dive

This project implements a **Binary Logistic Regression** model from scratch using `numpy`. It treats the MNIST digit classification problem as a binary task: **"Is this image a number 6, or is it not?"**

This document details the mathematical theory, code implementation, and the dimensional flow of data through the network.

---

## 1. Theoretical Foundation

Binary Logistic Regression is a supervised learning algorithm used to predict the probability that an input instance belongs to a specific class (Class 1 vs. Class 0).

### 1.1 The Linear Model (The Score)
The fundamental building block of the network is the linear equation. In code, this corresponds to the "Forward Propagation" step. We are calculating a raw score ($z$) which represents the aggregate evidence that an image is a "6".

The equation is:
$$ z = X \cdot W + b $$

Here is the detailed breakdown of how this math translates directly into the code structure:

#### A. The Input Matrix ($X$) - `batch_images`
*   **Concept:** The model cannot "see" a 2D grid. It sees a list of values.
*   **In Code:** We flatten the $28 \times 28$ image into a single row vector of 784 pixels.
*   **Batching:** To utilize the speed of linear algebra libraries (like BLAS/LAPACK under numpy's hood), we don't process one image at a time. We process a batch of $N$ images.
*   **Shape:** `(N, 784)`.
    *   Row $i$ represents the $i$-th image in the batch.
    *   Column $j$ represents the $j$-th pixel (feature) of that image.

#### B. The Weight Matrix ($W$) - `self.weight_matrix`
*   **Concept:** The Weights act as a **"Template"** or **"Filter"**.
*   **Interpretation:**
    *   **Positive Weight ($w_j > 0$):** Means this specific pixel is usually *black* (active) for the number 6. (e.g., pixels in the bottom loop of the 6).
    *   **Negative Weight ($w_j < 0$):** Means this pixel is usually *white* (inactive) for the number 6. (e.g., pixels in the top right corner, which are empty for a 6 but filled for a 9 or 7).
    *   **Zero Weight ($w_j \approx 0$):** This pixel is irrelevant (e.g., background corners).
*   **Shape:** `(784, 1)`. It is a column vector matching the feature count.

#### C. The Dot Product ($X \cdot W$)
*   **The Operation:** `batch_images.dot(self.weight_matrix)`
*   **Mechanism:** This performs a weighted sum of the pixels for every image in the batch simultaneously.
    $$ \sum_{j=1}^{784} (\text{Pixel}_j \times \text{Weight}_j) $$
*   **Result:** If the image pixels align with the "Template" (positive pixels hit positive weights), the sum is a high number. If they mismatch (active pixels hit negative weights), the sum is low.
*   **Dimensional Result:** `(N, 784) \cdot (784, 1) \rightarrow (N, 1)`. We now have a single raw score for every image in the batch.

#### D. The Bias ($b$) - `self.bias`
*   **Concept:** The Bias is the **"Threshold"**. It represents the inherent prior probability of being a "6" regardless of the pixel data.
*   **Geometric View:** Without a bias, the decision boundary (hyperplane) is forced to pass through the origin $(0,0)$. The bias allows the separating line to shift left or right, fitting the data better.
*   **Broadcasting in Code:**
    *   The bias is a single scalar `(1, 1)`.
    *   The result of the dot product is `(N, 1)`.
    *   When we run `.dot(...) + self.bias`, Python performs **Broadcasting**. It automatically expands the single bias value to a vector of size `(N, 1)` and adds it to every single image score in the batch.

#### E. The Logit ($z$) - `logits`
*   **Definition:** The final output of the linear layer.
*   **Range:** $(-\infty, +\infty)$.
*   **Meaning:**
    *   $z > 0$: The model leans towards "It is a 6".
    *   $z < 0$: The model leans towards "It is NOT a 6".
    *   $z = 0$: The model is undecided (on the decision boundary).

### 1.2 The Activation Function (The Probability)
The linear score $z$ (logits) can be any real number, such as $45.2$ or $-9999$. This is not useful for a probability estimate. To convert $z$ into a standardized value strictly between 0 and 1, we use the **Sigmoid Function**.

The equation is:
$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

In the code, this is handled by the `_sigmoid` method. Here is the deep dive into its mechanics:

#### A. The "Squashing" Effect
The primary purpose of Sigmoid is to map the infinite number line onto a finite scale $[0, 1]$.
*   **High Confidence Positive:** If $z$ is large (e.g., $z=10$), $e^{-10}$ is tiny ($\approx 0.000045$). Result: $\frac{1}{1.000045} \approx 0.9999$.
*   **High Confidence Negative:** If $z$ is large negative (e.g., $z=-10$), $e^{-(-10)}$ is huge ($22026$). Result: $\frac{1}{22027} \approx 0.000045$.
*   **The Uncertainty Point:** If $z=0$, $e^{0}=1$. Result: $\frac{1}{1+1} = 0.5$. This is the decision boundary.

#### B. Vectorized Implementation (`np.exp`)
We do not use a loop to calculate this for every image.
*   **Code:** `return 1 / (1 + np.exp(-z))`
*   **Numpy Mechanics:** `np.exp` is an **element-wise** operation.
    *   **Input:** The `logits` array of shape `(N, 1)`.
    *   **Operation:** Numpy calculates $e^x$ for every single item in the array simultaneously, utilizing CPU vector instructions (SIMD).
    *   **Output:** An array of `probs` with the exact same shape `(N, 1)`.

#### C. Why Sigmoid and not a "Step Function"?
You might ask: *Why not just say "If $z > 0$ output 1, else output 0"?*
*   **The Problem:** A hard "Step Function" is flat everywhere and has a vertical jump at 0. Its derivative is 0 everywhere (except at the jump where it's undefined).
*   **The Solution:** To train a model using Gradient Descent, we need a **Slope** (Gradient). The Sigmoid function is smooth and differentiable. It provides a gradient signal: "You are close to 1, but not quite there, keep pushing."

### 1.3 The Loss Function (The Error)
To train the model, we need a mathematical quantification of "how wrong" the model is. We use **Binary Cross-Entropy Loss** (also known as Log Loss). This function penalizes confident wrong answers heavily.

The equation is:
$$ J = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] $$

#### A. The Logic of Logarithms
Why do we use logs?
*   **Inputs:** Probabilities are between 0 and 1.
*   **Log Behavior:** $\log(1) = 0$ and $\log(0) \to -\infty$.
*   **The Penalty:**
    *   If the Label is 1 ($y=1$), we look at $\log(\hat{y})$. If prediction $\hat{y}$ is 1.0, cost is 0. If $\hat{y}$ is 0.01, $\log(0.01) = -4.6$, so cost is 4.6 (high penalty).
    *   The negative sign at the front flips the negative log results into positive cost values.

#### B. Numerical Stability (The Epsilon)
In code, a pure mathematical implementation is dangerous.
*   **The Crash:** If the model is "dead wrong" and predicts exactly 0.0 for a label of 1, we try to calculate $\log(0)$. This returns `-inf` and causes the math to break.
*   **The Fix:** We add a tiny "epsilon" ($\epsilon = 1 \times 10^{-12}$).
*   **In Code:** `np.log(probs + eps)`
*   This ensures the argument to log is at minimum $0.000000000001$, capping the maximum penalty instead of letting it go to infinity.

#### C. Vectorized Calculation
The implementation in `_binary_cross_entropy_loss` does not loop through samples.
*   **Code:** `-(labels * np.log(probs) + (1-labels) * np.log(1-probs))`
*   **Masking:**
    *   `labels` is a vector of 1s and 0s.
    *   When `labels` is 1, the right side of the plus sign `(1-labels)` becomes 0, effectively deleting the "Case 0" term.
    *   When `labels` is 0, the left side becomes 0, deleting the "Case 1" term.
*   **Averaging:** Finally, we take `np.mean(...)`, which corresponds to the $\frac{1}{N} \sum$ part of the equation, giving us the average error over the batch.

### 1.4 Regularization (The Constraint)
If we give the model total freedom, it might construct a "Template" (Weight matrix) with massive values to force the math to work for difficult, noisy images. This leads to **Overfitting**—the model memorizes the training data but fails on new data.

To solve this, we add a penalty for having large weights: **L2 Regularization**.

$$ J_{final} = J_{CE} + \frac{\lambda}{2N} \sum_{j=1}^{784} W_j^2 $$

#### A. The L2 Norm
*   **Concept:** We sum the square of every weight in the matrix.
*   **Effect:** If a weight is 10, its penalty is 100. If it is 0.1, its penalty is 0.01. This strongly discourages outliers (large weights) and encourages a "smoother" distribution of weights.

#### B. The Lambda ($\lambda$) Parameter
*   **Hyperparameter:** This is a knob we turn during initialization.
*   **High Lambda:** Heavy restriction. Weights will be very small. Risk of Underfitting.
*   **Low Lambda:** Light restriction. Weights can grow large. Risk of Overfitting.

#### C. Code Implementation
*   **Calculation:** `l2_cost = (self.lambda_ / (2 * m)) * np.sum(np.square(self.weight_matrix))`
*   **Integration:** This value is added to the `loss` solely for tracking purposes.
*   **Note on Bias:** We usually *do not* regularize the bias term $b$. The bias merely shifts the threshold and doesn't contribute to the "complexity" or "noisiness" of the decision boundary in the same way weights do.

### 1.5 Gradient Descent (The Learning)
Now that we have a Loss ($J$), we need to minimize it. We use **Gradient Descent**, which is an iterative optimization algorithm.

$$ W_{new} = W_{old} - \alpha \cdot \frac{\partial J}{\partial W} $$

#### A. The Gradient ($\nabla J$)
*   **Definition:** The gradient is a vector of partial derivatives. It represents the **slope** of the error landscape.
*   **Direction:** The gradient vector points in the direction of *steepest ascent* (where error grows fastest).
*   **Calculus in Code:** In our `train_model` method, we compute `dW`.
    *   If `dW[i]` is positive, it means increasing Weight $i$ increases Error. So we must *decrease* Weight $i$.
    *   If `dW[i]` is negative, it means increasing Weight $i$ decreases Error. So we must *increase* Weight $i$.

#### B. The Learning Rate ($\alpha$)
*   **Variable:** `self.learning_rate`.
*   **Role:** This controls the size of the step we take.
    *   **Too Small:** The model learns effectively but takes forever to converge.
    *   **Too Large:** The model steps *over* the valley bottom and creates unstable oscillations (divergence).

#### C. The Update Step
*   **Code:** `self.weight_matrix -= self.learning_rate * dW`
*   **Mechanism:**
    1.  Calculate direction to increase error (`dW`).
    2.  Flip direction (Multiply by -1, represented by the `-=` operator).
    3.  Scale the step by Learning Rate.
    4.  Apply change to the existing memory (`self.weight_matrix`).

---

## 2. Code Breakdown & Method Analysis

### Class: `BinaryLogisticRegression`

#### `__init__`
Initializes the hyperparameters.
*   **`learning_rate`**: The step size for gradient descent. Too high = overshoot; too low = too slow.
*   **`batch_size`**: How many images we process at once before updating weights. This makes learning smoother and computationally faster (Matrix operations).
*   **`lambda_`**: The regularization strength.

#### `_sigmoid(z)`
Applies the math formula $\frac{1}{1 + e^{-z}}$.
*   *Note:* `np.exp` handles arrays element-wise, so we can pass a whole batch of scores at once.

#### `_generate_binary_labels`
The MNIST dataset comes with labels 0 through 9. This method performs a **One-vs-Rest** transformation:
*   Input Label array: `[0, 6, 9, 6]`
*   Condition `labels == 6`: `[False, True, False, True]`
*   Cast to float: `[0.0, 1.0, 0.0, 1.0]`
*   Reshape: Ensures it is a column vector `(N, 1)` to match matrix math requirements.

#### `_normalize_image_data`
Raw pixel values range from 0 (white) to 255 (black).
1.  **Reshape:** Flattens `(N, 28, 28)` to `(N, 784)`.
2.  **Scale:** Divides by 255.0.
    *   *Why?* Gradient descent converges much faster when features are on a small scale (0 to 1) rather than a large scale (0 to 255).

#### `_binary_cross_entropy_loss`
Computes the cost.
*   **Epsilon (`eps = 1e-12`):** A tiny number added inside the `log`.
    *   *Why?* $\log(0)$ is $-\infty$, which crashes the program. Epsilon ensures we never take the log of exactly zero.

#### `train_model` (The Engine)
This method contains the training loop.
1.  **Weight Initialization:** `np.random.normal`. We start with small random numbers. If we started with all zeros, the model might get stuck in symmetry (though less critical in logistic regression than deep neural nets, it's still best practice).
2.  **Shuffling:** `np.random.permutation`. We shuffle data every epoch so the model doesn't learn the order of the data (e.g., if all 6s were at the end, the model would learn weird temporal patterns).

---

## 3. Deep Dive: The Training Step (Math to Code)

Inside the training loop, the following 7 steps happen for every batch. This is the heart of the algorithm.

### Step 1: Forward Propagation (Linear)
```python
logits = batch_images.dot(self.weight_matrix) + self.bias
```
*   **Math:** $Z = X \cdot W + b$
*   **Explanation:** We multiply the batch of images by the weights.
*   **Shape:** `(Batch, 784) · (784, 1)` $\to$ `(Batch, 1)`. The bias is added to every row via broadcasting.

### Step 2: Forward Propagation (Activation)
```python
probs = self._sigmoid(logits)
```
*   **Math:** $A = \sigma(Z)$
*   **Explanation:** This transforms raw scores into probabilities.

### Step 3: Gradient Calculation (The Derivative of Loss)
```python
dlogits = (probs - batch_labels) / batch_images.shape[0]
```
*   **Math:** $\frac{\partial J}{\partial Z} = \frac{1}{N}(\hat{y} - y)$
*   **Explanation:** This is a surprisingly elegant result of calculus. The derivative of the Binary Cross Entropy loss function with respect to the linear input $z$ is simply the **Prediction minus the Target**.
    *   If prediction is 1.0 and target is 1.0, difference is 0 (No change needed).
    *   If prediction is 0.9 and target is 0.0, difference is 0.9 (Push prediction down!).
    *   We divide by batch size `N` to get the *average* gradient.

### Step 4: Backpropagation (Weights)
```python
dW = batch_images.T.dot(dlogits)
```
*   **Math:** $\frac{\partial J}{\partial W} = X^T \cdot \frac{\partial J}{\partial Z}$
*   **Explanation:** By Chain Rule: "How much loss changes w.r.t weights" = "Input values" $\times$ "Error at output".
*   **Why Transpose?** `batch_images` is `(N, 784)`. `dlogits` is `(N, 1)`. To multiply them and get a result of size `(784, 1)` (the shape of our weights), we must transpose images to `(784, N)`.

### Step 5: Backpropagation (Bias)
```python
db = np.sum(dlogits, axis=0, keepdims=True)
```
*   **Math:** $\frac{\partial J}{\partial b} = \sum \frac{\partial J}{\partial Z}$
*   **Explanation:** The bias affects every sample equally. So, we simply sum up the error gradients from all samples in the batch to find out how to adjust the bias.

### Step 6: Regularization Gradient
```python
if self.lambda_ > 0:
    dW += (self.lambda_ / batch_images.shape[0]) * self.weight_matrix
```
*   **Math:** $\frac{\partial}{\partial W} (\frac{\lambda}{2} W^2) = \lambda W$
*   **Explanation:** We add a small fraction of the current weight to the gradient. This effectively pushes the weight towards zero during the update step. This is often called "Weight Decay".

### Step 7: Parameter Update
```python
self.weight_matrix -= self.learning_rate * dW
self.bias -= self.learning_rate * db
```
*   **Action:** We move the weights in the direction that reduces loss (opposite to the gradient).

---

## 4. Dimensional Analysis Example

It is crucial to understand how the shapes of the matrices change to ensure the math works.

**Scenario:**
*   We are processing 1 mini-batch.
*   **Batch Size:** 256 images.
*   **Input Dimensions:** $28 \times 28 = 784$ pixels.

| Variable | Code Name | Matrix Shape | Description |
| :--- | :--- | :--- | :--- |
| **Input Batch** | `batch_images` | `(256, 784)` | 256 rows (images), 784 columns (pixels). |
| **Weights** | `self.weight_matrix` | `(784, 1)` | 784 weights, 1 for each pixel. |
| **Bias** | `self.bias` | `(1, 1)` | A single scalar value. |
| **Logits** | `logits` | `(256, 1)` | Result of Dot Product. `(256,784) x (784,1)`. |
| **Probabilities** | `probs` | `(256, 1)` | Probabilities (0.0 to 1.0) for the 256 images. |
| **Labels** | `batch_labels` | `(256, 1)` | True values (1.0 or 0.0) for the 256 images. |
| **Error Gradient** | `dlogits` | `(256, 1)` | The raw error signal for each image. |
| **Weight Grad** | `dW` | `(784, 1)` | Calculated by `(784, 256) x (256, 1)`. Tells us how to update each of the 784 weights. |
| **Bias Grad** | `db` | `(1, 1)` | The sum of all 256 errors. |

### Visualizing the Transformation
1.  **Start:** 256 Images.
2.  **Multiply:** Each image is compared against the "Template" (Weights) for the number 6.
3.  **Result:** 256 Scores.
4.  **Compare:** 256 Scores vs 256 Answers.
5.  **Learn:** The discrepancies are mapped back to the 784 pixel positions to adjust the template for the next round.