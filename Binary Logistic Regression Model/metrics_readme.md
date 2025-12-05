# BinaryMetrics: Evaluation for Two-Class Problems

This class provides a standalone implementation of performance metrics specifically for **Binary Classification** (e.g., "Is this a 6 or not?"). It removes dependency on `scikit-learn` by using vectorized `numpy` operations.

---

## 1. Why do we need this?

In Binary classification, especially with **Imbalanced Datasets**, Accuracy is dangerous.

*   **The Scenario:** You are building a detector for the number "6".
*   **The Data:** In MNIST, only ~10% of images are "6". The other 90% are "Not 6".
*   **The Lazy Model:** If a model predicts **"Not 6" (0)** for every single image, it achieves **90% Accuracy**.
*   **The Problem:** The model looks successful, but it is actually useless because it never caught a single "6".

We need **Precision** and **Recall** to see through this illusion.

---

## 2. Core Concepts: The Confusion Matrix

In a binary world, we usually designate one class as **Positive (1)** (the thing we are looking for, e.g., the digit 6) and the other as **Negative (0)**.

| Term | Code Logic | Meaning |
| :--- | :--- | :--- |
| **TP (True Positive)** | `Pred == 1` AND `True == 1` | The model correctly spotted a 6. |
| **TN (True Negative)** | `Pred == 0` AND `True == 0` | The model correctly ignored a non-6. |
| **FP (False Positive)** | `Pred == 1` AND `True == 0` | **"False Alarm"**: The model thought it was a 6, but it wasn't. |
| **FN (False Negative)** | `Pred == 0` AND `True == 1` | **"Missed detection"**: It was a 6, but the model missed it. |

---

## 3. The Metrics Explained

### 3.1 Accuracy
Answers: **"Overall, how often is the model correct?"**

$$ \text{Accuracy} = \frac{TP + TN}{\text{Total Samples}} $$

*   **In Code:** `np.mean(y_true == y_pred)`
*   **Limitation:** Heavily skewed by the majority class (the "Not 6"s).

### 3.2 Precision (The "Quality" Metric)
Answers: **"When the model claims it found a 6, how likely is it to actually be a 6?"**

$$ \text{Precision} = \frac{TP}{TP + FP} $$

*   **High Precision:** The model is careful. It doesn't make false accusations.
*   **Low Precision:** The model is noisy. It flags many non-6s as 6s.

### 3.3 Recall (The "Quantity" Metric)
Answers: **"Out of all the 6s that exist in the world, what percentage did the model find?"**

$$ \text{Recall} = \frac{TP}{TP + FN} $$

*   **High Recall:** The model is a wide net. It catches almost every 6.
*   **Low Recall:** The model is blind to many 6s.

### 3.4 F1-Score (The Harmonic Mean)
This combines Precision and Recall into a single number. It is the best metric for comparing models on imbalanced data.

$$ F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$

---

## 4. Code Implementation Deep Dive

The class uses `numpy` boolean masking to calculate these counts without loops. We assume the "Positive" class is labeled `1` and the "Negative" class is `0`.

### Step 1: Create Masks
We create boolean arrays representing the "Positive" (Class 1) predictions and realities.
```python
# Did the model predict Positive (1)?
pred_pos = (y_pred == 1)
# Was the actual label Positive (1)?
true_pos = (y_true == 1)
```

### Step 2: Bitwise Logic
We use `&` (AND) and `~` (NOT) to isolate the quadrants of the confusion matrix.
```python
# True Positive: Prediction is 1 AND Truth is 1
tp = np.sum(pred_pos & true_pos)

# False Positive: Prediction is 1 BUT Truth is NOT 1 (0)
fp = np.sum(pred_pos & (~true_pos))

# False Negative: Prediction is NOT 1 (0) BUT Truth is 1
fn = np.sum((~pred_pos) & true_pos)
```

### Step 3: Safety Checks
We must handle cases where the denominator is zero (e.g., if the model never predicts "1", `TP + FP` is 0).
```python
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
```

---

## 5. Usage Example

You can use the static methods directly without creating an instance of the class.

```python
from binary_metrics import BinaryMetrics
import numpy as np

# Mock Data (1 = Is a '6', 0 = Is NOT a '6')
y_true = np.array([0, 1, 1, 0, 1, 0])
y_pred = np.array([0, 1, 0, 0, 1, 1])

# 1. Get Accuracy
acc = BinaryMetrics.accuracy(y_true, y_pred)
print(f"Accuracy: {acc:.2f}") 
# Output: 0.67 (4/6 correct)

# 2. Get Detailed Report
BinaryMetrics.print_binary_report(y_true, y_pred)
```

**Output:**
```text
--- Binary Classification Report ---
Precision (Positive Class): 0.6667
Recall (Positive Class):    0.6667
F1-Score:                   0.6667
```