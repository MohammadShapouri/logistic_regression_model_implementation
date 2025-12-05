# ClassificationMetrics: Dependency-Free Evaluation

This class provides a standalone implementation of standard classification performance metrics. It removes the dependency on `scikit-learn`, calculating **Accuracy, Precision, Recall, and F1-Score** from scratch using vectorized `numpy` operations.

---

## 1. Why do we need this?

In Multi-Class classification (like MNIST), simple **Accuracy** is often misleading.
*   **Scenario:** If you have 100 images, and 90 are "Class A" and 10 are "Class B".
*   **Lazy Model:** If a model blindly guesses "Class A" for everything, it gets **90% Accuracy**.
*   **Reality:** It failed completely to detect "Class B".

To catch this, we calculate metrics **per class** (One-vs-Rest).

---

## 2. Core Concepts: The Confusion Matrix

For a specific class (e.g., the number **5**), we categorize every prediction into one of three buckets:

| Term | Code Logic | Meaning |
| :--- | :--- | :--- |
| **TP (True Positive)** | `Pred == 5` AND `True == 5` | The model correctly spotted a 5. |
| **FP (False Positive)** | `Pred == 5` AND `True != 5` | The model *thought* it was a 5, but it was actually a 3 (false alarm). |
| **FN (False Negative)** | `Pred != 5` AND `True == 5` | The image *was* a 5, but the model missed it (predicted 6). |

---

## 3. The Metrics Explained

### 3.1 Accuracy
The simplest metric. It answers: **"What fraction of predictions were correct?"**

$$ \text{Accuracy} = \frac{\text{Total Correct Predictions}}{\text{Total Samples}} $$

*   **In Code:** `np.mean(y_true == y_pred)`
*   **Mechanism:** Creates a boolean array (True for matches, False for mismatches) and calculates the average (where True=1, False=0).

### 3.2 Precision (The "Trust" Metric)
Answers: **"When the model claims an image is a 5, how confident can I be that it is actually a 5?"**

$$ \text{Precision} = \frac{TP}{TP + FP} $$

*   **High Precision:** The model is conservative. It rarely cries "Wolf!". When it guesses 5, it is almost certainly a 5.
*   **Low Precision:** The model is "trigger happy". It guesses 5 for many things that aren't 5.

### 3.3 Recall (The "Dragnet" Metric)
Answers: **"Out of all the actual 5s that exist in the dataset, what fraction did the model manage to find?"**

$$ \text{Recall} = \frac{TP}{TP + FN} $$

*   **High Recall:** The model finds nearly every 5. It doesn't miss much.
*   **Low Recall:** The model is oblivious. There are many 5s right in front of it, but it thinks they are 6s or 8s.

### 3.4 F1-Score (The Balance)
Precision and Recall are often a trade-off.
*   If you predict *everything* is a 5, you get 100% Recall (you found all of them!) but terrible Precision.
*   If you predict only one image is a 5 (and you are right), you get 100% Precision, but terrible Recall.

The **F1-Score** is the **Harmonic Mean** of the two. It punishes extreme values.

$$ F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$

---

## 4. Code Implementation Deep Dive

The `print_classification_report` method uses **Boolean Masking** to calculate these stats efficiently without slow Python loops.

### Step 1: Create Masks
For every class (loop `i` from 0 to 9), we generate boolean arrays:
```python
# Is the prediction Class i?
pred_mask = (y_pred == i) 
# Is the actual label Class i?
true_mask = (y_true == i)
```

### Step 2: Bitwise Logic (Calculate TP, FP, FN)
We use the `&` (AND) operator and `~` (NOT) operator to find intersections.
```python
# Intersection: Prediction IS i AND Truth IS i
tp = np.sum(pred_mask & true_mask)

# Intersection: Prediction IS i BUT Truth IS NOT i
fp = np.sum(pred_mask & (~true_mask))

# Intersection: Prediction IS NOT i BUT Truth IS i
fn = np.sum((~pred_mask) & true_mask)
```

### Step 3: Zero Division Safety
If the model never predicts a class (e.g., it never guesses "9"), then `TP + FP` is 0. Division by zero crashes the program.
```python
# Python ternary operator for safety
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
```

---

## 5. Usage Example

You do not need to instantiate this class. It uses `@staticmethod`, so you can call it directly.

```python
from classification_metrics import ClassificationMetrics
import numpy as np

# Mock Data
y_true = np.array([0, 1, 2, 2, 0])
y_pred = np.array([0, 0, 2, 2, 1])

# 1. Get Accuracy
acc = ClassificationMetrics.accuracy(y_true, y_pred)
print(f"Accuracy: {acc}") # Output: 0.6 (3/5 correct)

# 2. Get Full Report
ClassificationMetrics.print_classification_report(y_true, y_pred, num_classes=3)
```

**Output:**
```text
Classification report (per-class precision/recall/f1):
Class      Precision  Recall     F1-Score  
---------------------------------------------
0          0.5000     0.5000     0.5000    
1          0.0000     0.0000     0.0000    
2          1.0000     1.0000     1.0000    
```