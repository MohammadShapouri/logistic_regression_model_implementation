# Gradient Descent
### 1. The Analogy: The Hiker in the Fog

Imagine you are hiking on a large mountain range at night. There is thick fog, so you cannot see the landscape around you.
*   **Your Position:** Represents the current values of your **Weights ($W$)**.
*   **Your Altitude:** Represents the **Loss ($J$)**. The higher you are, the higher the error.
*   **The Goal:** You want to reach the absolute bottom of the valley (Loss = 0), where the model is perfect.

**How do you get down if you can't see?**
You feel the ground under your feet.
1.  **The Gradient:** You realize the ground slopes **upwards** to your right. This is the **Gradient**. It points up the steepest hill.
2.  **The Update:** Since you want to go *down*, you turn 180 degrees and take a step to the **left** (opposite the gradient).

If you keep doing this—feeling the slope and stepping the opposite way—you will eventually reach the bottom of the valley.

---

### 2. The Math: Why do we subtract?

The update formula in the code is:
$$ W_{new} = W_{old} - \text{learning\_rate} \times \text{gradient} $$

*   **Gradient ($\frac{\partial J}{\partial W}$):** Mathematically, the derivative points in the direction of the **steepest increase**. If you add the gradient, the Loss ($J$) gets bigger.
*   **Minus Sign ($- $):** We want the Loss to get smaller. Therefore, we **subtract** the gradient to go in the direction of the **steepest decrease**.

---

### 3. A Numerical Example

Let's simplify the MNIST problem to a tiny problem with just **1 Weight ($w$)**.

Suppose our Loss function is a simple curve:
$$ J(w) = w^2 $$
*(We know the minimum of this curve is at $w=0$, where Loss=0. But the computer doesn't know that yet.)*

**The Setup:**
*   **Current Weight ($w_{old}$):** `3`
*   **Learning Rate ($\alpha$):** `0.1`

#### Step 1: Calculate Current Loss
$$ J = 3^2 = 9 $$
The error is high. We want to get to 0.

#### Step 2: Calculate Gradient
We need the derivative (slope) of $w^2$. Use basic calculus power rule ($2w$):
$$ \text{Gradient} = 2 \times w_{old} = 2 \times 3 = \mathbf{6} $$
*   **Analysis:** The gradient is positive (+6). This tells the computer: "If you increase $w$, the error goes UP."

#### Step 3: The Update (Gradient Descent)
We use the formula:
$$ w_{new} = w_{old} - (\alpha \times \text{Gradient}) $$
$$ w_{new} = 3 - (0.1 \times 6) $$
$$ w_{new} = 3 - 0.6 $$
$$ w_{new} = \mathbf{2.4} $$

#### Step 4: Verify Improvement
Let's check the Loss with our new weight ($2.4$):
$$ J = 2.4^2 = \mathbf{5.76} $$

**Result:**
*   **Old Loss:** 9
*   **New Loss:** 5.76
*   **Conclusion:** The error went down! We moved closer to the target ($w=0$).

---

### 4. What happens if we do it again?

Let's take another step from $w = 2.4$.

1.  **Gradient:** $2 \times 2.4 = 4.8$
2.  **Update:** $w_{new} = 2.4 - (0.1 \times 4.8)$
3.  **Math:** $2.4 - 0.48 = \mathbf{1.92}$
4.  **New Loss:** $1.92^2 \approx \mathbf{3.68}$

The Loss dropped from **9** $\to$ **5.76** $\to$ **3.68**.
As we repeat this loop (Epochs), the weight $w$ will eventually reach `0.0`, and the Loss will reach `0.0`.

### Summary in Context of Your Code
In your MNIST code, you don't have just one $w$, you have **784** of them (one for each pixel).
The line:
```python
self.weight_matrix -= self.learning_rate * dW
```
Performs this exact "step down the hill" logic for **all 784 weights simultaneously**.