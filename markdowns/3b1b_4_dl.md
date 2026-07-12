# Backpropagation Calculus

**Source:** 3Blue1Brown — Deep Learning, Chapter 4
**Prerequisite:** Chapter 3 (intuitive walkthrough of backpropagation)

---

## Timestamps

| Time  | Topic |
|-------|-------|
| 0:00  | Introduction |
| 0:38  | The Chain Rule in networks |
| 3:56  | Computing relevant derivatives |
| 4:45  | What do the derivatives mean? |
| 5:39  | Sensitivity to weights/biases |
| 6:42  | Layers with additional neurons |
| 9:13  | Recap |

---

## 1. Setup: A Simple One-Neuron-Per-Layer Network

- Consider the simplest possible network: **one neuron per layer**.
- The network is determined by **three weights** and **three biases**.
- **Goal:** understand how sensitive the cost function `C` is to each variable, so we know which adjustments cause the most efficient decrease in cost.

### Notation

| Symbol | Meaning |
|--------|---------|
| `a^(L)` | Activation of a neuron in layer `L` |
| `a^(L-1)` | Activation of the previous layer's neuron |
| `y` | Desired output (e.g., 0 or 1) |
| `w^(L)` | Weight connecting layer `L-1` to layer `L` |
| `b^(L)` | Bias at layer `L` |
| `z^(L)` | Weighted sum before the activation function |
| `C₀` | Cost for a single training example |

> **Note:** Superscripts like `(L)` are **layer indices**, not exponents.

### Forward Pass Equations

1. **Weighted sum:** `z^(L) = w^(L) · a^(L-1) + b^(L)`
2. **Activation:** `a^(L) = σ(z^(L))` (sigmoid, ReLU, etc.)
3. **Cost (single example):** `C₀ = (a^(L) − y)²`

> Think of it as a chain: **weight + previous activation + bias → z → a → cost**.

---

## 2. The Chain Rule Applied to Networks

The central question: **What is `∂C/∂w^(L)`?** (How sensitive is the cost to small changes in the weight?)

The chain rule breaks this into three simpler ratios:

```
∂C/∂w^(L) = (∂z^(L)/∂w^(L)) · (∂a^(L)/∂z^(L)) · (∂C/∂a^(L))
```

> A tiny nudge to `w^(L)` → nudges `z^(L)` → nudges `a^(L)` → nudges `C`.

---

## 3. Computing Each Derivative

| Derivative | Formula | Intuition |
|-----------|---------|-----------|
| `∂C/∂a^(L)` | `2(a^(L) − y)` | Proportional to the error; bigger error → bigger gradient |
| `∂a^(L)/∂z^(L)` | `σ'(z^(L))` | Derivative of whatever activation function you use |
| `∂z^(L)/∂w^(L)` | `a^(L-1)` | The previous neuron's activation |

Putting it all together:

```
∂C/∂w^(L) = a^(L-1) · σ'(z^(L)) · 2(a^(L) − y)
```

> **"Neurons that fire together wire together"** — the influence of a weight nudge depends on how strongly the previous neuron is firing (`a^(L-1)`).

---

## 4. Sensitivity to Bias

For the bias derivative, the calculation is nearly identical:

```
∂C/∂b^(L) = 1 · σ'(z^(L)) · 2(a^(L) − y)
```

- The only difference: `∂z^(L)/∂b^(L) = 1` (instead of `a^(L-1)` for the weight).

---

## 5. Propagating Backwards

To see how cost depends on the **previous layer's activation**:

```
∂C/∂a^(L-1) = w^(L) · σ'(z^(L)) · 2(a^(L) − y)
```

- `∂z^(L)/∂a^(L-1) = w^(L)` — the sensitivity to the previous activation is the weight itself.
- Even though we can't directly change `a^(L-1)`, tracking this lets us **iterate the chain rule backwards** through all layers — this is the core idea of backpropagation.

---

## 6. Extending to Multiple Neurons Per Layer

When layers have **multiple neurons**, the math stays the same in structure — just with more indices.

### Updated Notation

| Symbol | Meaning |
|--------|---------|
| `a^(L)_j` | Activation of neuron `j` in layer `L` |
| `w^(L)_jk` | Weight from neuron `k` in layer `L-1` to neuron `j` in layer `L` |
| `z^(L)_j` | Weighted sum for neuron `j` in layer `L` |

> The index order `jk` matches standard **weight matrix** notation (row = destination, column = source).

### Cost with Multiple Output Neurons

```
C₀ = Σⱼ (a^(L)_j − yⱼ)²
```

### Key Difference: Multiple Paths

When computing `∂C/∂a^(L-1)_k`, neuron `k` in layer `L-1` influences the cost through **multiple neurons** in layer `L`:

```
∂C/∂a^(L-1)_k = Σⱼ (∂C/∂a^(L)_j) · (∂a^(L)_j/∂z^(L)_j) · (∂z^(L)_j/∂a^(L-1)_k)
```

- You must **sum over all paths** (all neurons `j` in layer `L` that neuron `k` connects to).
- This is the main new complexity — everything else is structurally identical to the single-neuron case.

---

## 7. Full Cost and the Gradient

- The derivative for a **single training example** gives one component.
- The **full cost function** averages over all training examples → the derivative is also **averaged** over all examples.
- Each `∂C/∂w` is one component of the **gradient vector**, which points in the direction of steepest ascent.
- **Gradient descent** repeatedly steps opposite to the gradient to minimize cost.

---

## Key Takeaways

1. **The chain rule is the engine of backpropagation** — it decomposes `∂C/∂w` into a product of simpler, local derivatives.
2. **Three key derivatives** per connection: cost w.r.t. activation, activation w.r.t. weighted sum, weighted sum w.r.t. weight.
3. **Weight sensitivity depends on input strength** — `∂z/∂w = a^(L-1)`, so strongly activated neurons have more influence on learning (neurons that fire together wire together).
4. **Bias sensitivity is simpler** — `∂z/∂b = 1`, so bias gradients don't depend on previous activations.
5. **Backward propagation works by iterating the chain rule** — computing `∂C/∂a^(L-1)` lets you recurse through all previous layers.
6. **Multiple neurons add summation, not new concepts** — each neuron influences cost through multiple paths, so you sum over those paths.
7. **The gradient vector** collects all partial derivatives (w.r.t. every weight and bias) and is used for gradient descent.
