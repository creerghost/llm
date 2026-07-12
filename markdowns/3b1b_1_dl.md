# But What Is a Neural Network?

**Source:** 3Blue1Brown — *Deep Learning, Chapter 1*
**Recommended reading:** Michael Nielsen, [*Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/) (free online)

---

## Timestamps

| Time | Topic |
|-------|-------|
| 0:00 | Introduction example |
| 1:07 | Series preview |
| 2:42 | What are neurons? |
| 3:35 | Introducing layers |
| 5:31 | Why layers? |
| 8:38 | Edge detection example |
| 11:34 | Counting weights and biases |
| 12:30 | How learning relates |
| 13:26 | Notation and linear algebra |
| 15:17 | Recap |
| 16:27 | Final words |
| 17:03 | ReLU vs Sigmoid |

---

## 1. The Problem: Handwritten Digit Recognition

- Your brain effortlessly recognizes a sloppy "3" rendered at **28×28 pixels**, even though different 3s activate completely different cells in your eye.
- Writing a *program* to do this is **dauntingly difficult** — this is exactly what neural networks solve.
- The running example throughout this series: a network that learns to classify handwritten digits (the classic **MNIST** task).

---

## 2. What Is a Neuron?

- A neuron is simply **a thing that holds a number** — specifically a value between 0 and 1.
- That number is called the neuron's **activation**.
- Think of a neuron as "lit up" when its activation is high.

> *"Right now when I say neuron, all I want you to think about is a thing that holds a number."*

---

## 3. Network Structure (Layers)

| Layer | Neurons | Role |
|-------|---------|------|
| **Input** | 784 (= 28×28) | One neuron per pixel; activation = grayscale value (0 = black, 1 = white) |
| **Hidden 1** | 16 | Detect low-level features (e.g., edges) |
| **Hidden 2** | 16 | Combine features into patterns (e.g., loops, lines) |
| **Output** | 10 | One per digit (0–9); highest activation = network's answer |

- The choice of **two hidden layers with 16 neurons each** is somewhat arbitrary — in practice you experiment with architecture.
- **Key principle:** activations in one layer determine activations in the next.

---

## 4. Why Layers? (Hierarchical Decomposition)

The hope is that layers decompose recognition into **sub-problems**:

1. **Layer 2** detects **small edges** (short oriented line segments).
2. **Layer 3** combines edges into **sub-components** — loops, long lines, curves.
3. **Output layer** maps combinations of sub-components to **digits**.

**Example:** A "9" = loop at top + vertical line on the right.

> *"There are all sorts of intelligent things you might want to do that break down into layers of abstraction."*

This hierarchy also applies beyond vision — e.g., speech: raw audio → sounds → syllables → words → phrases.

---

## 5. Weights, Biases, and Activation

### Weighted Sum

For a single neuron in layer 2, connected to all 784 input neurons:

1. Assign a **weight** `w_i` to each connection.
2. Compute the **weighted sum**: `Σ w_i · a_i`
3. Positive weights in a region of interest + negative weights around it → the neuron responds to an **edge** in that region.

### Bias

- A **bias** `b` is added to the weighted sum before activation: `z = Σ w_i · a_i + b`
- The bias controls **how high the weighted sum must be** before the neuron fires.
- Example: bias of −10 means the weighted sum must exceed 10 to produce meaningful activation.

### Sigmoid (Squishification)

- The sigmoid function `σ(z) = 1 / (1 + e^(−z))` squishes any real number into **(0, 1)**.
- Very negative → ≈ 0, very positive → ≈ 1.

**Putting it together for one neuron:**

```
activation = σ( w₁a₁ + w₂a₂ + ... + w₇₈₄a₇₈₄ + b )
```

- **Weights** → what pattern the neuron detects.
- **Bias** → the threshold for activation.

---

## 6. Counting Parameters

| Connection | Weights | Biases |
|------------|---------|--------|
| Input → Hidden 1 | 784 × 16 = 12,544 | 16 |
| Hidden 1 → Hidden 2 | 16 × 16 = 256 | 16 |
| Hidden 2 → Output | 16 × 10 = 160 | 10 |
| **Total** | | **≈ 13,000** |

> *"13,000 knobs and dials that can be tweaked and turned to make this network behave in different ways."*

**Learning** = finding the right setting for all 13,000 parameters so the network solves the problem.

---

## 7. Compact Notation (Linear Algebra)

The full layer transition can be written as:

```
a⁽ˡ⁾ = σ( W · a⁽ˡ⁻¹⁾ + b )
```

Where:
- **`a⁽ˡ⁻¹⁾`** — column vector of activations from the previous layer
- **`W`** — weight matrix (each row = weights for one neuron in the next layer)
- **`b`** — bias vector
- **`σ`** — sigmoid applied element-wise

This makes code **simpler and faster** — libraries heavily optimize matrix multiplication.

---

## 8. The Network as a Function

- Each neuron is really a **function** of all neurons in the previous layer.
- The entire network is one big function: **784 inputs → 10 outputs**.
- It involves 13,000 parameters, iterated matrix–vector products, and sigmoid squishing.

> *"It's an absurdly complicated function… and in a way it's kind of reassuring that it looks complicated. If it were any simpler, what hope would we have that it could take on the challenge of recognizing digits?"*

---

## 9. ReLU vs Sigmoid

| | Sigmoid | ReLU |
|---|---------|------|
| **Formula** | `σ(z) = 1/(1+e^(−z))` | `max(0, z)` |
| **Range** | (0, 1) | [0, ∞) |
| **Motivation** | Biological: neuron active/inactive | Simpler; passes threshold → identity, else 0 |
| **Usage** | Early/"old school" networks | **Standard in modern deep networks** — much easier to train |

**ReLU** = **Re**ctified **L**inear **U**nit.

---

## Key Takeaways

1. **A neuron is just a number** (its activation), not something mystical.
2. Neural networks are organized in **layers**: input → hidden → output.
3. Layers enable **hierarchical feature detection**: pixels → edges → patterns → digits.
4. Each neuron computes a **weighted sum** of previous activations, adds a **bias**, and passes the result through an **activation function**.
5. The compact form `a⁽ˡ⁾ = σ(W · a⁽ˡ⁻¹⁾ + b)` captures an entire layer transition.
6. The whole network is **just a function** — a complicated one with ~13,000 tunable parameters.
7. **Learning** (covered in Chapter 2) = finding the right values for all those weights and biases.
8. Modern networks prefer **ReLU** over sigmoid because it trains more easily.
