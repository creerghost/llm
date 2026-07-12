# Gradient Descent — How Neural Networks Learn

**Source:** 3Blue1Brown · Deep Learning, Chapter 2

---

## Timestamps

| Time | Topic |
|-------|-------------------------------|
| 0:00 | Introduction |
| 0:30 | Recap |
| 1:49 | Using training data |
| 3:01 | Cost functions |
| 6:55 | Gradient descent |
| 11:18 | More on gradient vectors |
| 12:19 | Gradient descent recap |
| 13:01 | Analyzing the network |
| 16:37 | Learning more |
| 17:38 | Lisha Li interview |
| 19:58 | Closing thoughts |

---

## 1. Quick Recap: Network Structure

- Input: **784 neurons** (28×28 pixel grid, each pixel a grayscale value 0–1)
- Two hidden layers of **16 neurons** each
- Output: **10 neurons** (one per digit 0–9)
- Each neuron computes: **weighted sum of previous activations + bias**, then passed through an activation function (sigmoid or ReLU)
- Total tuneable parameters: **~13,000** weights and biases

---

## 2. Training Data & The Learning Goal

- The network is shown **labelled training images** (MNIST dataset — tens of thousands of handwritten digits)
- Goal: adjust all 13,000 weights and biases so the network's performance improves
- **Generalization** is tested by evaluating on *unseen* labelled data after training

> "As provocative as it is to describe a machine as *learning*, once you see how it works, it feels a lot less like some crazy sci-fi premise, and a lot more like a calculus exercise."

---

## 3. The Cost Function

- Weights and biases start **randomly initialized** → network performs terribly at first
- **Cost of a single example:**
  - Feed an image through the network and look at the 10 output activations
  - Sum the **squared differences** between each output activation and the desired value (1 for the correct digit, 0 for all others)
  - Small cost → confident correct classification; large cost → network is lost

- **Overall cost function** = **average cost** across all training examples
  - Input: the 13,000 weights and biases
  - Output: a single number measuring "how lousy" the network is
  - Defined in terms of the network's performance over *all* training data

> "Just telling the computer what a crappy job it's doing isn't very helpful. You want to tell it *how* to change those weights and biases so that it gets better."

---

## 4. Gradient Descent

### The Core Idea (1D case)
- For a function with one input, start at any point and **check the slope**
  - Slope positive → step **left**
  - Slope negative → step **right**
- Repeat: you approach a **local minimum**
- Making step size **proportional to the slope** prevents overshooting (steps shrink as you near the minimum)
- **Caveat:** you may land in a local minimum that isn't the *global* minimum — depends on your starting point

### Scaling to Multiple Dimensions
- With 2 inputs: instead of "slope," ask **which direction in input space decreases the output fastest** → this is the **negative gradient**
- The **gradient** of a function points in the direction of **steepest ascent**; its negative points **downhill**
- The **magnitude** of the gradient indicates how steep that steepest slope is

### Applying to the Network (13,000 dimensions)
- Organize all 13,000 weights and biases into one giant vector
- Compute the **negative gradient of the cost function** — a vector pointing in the direction that decreases cost fastest
- Nudge all parameters in that direction, then repeat

> "Each component of the negative gradient tells us two things: the **sign** tells us whether to nudge up or down, and the **relative magnitudes** tell us which changes matter more."

### The Algorithm
1. Compute the gradient of the cost function
2. Take a small step in the **negative gradient** direction
3. Repeat until convergence

- This is called **gradient descent** — converging toward a local minimum of the cost function
- The efficient algorithm for computing this gradient is **backpropagation** (covered in Chapter 3)

---

## 5. Why Smooth Activations Matter

- For gradient descent to work, the cost function needs to be **smooth** (differentiable)
- This is why artificial neurons use **continuous activations** (sigmoid, ReLU) rather than binary on/off states like biological neurons

---

## 6. Interpreting the Gradient Vector

- The gradient encodes the **relative importance** of each weight and bias — which changes give the most "bang for your buck"
- Example: gradient = `(3, 1)` means:
  - Changes to the first variable are **3× more impactful** than the second
  - The direction `(3, 1)` is steepest ascent; `(-3, -1)` is steepest descent

### Summary of layers of abstraction:
| Layer | Input | Output |
|-------|-------|--------|
| **Network** | 784 pixel values | 10 activations |
| **Cost function** | ~13,000 weights & biases | Single "lousiness" score |
| **Gradient** | ~13,000 weights & biases | Direction of fastest cost change |

---

## 7. How Well Does It Actually Perform?

- The 2-hidden-layer (16 neurons each) network achieves **~96% accuracy** on unseen images
- With tweaks to architecture: up to **~98%**
- This is a "plain vanilla" network — more sophisticated architectures do better, but the baseline result is already impressive

---

## 8. What Are the Hidden Layers Really Doing?

- **Hope:** second layer detects edges → third layer detects patterns (loops, lines) → output recognizes digits
- **Reality (for this simple network):** the learned weight patterns look **almost random** with only vague loose patterns — not clean edge detectors
- The network found a local minimum that classifies well but doesn't learn interpretable features

### The Random Input Test
- Feed in **random noise** → the network **confidently outputs a specific digit** instead of expressing uncertainty
- The network can *recognize* digits but has **no idea how to draw them**
- This is because training is narrowly constrained: all images are centred, clearly defined digits in a tiny grid

> "Even if this network can recognize digits pretty well, it has no idea how to draw them."

---

## 9. Insights from Research (Lisha Li)

### Memorization vs. Learning (Zhang et al.)
- A deep network trained on **randomly shuffled labels** can still achieve perfect *training* accuracy
- It simply **memorizes** the random assignments — the millions of weights are enough capacity
- Raises the question: does minimizing cost actually learn image structure?

### Structured Data is Easier to Learn
- On **random labels**, the training accuracy curve decreases **slowly and linearly** — struggling to find the right local minimum
- On **properly labelled data**, accuracy drops fast then quickly converges — finding the minimum is **much easier**
- This suggests the network *is* doing something smarter than raw memorization

### Quality of Local Minima
- Research shows that for structured datasets, the local minima networks tend to find are of **roughly equal quality**
- Structured data → good local minima are easier to reach

---

## Key Takeaways

1. **Learning = minimizing a cost function.** The cost function measures how badly the network performs across all training data.
2. **Gradient descent** is the optimization algorithm: repeatedly nudge parameters in the direction that decreases cost fastest (negative gradient).
3. The **gradient** encodes both direction *and* relative importance of each parameter change.
4. **Backpropagation** is the efficient method for computing this gradient (next chapter).
5. Smooth, continuous activation functions are essential — they make the cost function differentiable.
6. This simple network achieves ~96% accuracy but learns **random-looking features**, not the clean edge/pattern detectors we'd hope for.
7. Modern research shows: networks can memorize random data, but **structured data is learned faster**, suggesting real learning goes beyond memorization.
8. This "vanilla" architecture is the **foundation** — understanding it is a prerequisite for modern deep learning.
