# Backpropagation, Intuitively

**Source:** 3Blue1Brown — Deep Learning, Chapter 3
**Video:** [Backpropagation, intuitively](https://www.youtube.com/watch?v=Ilg3gGewQ5U)

---

## Timestamps

| Time | Topic |
|-------|-------------------------------|
| 0:00 | Introduction |
| 0:23 | Recap |
| 3:07 | Intuitive walkthrough example |
| 9:33 | Stochastic gradient descent |
| 12:28 | Final words |

---

## 1. Recap: Setup So Far

- We're working with a network that recognizes **handwritten digits** (MNIST).
- Architecture: **784** input neurons (pixel values) → **2 hidden layers** (16 neurons each) → **10** output neurons (one per digit).
- **Learning** = finding the weights and biases that **minimize a cost function**.
- **Cost of a single example:** sum of squared differences between the network's output and the desired output.
- **Total cost:** average cost across all training examples.
- **Gradient descent** tells us to follow the **negative gradient** of the cost function to adjust weights and biases most efficiently.

> **Backpropagation is the algorithm for computing that gradient.**

---

## 2. Interpreting the Gradient

- The gradient vector lives in a space with as many dimensions as there are weights and biases (e.g., ~13,000).
- Instead of visualizing direction in 13,000-D space, think of each component as: **how sensitive is the cost to this particular weight or bias?**
- Example: if the gradient component for weight `w₁` is **3.2** and for `w₂` is **0.1**, then the cost is **32× more sensitive** to `w₁` — a small wiggle to `w₁` affects the cost 32× more than the same wiggle to `w₂`.

---

## 3. Intuitive Walkthrough — What One Training Example Wants

### 3.1 Desired Output Changes

Consider a single training image of a **"2"** fed into an untrained network:

- The output layer produces essentially random activations (e.g., 0.5, 0.8, 0.2, …).
- We **want** the neuron for digit 2 to increase, and **all other** output neurons to decrease.
- The size of each desired nudge is **proportional to how far** the current value is from its target.

### 3.2 Three Ways to Increase a Neuron's Activation

Recall: `activation = σ(w₁a₁ + w₂a₂ + ... + b)`

To increase the digit-2 neuron's activation, you can:

1. **Increase the bias** `b`
2. **Increase the weights** `wᵢ` — especially those connected to bright (highly active) neurons in the previous layer, since they have the most influence (`wᵢ × aᵢ` is larger when `aᵢ` is large)
3. **Change the previous-layer activations** `aᵢ` — brighten neurons connected with positive weights, dim those connected with negative weights; changes proportional to weight magnitude give the most effect

> This is reminiscent of **Hebbian theory** in neuroscience: *"neurons that fire together wire together."* The biggest weight increases happen between neurons that are most active and neurons we wish to become more active.

### 3.3 Propagating Backwards

- We can't directly change activations — only weights and biases. But we can **record what changes we'd want** at each layer.
- The digit-2 output neuron wants certain changes to the second-to-last layer.
- **Every other output neuron also has desires** for that same layer (they want to become *less* active).
- All these desires are **added together**, weighted by the corresponding connection weights and by how much each output neuron needs to change.
- This gives a **list of nudges** for the second-to-last layer.
- Then **recursively apply the same process** backwards through the network — this is the "back-propagation."

### 3.4 Combining Across All Training Examples

- A single example's nudges would just incentivize the network to always output that example's label.
- So we perform backprop for **every training example**, then **average** the desired nudges.
- This averaged collection of nudges ≈ the **negative gradient** of the cost function (or something proportional to it).

---

## 4. Stochastic Gradient Descent (SGD)

Computing the true gradient requires processing **all** training examples per step — extremely slow.

**The practical solution — SGD:**

1. **Randomly shuffle** the training data
2. **Divide into mini-batches** (e.g., 100 examples each)
3. **Compute a gradient step** using only one mini-batch at a time
4. Repeat through all mini-batches

**Trade-off:**
- Each mini-batch step is **not** the exact gradient — it's an approximation
- But it's much **faster** and still gives a good-enough direction

> Think of it as *"a drunk man stumbling quickly down a hill"* vs. *"a careful man computing the exact steepest direction before each slow, precise step."* The drunk man reaches the bottom faster.

- Over many mini-batches, the network **converges towards a local minimum** of the cost function.

---

## 5. The Data Bottleneck

- Backpropagation (and ML in general) requires **lots of labeled training data**.
- MNIST is a great example: tens of thousands of handwritten digits already labeled by humans.
- In practice, **getting labeled data** is often the hardest part of a machine learning project.

---

## Key Takeaways

1. **Backpropagation computes the gradient** of the cost function with respect to every weight and bias — it tells you *how* and *how much* to adjust each parameter.
2. **For a single training example**, backprop asks: *"what nudges to weights and biases would reduce the cost most?"* — then it propagates those desired changes backwards through layers.
3. **Sensitivity matters:** adjustments are proportional to how much influence each weight has (larger activations → larger influence).
4. **Averaging over examples:** the true gradient is the average of all individual examples' desired changes.
5. **SGD makes it practical:** instead of using all examples per step, use random mini-batches for a noisy but fast approximation.
6. **Convergence:** repeated mini-batch updates lead the network toward a local minimum — a network that performs well on training data.
7. **Data is king:** the algorithm is only as good as the labeled training data you feed it.
