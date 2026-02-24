# 🧠 Deep Learning Foundations

A comprehensive series of **17 Jupyter notebooks** covering deep learning from the ground up — math, frameworks, and neural networks — each paired with a video walkthrough.

---

## 📚 Notebooks & Video Tutorials

### 1. NumPy Foundations for Deep Learning
- **Notebook:** `1_numpy_foundations_for_deep_learning.ipynb`
- **Video:** [Watch the video](https://youtu.be/CKYMAUN-IMw)
- **Topics:** Array creation, matrix multiplication, broadcasting, indexing, linear algebra basics with NumPy

---

### 2. PyTorch Tensors — From Zero to Hero
- **Notebook:** `2_pytorch_tensors_from_zero_to_hero.ipynb`
- **Video:** [Watch on YouTube](https://youtu.be/6fSNeE12880)
- **Topics:** Tensor creation and attributes, GPU operations, einsum notation, autograd fundamentals

---

### 3. TensorFlow Tensor Operations Tutorial
- **Notebook:** `3_tensorflow_tensor_operations_tutorial.ipynb`
- **Video:** [Watch on YouTube](https://youtu.be/OJIZ67i3OD8)
- **Topics:** TensorFlow tensors, broadcasting rules, einsum operations, comparison with NumPy

---

### 4. JAX Deep Learning Tutorial
- **Notebook:** `4_jax_deep_learning_tutorial.ipynb`
- **Video:** [Watch on YouTube](https://youtu.be/kj5-jYy3S3U)
- **Topics:** JAX fundamentals, functional programming style, immutable arrays, PRNG, `jit`, `grad`, `vmap` transformations

---

### 5. Calculus for Deep Learning
- **Notebook:** `5_calculus_for_deep_learning.ipynb`
- **Video:** [Watch on YouTube](https://youtu.be/zqxdF_21JBE)
- **Topics:** Derivatives and gradients, chain rule, backpropagation step-by-step, the training loop

---

### 6. Probability Fundamentals for Deep Learning
- **Notebook:** `6_probability_fundamentals_for_deep_learning.ipynb`
- **Video:** [Watch on YouTube](https://drive.google.com/file/d/15x4vrK-hKMm5BusGQSAHz5-ULUgQOkWz/view?usp=sharing)
- **Topics:** Probability basics, softmax, cross-entropy loss, entropy, Bayes' theorem applied to neural networks

---

### 7. Probability for Deep Learning
- **Notebook:** `7_probability_for_deep_learning.ipynb`
- **Video:** [Watch on YouTube](https://drive.google.com/file/d/15X9F2N04D0c-LL80-Ob6agPJS33LqbqR/view?usp=sharing)
- **Topics:** Expected value, common distributions, KL divergence, VAE reparameterization trick

---

### 8. Linear Algebra for Deep Learning
- **Notebook:** `8_linear_algebra_for_deep_learning.ipynb`
- **Video:** [Watch on YouTube](https://drive.google.com/file/d/1GL4zh0Ldd2IsvEw0kX3o3rJANlty1BuQ/view?usp=sharing)
- **Topics:** Vectors & matrices, dot products, norms, eigenvalues/eigenvectors, attention mechanism foundations

---

### 9. Neural Networks from Scratch
- **Notebook:** `9_neural_networks_from_scratch.ipynb`
- **Video:** [Watch on YouTube](https://youtu.be/flydJXj3Ta0)
- **Topics:** Building a full neural network using only NumPy, forward pass, backpropagation, SGD & Adam optimizer

---

### 10. Why Deep Learning Works — Geometric Intuition
- **Notebook:** `10_why_deep_learning_works_geometric_intuition.ipynb`
- **Video:** [Watch on YouTube](https://youtu.be/gpMSvRfdy9E)
- **Topics:** Universal approximation theorem, ReLU as space folding, decision boundary visualization, deep vs. wide networks

---

### 11. PyTorch Neural Networks Tutorial
- **Notebook:** `11__pytorch_neural_networks_tutorial.ipynb`
- **Video:** [Watch on YouTube](https://youtu.be/tEZchLR2bJc)
- **Topics:** `nn.Module`, autograd in practice, full training loops, loss functions, einsum in PyTorch

---

### 12. PyTorch Advanced Tutorial
- **Notebook:** `12_pytorch_advanced_tutorial.ipynb`
- **Video:** [Watch on YouTube](https://youtu.be/gYOVdy2A_28)
- **Topics:** Custom autograd functions, hooks, Jacobian/Hessian, spectral normalization, transformer building blocks

---

### 13. Keras & TensorFlow Neural Networks Tutorial
- **Notebook:** `13_keras_tensorflow_neural_networks_tutorial.ipynb`
- **Video:** [Watch on YouTube](https://youtu.be/UN6jNuR0cqM)
- **Topics:** Keras Sequential, Functional & Subclass APIs, GradientTape, full training pipeline

---

### 14. Keras & TensorFlow Advanced Tutorial
- **Notebook:** `14_keras_tensorflow_advanced_tutorial.ipynb`
- **Video:** [Watch on YouTube](<!-- ADD YOUR VIDEO LINK HERE -->)
- **Topics:** Custom gradients, convolution layers from scratch, attention mechanisms, residual blocks

---

### 15. NumPy Foundations — Extended
- **Notebook:** `15_numpy_foundations_for_deep_learning.ipynb`
- **Video:** [Watch on YouTube](https://youtu.be/vh_MnsBxm6E)
- **Topics:** Extended NumPy concepts, advanced indexing, matrix multiplication deep dive for neural networks

---

### 16. JAX Deep Learning Tutorial (Advanced)
- **Notebook:** `16_colab_5_jax_deep_learning_tutorial.ipynb`
- **Video:** [Watch on YouTube](https://youtu.be/s5eIk5Z4GCU)
- **Topics:** Advanced JAX operations, `vmap` for batching, `jit` compilation internals, neural network implementation

---

### 17. JAX Neural Networks Tutorial (Flax & Optax)
- **Notebook:** `17_colab_4_jax_neural_networks_tutorial.ipynb`
- **Video:** [Watch on YouTube](https://youtu.be/DsfTCcHuUtg)
- **Topics:** Flax neural network modules, Optax optimizers, full JAX training loop, einsum patterns, NumPy vs JAX vs Flax comparison

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/ArshanBhanage/Deep-Learning-Foundations.git
cd Deep-Learning-Foundations

# 2. Install all dependencies
pip install numpy matplotlib seaborn scipy torch tensorflow jax flax optax scikit-learn

# 3. Open any notebook in Jupyter or Google Colab
jupyter notebook
```

---

## 🗺️ Learning Path

```
FOUNDATIONS (Start Here)
    ├── Colab 1  → NumPy Foundations
    ├── Colab 15 → NumPy Extended
    │
MATH FOR DL
    ├── Colab 5  → Calculus
    ├── Colab 6  → Probability Fundamentals
    ├── Colab 7  → Probability Advanced
    └── Colab 8  → Linear Algebra
    │
FRAMEWORKS — TENSORS
    ├── Colab 2  → PyTorch Tensors
    ├── Colab 3  → TensorFlow Tensors
    ├── Colab 4  → JAX Intro
    └── Colab 16 → JAX Advanced
    │
NEURAL NETWORKS
    ├── Colab 9  → Neural Networks from Scratch
    ├── Colab 10 → Geometric Intuition (Why DL Works)
    ├── Colab 11 → PyTorch Neural Networks
    ├── Colab 12 → PyTorch Advanced
    ├── Colab 13 → Keras/TF Neural Networks
    ├── Colab 14 → Keras/TF Advanced
    └── Colab 17 → JAX Neural Networks (Flax)
```

---

## 🛠️ Topics Covered

| Category | Topics |
|----------|--------|
| **Foundations** | NumPy, array ops, broadcasting, indexing |
| **Mathematics** | Calculus, Linear Algebra, Probability |
| **Frameworks** | PyTorch, TensorFlow/Keras, JAX |
| **Neural Networks** | From scratch, architecture design, optimization |
| **Advanced** | Custom autograd, attention, transformers, spectral norm, Flax |

---

## 📎 Notes

- Each notebook has a **▶ Open in Colab** button at the top for one-click execution
- Notebooks are self-contained with explanations, visualizations, and runnable code
- Video links above correspond to the recorded walkthroughs — add your own recording links where marked `<!-- ADD YOUR VIDEO LINK HERE -->`
