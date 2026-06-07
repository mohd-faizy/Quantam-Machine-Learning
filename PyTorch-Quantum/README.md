# PyTorch Quantum Learning Track

This directory contains a progressive learning track demonstrating how to implement, simulate, and train Quantum Neural Networks (QNNs) and Variational Quantum Circuits (VQCs) using **PyTorch**.

---

## 1. PyTorch-Based Quantum Simulation

Using PyTorch as a quantum simulator allows you to treat quantum gates as differentiable tensor operations. Because PyTorch tracks all operations in a directed acyclic graph (DAG) for autograd, you can backpropagate gradients directly from the loss function (such as state infidelity) back to the variational parameters of the quantum gates. This enables efficient hybrid quantum-classical optimization loops entirely within standard machine learning pipelines.

### Why PyTorch is Ideal for QML Simulation:
* **Autograd Integration**: Parameters of quantum gates (like rotation angles $\theta$) are treated as standard PyTorch tensors with `requires_grad=True`.
* **GPU Acceleration**: Tensor operations (matrix multiplication and contractions) scale seamlessly to CUDA-enabled devices.
* **Unified Pipeline**: Quantum circuits can be directly integrated into classical deep learning architectures (e.g. feeding classical neural network outputs into quantum circuit parameters).

---

## 2. The torchquantum Ecosystem & Dependency Landscape

The **`torchquantum`** library (developed by the MIT HAN Lab) was designed to speed up quantum simulation and QNN training. It supports batched execution of quantum circuits and implements fast GPU-accelerated simulation kernels.

### The Legacy Dependency Problem
The official `torchquantum` library has not been actively updated since late 2022. It relies strictly on legacy pre-1.0 Qiskit packages (such as `qiskit-aer==0.11.0` and `qiskit==0.45.x`).
* **The Namespace Shift**: Modern Qiskit SDKs (1.0+ and 2.x+) restructured their namespaces. The legacy import path `qiskit.providers.aer` (used internally by `torchquantum` to configure noise models) no longer exists.
* **Compatibility Issue**: Installing `torchquantum` in modern Python (3.12+) or Qiskit (2.x) environments causes `ModuleNotFoundError: No module named 'qiskit.providers.aer'` when importing the library.

### Our Strategy in This Track
1. **Pure PyTorch Simulation (`01_pytorch_statevector.py`)**: To ensure you can run QML training on modern systems (including Python 3.14+), we provide a first-principles 2-qubit statevector simulator built exclusively with standard PyTorch complex tensors and autograd.
2. **Official Syntax Showcase (`02_torchquantum_vqc.py`)**: We provide a syntactically correct reference of the official `torchquantum` library. If run in an incompatible modern environment, it catches the import error and prints instructions on setting up a legacy virtual environment.
3. **Hybrid Quantum Classifier (`03_hybrid_quantum_classifier.py`)**: A complete classical-quantum network mapping features to state angles, simulating the state vector, and classifying non-linear data (solving the XOR problem) using a standard classical MLP optimizer loop.

---

## 3. Learning Track Overview

Follow the scripts in numerical order to learn PyTorch-based quantum machine learning.

| Track Number | Script Name | Core Concept | Quantum Gates Used | Target Outcome |
| :---: | :--- | :--- | :---: | :--- |
| **01** | [01_pytorch_statevector.py](01_pytorch_statevector.py) | First-Principles QML | $RY(\theta), CNOT$ | Construct a statevector simulator and train it via autograd to prepare a Bell state. |
| **02** | [02_torchquantum_vqc.py](02_torchquantum_vqc.py) | torchquantum Syntax | $RY(\theta), CNOT$ | Showcase MIT's `tq.QuantumModule` layout and provide legacy fallback setup instructions. |
| **03** | [03_hybrid_quantum_classifier.py](03_hybrid_quantum_classifier.py) | Hybrid Classification | $RY(\theta), CNOT$ | Train a hybrid classical MLP + quantum circuit classifier to solve the XOR classification problem. |

---

## 4. Quick Start

Ensure you have PyTorch installed:

```bash
pip install torch
```

To run the simulator and train the variational parameters:

```bash
python PyTorch-Quantum/01_pytorch_statevector.py
```

To view the official `torchquantum` legacy syntax and diagnostics:

```bash
python PyTorch-Quantum/02_torchquantum_vqc.py
```

To train the hybrid classical-quantum classifier on XOR data:

```bash
python PyTorch-Quantum/03_hybrid_quantum_classifier.py
```

---

## 5. Mathematical Foundations of PyTorch Simulation

A 2-qubit statevector $|\psi\rangle$ is represented as a complex tensor of shape `(2, 2)` where the indices correspond to the basis states $|00\rangle, |01\rangle, |10\rangle, |11\rangle$:

$$|\psi\rangle = \begin{pmatrix} a_{00} & a_{01} \\ a_{10} & a_{11} \end{pmatrix}$$

### Single-Qubit Rotations
A parameterized Y-rotation gate $RY(\theta)$ is defined as:

$$RY(\theta) = \begin{pmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{pmatrix}$$

In our simulator, applying $RY(\theta)$ to Qubit 0 (the row index) is implemented as a matrix multiplication:

$$|\psi'\rangle = RY(\theta) \cdot |\psi\rangle$$

Applying $RY(\theta)$ to Qubit 1 (the column index) is implemented as:

$$|\psi'\rangle = |\psi\rangle \cdot RY(\theta)^T$$

### Entanglement
The controlled-NOT gate ($CNOT$) with Qubit 0 as control and Qubit 1 as target maps the states as:

$$|00\rangle \to |00\rangle, \quad |01\rangle \to |01\rangle, \quad |10\rangle \to |11\rangle, \quad |11\rangle \to |10\rangle$$

In tensor terms, this corresponds to keeping row 0 ($Q_0 = |0\rangle$) unchanged and flipping row 1 ($Q_0 = |1\rangle$) along the second dimension (columns):

$$\text{state}[0, :] \to \text{state}[0, :]$$
$$\text{state}[1, :] \to \text{state}[1, :].\text{flip}(0)$$

### Optimization
To prepare the maximally entangled Bell state $|\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}$, we minimize the state infidelity:

$$\mathcal{L}(\theta_0, \theta_1) = 1 - \left| \langle \Phi^+ | \psi(\theta_0, \theta_1) \rangle \right|^2$$

PyTorch computes the gradients $\nabla_\theta \mathcal{L} = \left(\frac{\partial \mathcal{L}}{\partial \theta_0}, \frac{\partial \mathcal{L}}{\partial \theta_1}\right)$ and updates the angles via the Adam optimizer.

---

## 6. Suggested Study Path

```
  Step 1: First-Principles Simulator (01_pytorch_statevector.py)
                |
                v
  Step 2: Understand Autograd & Complex Tensors for Gate Multiplications
                |
                v
  Step 3: Train a Hybrid Classifier on Non-linear Data (03_hybrid_quantum_classifier.py)
                |
                v
  Step 4: Analyze torchquantum API Structure (02_torchquantum_vqc.py)
                |
                v
  Step 5: (Optional) Set up legacy Python 3.10 virtual environment to run QNNs
```

---

## 7. Glossary of Terms

* **Statevector**: A mathematical vector containing the probability amplitudes of all possible computational basis states of a quantum system.
* **Unitary Gate**: A complex square matrix representing a reversible quantum operation. Unitary matrices preserve the norm of the statevector.
* **Autograd**: PyTorch's automatic differentiation engine that records operations to automatically calculate gradients of a loss function.
* **Parameterized Gate**: A quantum gate whose operation depends on continuous variables (e.g. rotation angles).
* **Expectation Value**: The average outcome of measuring an observable many times on a quantum state.
* **Quantum Neural Network (QNN)**: A parameterized quantum circuit containing trainable gates that can represent functional relationships or classification boundaries.
