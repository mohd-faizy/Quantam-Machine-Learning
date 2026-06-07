# TensorFlow Quantum (TFQ)

This module demonstrates how to construct, execute, and train hybrid classical-quantum models by integrating **Google Cirq** circuits directly inside **TensorFlow/Keras** pipelines.

## 1. Setup & Environment Pinned Stack

TensorFlow Quantum requires a tightly coupled python environment. The pinned requirements in the repository root contain:

```text
tensorflow==2.18.1
tensorflow-quantum==0.7.6
cirq
sympy
```

### Installation
From the root of the repository, execute:
```bash
pip install -r requirements.txt
```

---

## 2. The TFQ Pipeline Mental Model

In TFQ, quantum circuits are treated as Keras layers. The typical data and parameter flow follows this sequence:

```
  +--------------------------------+
  |    Classical Input Data (x)    |
  +--------------------------------+
                  |
                  v
  +--------------------------------+
  |  Cirq Encoding (State Prep)   |
  +--------------------------------+
                  |
                  v
  +--------------------------------+
  |   Convert to TF String Tensor  |
  +--------------------------------+
                  |
                  v
  +--------------------------------+
  |  tfq.layers.PQC (QNN Layer)    |
  +--------------------------------+
                  |
                  v
  +--------------------------------+
  | Keras Loss / Parameter Update  |
  +--------------------------------+
```

---

## 3. Learning Path

Follow the scripts sequentially to learn how to bridge Cirq circuits with Keras.

| Step | Script File | Target Concept | What You Learn |
| :---: | :--- | :--- | :--- |
| **1** | [01_cirq_circuit_basics.py](01_cirq_circuit_basics.py) | Cirq Circuit Building | Creating qubits, gates, measurements, and simulators. |
| **2** | [02_tfq_circuit_tensor.py](02_tfq_circuit_tensor.py) | Circuit Serialization | Converting Cirq objects to TensorFlow string tensors. |
| **3** | [03_expectation_layer.py](03_expectation_layer.py) | Expectation Values | Measuring quantum observables from Keras inputs. |
| **4** | [04_pqc_layer.py](04_pqc_layer.py) | Trainable Layers | Using `tfq.layers.PQC` as a trainable layer. |
| **5** | [05_tiny_quantum_classifier.py](05_tiny_quantum_classifier.py)| Hybrid Classification | Training a classifier on a basic circuit dataset. |
| **6** | [06_data_reuploading_circuit.py](06_data_reuploading_circuit.py)| Data Re-uploading | Implementing the data re-uploading QNN design pattern. |

Run any script using python:
```bash
python TensorFlow-Quantum/01_cirq_circuit_basics.py
python TensorFlow-Quantum/05_tiny_quantum_classifier.py
```

---

## 4. TFQ Code Quick Reference

### Qubit Allocations
In Cirq, qubits are identified by spatial layout. For TFQ, `cirq.GridQubit` is preferred:
```python
import cirq
q0 = cirq.GridQubit(0, 0)
q1 = cirq.GridQubit(0, 1)
```

### Parameterizing Circuits (SymPy Integration)
Tunable gates are marked using symbolic variables. TensorFlow feeds numbers into these symbols during the forward and backward passes:
```python
import sympy
theta = sympy.Symbol("theta")
circuit = cirq.Circuit(cirq.ry(theta)(q0))
```

### Serializing Circuits
Keras batches expect tensor inputs. TFQ handles this by converting Cirq circuit objects into serialized string tensors:
```python
import tensorflow_quantum as tfq
circuit_tensor = tfq.convert_to_tensor([circuit])
```

### Defining Observables
Observables determine what operator is measured. These are defined using Pauli products:
```python
observable = cirq.Z(q0) * cirq.X(q1)
```

---

## 5. Common Mistakes & Best Practices

| Common Mistake | Root Cause | Recommended Fix |
| :--- | :--- | :--- |
| **Passing raw Cirq circuits to Keras** | Keras expects TensorFlow tensors. | Wrap circuits using `tfq.convert_to_tensor()`. |
| **Forgetting to define SymPy symbols** | Without symbols, TFQ cannot track gradients. | Define parameters using `sympy.Symbol("theta")`. |
| **Feeding large classical matrices directly** | Quantum states require encoding. | Build an initial feature map layer to encode inputs. |
| **Too many qubits/gates too early** | Quantum simulation scale limits classical memory. | Keep prototype models under 4 qubits and shallow. |

---

## 6. Glossary of TFQ & Cirq Terms

* **`GridQubit` / `LineQubit`**: Qubit representations in Cirq. `GridQubit` defines qubits by 2D grid coordinates $(row, col)$, whereas `LineQubit` defines qubits sequentially in a 1D line.
* **`sympy.Symbol`**: A mathematical variable used to label parameterized parameters in a Cirq circuit that can be updated during training.
* **`tfq.convert_to_tensor`**: A utility function that serializes Cirq `Circuit` or `PauliSum` objects into TensorFlow string tensors.
* **`tfq.layers.PQC`**: Parameterized Quantum Circuit layer. A Keras layer that takes circuits, outputs expectation values, and maintains trainable variables for the internal circuit parameters.
* **`tfq.layers.Expectation`**: A TFQ layer used to compute the expectation value of operators given circuit states and parameter inputs.
* **Pauli Observable**: An operator representing a measurement of the spin projections ($X, Y, Z$) on one or more qubits.
* **Keras Integration**: The capability to compile, optimize, and train quantum layers using standard classical optimizers (such as Adam or RMSprop) and loss functions.
