# TensorFlow Quantum

TensorFlow Quantum (TFQ) lets you use Cirq quantum circuits inside TensorFlow/Keras models. It is useful when you want a familiar deep-learning workflow, but the model includes a parameterized quantum circuit.

## Setup

Install the repository dependencies from the project root:

```bash
pip install -r requirements.txt
```

The pinned stack in this repository uses:

```text
tensorflow==2.18.1
tensorflow-quantum==0.7.6
cirq
sympy
```

## Learning Path

| Step | File | What it teaches |
|---:|---|---|
| 1 | `01_cirq_circuit_basics.py` | Build and simulate a small Cirq circuit before adding TensorFlow. |
| 2 | `02_tfq_circuit_tensor.py` | Convert Cirq circuits into tensors that TensorFlow can pass through a model. |
| 3 | `03_expectation_layer.py` | Measure expectation values from circuits with `tfq.layers.Expectation`. |
| 4 | `04_pqc_layer.py` | Use `tfq.layers.PQC` as a trainable Keras layer. |
| 5 | `05_tiny_quantum_classifier.py` | Train a minimal quantum classifier on a toy dataset. |
| 6 | `06_data_reuploading_circuit.py` | Build a small data re-uploading ansatz with feature and weight symbols. |

Run an example:

```bash
python TensorFlow-Quantum/01_cirq_circuit_basics.py
python TensorFlow-Quantum/04_pqc_layer.py
```

## TFQ Cheat Sheet

### Core Imports

```python
import cirq
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq
```

### Qubits

```python
q0 = cirq.GridQubit(0, 0)
q1 = cirq.GridQubit(0, 1)
```

Use `GridQubit` for TFQ examples because it works naturally with Cirq and Google-style circuit layouts.

### Circuits

```python
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
)
```

A Cirq circuit is the quantum program. TFQ does not replace Cirq; it uses Cirq circuits as model inputs and quantum layers.

### Symbols

```python
theta = sympy.Symbol("theta")
circuit = cirq.Circuit(cirq.ry(theta)(q0))
```

Symbols mark trainable or feedable parameters. TensorFlow supplies values for these symbols during model execution.

### Circuit Tensors

```python
circuit_tensor = tfq.convert_to_tensor([cirq.Circuit()])
```

TFQ represents circuits as TensorFlow tensors with dtype `tf.string`. This lets Keras batches contain quantum circuits.

### Observables

```python
observable = cirq.Z(q0)
```

An observable defines what the model measures. Common choices are `cirq.Z(q)`, `cirq.X(q)`, or sums/products of Pauli operators.

### Expectation Layer

```python
expectation = tfq.layers.Expectation()
values = expectation(
    circuit_tensor,
    symbol_names=[theta],
    symbol_values=tf.constant([[0.5]], dtype=tf.float32),
    operators=observable,
)
```

Use this when you want explicit control over symbols and values.

### PQC Layer

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(), dtype=tf.string),
    tfq.layers.PQC(circuit, observable),
])
```

`PQC` creates a trainable quantum layer. Keras sees it like a normal model component.

## Mental Model

```text
Cirq circuit + SymPy symbols
        |
        v
TFQ circuit tensor
        |
        v
Quantum layer measures expectation values
        |
        v
Keras loss + optimizer update parameters
```

## When To Use TensorFlow Quantum

Use TFQ when:

- You already know TensorFlow/Keras.
- You want quantum circuits inside a neural-network training loop.
- Your circuits are naturally written in Cirq.
- You want to prototype hybrid quantum-classical models.

Use Qiskit or PennyLane instead when:

- You mainly want hardware-provider workflows and circuit transpilation.
- You want very compact differentiable quantum examples.
- You want Torch/JAX-first workflows.

## Common Mistakes

| Mistake | Fix |
|---|---|
| Passing a Cirq circuit directly to a Keras model | Convert it with `tfq.convert_to_tensor`. |
| Forgetting `sympy.Symbol` for parameters | Use symbols for trainable or feedable gates. |
| Expecting TFQ to load classical arrays directly into qubits | Encode features as circuit operations first. |
| Measuring every qubit without a reason | Start with one observable such as `cirq.Z(readout)`. |
| Building deep circuits too early | Start with one or two qubits and shallow layers. |

## Files In This Folder

```text
01_cirq_circuit_basics.py
02_tfq_circuit_tensor.py
03_expectation_layer.py
04_pqc_layer.py
05_tiny_quantum_classifier.py
06_data_reuploading_circuit.py
TF-Q.png/
```

