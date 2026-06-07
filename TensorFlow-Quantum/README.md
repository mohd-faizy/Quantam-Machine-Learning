# TensorFlow Quantum

TensorFlow Quantum (TFQ) connects Cirq quantum circuits with TensorFlow/Keras models. In this repository TFQ is part of the unified QML stack.

## Examples

- `../quantum_code_lab/qml_hybrid_model.py`
- `../quantum_code_lab/tensorflow_quantum_pqc.py`
- `../qml/hybrid_models/README.md`
- `../requirements.txt`

## Minimal Concept

```python
import cirq
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq

q = cirq.GridQubit(0, 0)
theta = sympy.Symbol("theta")
circuit = cirq.Circuit(cirq.rx(theta)(q))

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(), dtype=tf.string),
    tfq.layers.PQC(circuit, cirq.Z(q)),
])
```

## Notes

Use TFQ when you want Keras-native training with Cirq circuits. Use PennyLane when you want flexible autodiff backends and compact local demos.
