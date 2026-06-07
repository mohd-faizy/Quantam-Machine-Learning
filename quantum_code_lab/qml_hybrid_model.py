"""Hybrid QML examples using PennyLane and TensorFlow Quantum."""

from __future__ import annotations


def pennylane_vqc_prediction(x: float = 0.2, theta: float = 0.7) -> float:
    try:
        import pennylane as qml
    except ImportError as exc:
        raise SystemExit("Install PennyLane: pip install pennylane") from exc

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit(feature: float, weight: float):
        qml.RY(feature, wires=0)
        qml.RZ(weight, wires=0)
        return qml.expval(qml.PauliZ(0))

    return float(circuit(x, theta))


def tensorflow_quantum_model_summary() -> str:
    try:
        import cirq
        import sympy
        import tensorflow as tf
        import tensorflow_quantum as tfq
    except ImportError as exc:
        raise SystemExit("Install the unified stack: pip install -r requirements.txt") from exc

    qubit = cirq.GridQubit(0, 0)
    theta = sympy.Symbol("theta")
    circuit = cirq.Circuit(cirq.rx(theta)(qubit))
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(), dtype=tf.string),
            tfq.layers.PQC(circuit, cirq.Z(qubit)),
        ]
    )
    return "\n".join(line for line in model.to_json().splitlines()[:3])


if __name__ == "__main__":
    print(f"PennyLane VQC expectation: {pennylane_vqc_prediction():.6f}")
    print(tensorflow_quantum_model_summary())

