"""TensorFlow Quantum parameterized quantum circuit example."""


def build_tfq_pqc_prediction() -> float:
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

    empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
    prediction = model(empty_circuit)
    return float(prediction.numpy()[0][0])


if __name__ == "__main__":
    print(f"tfq_expectation={build_tfq_pqc_prediction():.6f}")

