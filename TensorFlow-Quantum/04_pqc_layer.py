"""Create a trainable parameterized quantum circuit layer."""


def build_model():
    try:
        import cirq
        import sympy
        import tensorflow as tf
        import tensorflow_quantum as tfq
    except ImportError as exc:
        raise SystemExit("Install dependencies from the repo root: pip install -r requirements.txt") from exc

    qubit = cirq.GridQubit(0, 0)
    theta = sympy.Symbol("theta")
    circuit = cirq.Circuit(cirq.rx(theta)(qubit))

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(), dtype=tf.string),
            tfq.layers.PQC(circuit, cirq.Z(qubit)),
        ]
    )
    return model, cirq.Circuit()


def main() -> None:
    try:
        import tensorflow_quantum as tfq
    except ImportError as exc:
        raise SystemExit("Install dependencies from the repo root: pip install -r requirements.txt") from exc

    model, empty_circuit = build_model()
    prediction = model(tfq.convert_to_tensor([empty_circuit]))
    print(model.summary())
    print(f"Initial expectation: {float(prediction.numpy()[0][0]):.6f}")


if __name__ == "__main__":
    main()

