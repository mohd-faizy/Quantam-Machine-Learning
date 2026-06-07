"""Train a tiny TensorFlow Quantum classifier on two simple circuit classes."""


def make_dataset():
    import cirq
    import tensorflow as tf
    import tensorflow_quantum as tfq

    qubit = cirq.GridQubit(0, 0)
    circuits = [
        cirq.Circuit(),
        cirq.Circuit(cirq.X(qubit)),
    ]
    labels = tf.constant([[1.0], [-1.0]], dtype=tf.float32)
    return tfq.convert_to_tensor(circuits), labels, qubit


def build_classifier(qubit):
    import cirq
    import sympy
    import tensorflow as tf
    import tensorflow_quantum as tfq

    theta = sympy.Symbol("theta")
    model_circuit = cirq.Circuit(cirq.ry(theta)(qubit))
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(), dtype=tf.string),
            tfq.layers.PQC(model_circuit, cirq.Z(qubit)),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05), loss="mse")
    return model


def main() -> None:
    try:
        import tensorflow as tf  # noqa: F401
        import tensorflow_quantum as tfq  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install dependencies from the repo root: pip install -r requirements.txt") from exc

    x_train, y_train, qubit = make_dataset()
    model = build_classifier(qubit)
    history = model.fit(x_train, y_train, epochs=30, verbose=0)
    predictions = model(x_train).numpy().round(3)

    print(f"Final loss: {history.history['loss'][-1]:.6f}")
    print("Predictions:")
    print(predictions)


if __name__ == "__main__":
    main()

