"""Measure circuit expectation values with tfq.layers.Expectation."""


def main() -> None:
    try:
        import cirq
        import sympy
        import tensorflow as tf
        import tensorflow_quantum as tfq
    except ImportError as exc:
        raise SystemExit("Install dependencies from the repo root: pip install -r requirements.txt") from exc

    qubit = cirq.GridQubit(0, 0)
    theta = sympy.Symbol("theta")
    circuit = cirq.Circuit(cirq.ry(theta)(qubit))
    circuit_tensor = tfq.convert_to_tensor([circuit])

    expectation = tfq.layers.Expectation()
    values = expectation(
        circuit_tensor,
        symbol_names=[theta],
        symbol_values=tf.constant([[0.0], [1.5708], [3.1416]], dtype=tf.float32),
        operators=cirq.Z(qubit),
    )

    print(circuit)
    print(values.numpy())


if __name__ == "__main__":
    main()

