"""Convert Cirq circuits into TensorFlow Quantum circuit tensors."""


def main() -> None:
    try:
        import cirq
        import tensorflow_quantum as tfq
    except ImportError as exc:
        raise SystemExit("Install dependencies from the repo root: pip install -r requirements.txt") from exc

    qubit = cirq.GridQubit(0, 0)
    circuits = [
        cirq.Circuit(),
        cirq.Circuit(cirq.X(qubit)),
        cirq.Circuit(cirq.H(qubit)),
    ]

    tensor = tfq.convert_to_tensor(circuits)
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor dtype: {tensor.dtype}")
    print("Original circuits:")
    for circuit in circuits:
        print(circuit if circuit else "empty circuit")


if __name__ == "__main__":
    main()

