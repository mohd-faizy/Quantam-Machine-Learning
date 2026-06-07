"""Build a small data re-uploading circuit for QML experiments."""


def build_reuploading_circuit(layers: int = 2):
    try:
        import cirq
        import sympy
    except ImportError as exc:
        raise SystemExit("Install dependencies from the repo root: pip install -r requirements.txt") from exc

    qubit = cirq.GridQubit(0, 0)
    circuit = cirq.Circuit()
    feature_symbols = []
    weight_symbols = []

    for layer in range(layers):
        x_symbol = sympy.Symbol(f"x_{layer}")
        w_symbol = sympy.Symbol(f"w_{layer}")
        feature_symbols.append(x_symbol)
        weight_symbols.append(w_symbol)
        circuit.append(cirq.rx(x_symbol)(qubit))
        circuit.append(cirq.ry(w_symbol)(qubit))

    return circuit, feature_symbols, weight_symbols


def main() -> None:
    circuit, features, weights = build_reuploading_circuit()
    print(circuit)
    print(f"Feature symbols: {features}")
    print(f"Weight symbols: {weights}")


if __name__ == "__main__":
    main()

