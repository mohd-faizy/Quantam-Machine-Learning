"""Build and simulate a small Cirq circuit."""


def main() -> None:
    try:
        import cirq
    except ImportError as exc:
        raise SystemExit("Install dependencies from the repo root: pip install -r requirements.txt") from exc

    q0, q1 = cirq.GridQubit.rect(1, 2)
    circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.measure(q0, q1, key="bell"),
    )

    result = cirq.Simulator().run(circuit, repetitions=20)
    print(circuit)
    print(result.histogram(key="bell"))


if __name__ == "__main__":
    main()

